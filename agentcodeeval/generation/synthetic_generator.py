"""
Synthetic Project Generator for AgentCodeEval

This module generates realistic multi-file software projects using LLMs.
The generated projects serve as contexts for evaluating long-context LLMs 
in software development agent scenarios.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import random

import openai
import boto3
import google.generativeai as genai
from rich.console import Console
from rich.progress import Progress, TaskID

from ..core.config import Config
from ..core.task import TaskCategory, DifficultyLevel
from ..utils.rate_limiter import APIRateLimitManager


# Set up logging
logger = logging.getLogger(__name__)

def setup_generation_logging(log_file: str = None) -> logging.Logger:
    """Setup structured logging for generation process"""
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/generation_{timestamp}.log"
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Setup file handler
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Setup console handler (for terminal output)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Setup logger
    gen_logger = logging.getLogger('agentcodeeval.generation')
    gen_logger.setLevel(logging.INFO)
    gen_logger.addHandler(file_handler)
    gen_logger.addHandler(console_handler)
    
    return gen_logger


class APIError(Exception):
    """Custom exception for API errors with specific provider info"""
    def __init__(self, provider: str, error_type: str, message: str, original_error: Exception = None, should_retry: bool = True):
        self.provider = provider
        self.error_type = error_type
        self.message = message
        self.original_error = original_error
        self.should_retry = should_retry
        super().__init__(f"{provider} {error_type}: {message}")


class CriticalAuthError(Exception):
    """Critical authentication error that should stop the entire process"""
    def __init__(self, provider: str, message: str):
        self.provider = provider
        self.message = message
        super().__init__(f"🚨 CRITICAL AUTH FAILURE - {provider}: {message}")


async def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0, provider: str = "Unknown"):
    """Enhanced retry logic with exponential backoff and critical error handling"""
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            error_str = str(e).lower()
            
            # Critical auth errors - stop immediately, don't retry
            if any(pattern in error_str for pattern in [
                "auth", "unauthorized", "forbidden", "invalid api key", "api key", 
                "authentication failed", "credentials", "access denied", "token expired",
                "expiredtokenexception", "security token", "session token"
            ]):
                if "expired" in error_str or "expiredtoken" in error_str:
                    raise CriticalAuthError(provider, f"AWS session token expired: {str(e)}")
                else:
                    raise CriticalAuthError(provider, f"Authentication failed: {str(e)}")
            
            # Retryable errors
            elif any(pattern in error_str for pattern in [
                "rate limit", "too many requests", "connection", "timeout", 
                "network", "502", "503", "504", "internal error", "server error"
            ]):
                if attempt < max_retries - 1:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Final attempt failed
                    error_type = "RATE_LIMIT" if "rate limit" in error_str else "CONNECTION_ERROR"
                    raise APIError(provider, error_type, f"Max retries exceeded: {str(e)}", should_retry=False)
            
            # Unknown errors - treat as non-retryable
            else:
                raise APIError(provider, "UNKNOWN_ERROR", f"Unexpected error: {str(e)}", should_retry=False)
    
    # Should never reach here
    raise APIError(provider, "RETRY_EXHAUSTED", "All retry attempts failed", should_retry=False)


class ProjectComplexity(Enum):
    """Project complexity levels"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class ProjectDomain(Enum):
    """Software project domains with sub-categories for uniqueness"""
    # Web Applications (6 subcategories)
    WEB_ECOMMERCE = "web_ecommerce"
    WEB_SOCIAL = "web_social"
    WEB_CMS = "web_cms"
    WEB_DASHBOARD = "web_dashboard"
    WEB_BLOG = "web_blog"
    WEB_PORTFOLIO = "web_portfolio"
    
    # API Services (4 subcategories)  
    API_REST = "api_rest"
    API_GRAPHQL = "api_graphql"
    API_MICROSERVICE = "api_microservice"
    API_GATEWAY = "api_gateway"
    
    # Data Systems (5 subcategories)
    DATA_ANALYTICS = "data_analytics"
    DATA_ETL = "data_etl"
    DATA_WAREHOUSE = "data_warehouse"
    DATA_STREAMING = "data_streaming"
    DATA_LAKE = "data_lake"
    
    # ML/AI (4 subcategories)
    ML_TRAINING = "ml_training"
    ML_INFERENCE = "ml_inference"
    ML_NLP = "ml_nlp"
    ML_COMPUTER_VISION = "ml_computer_vision"
    
    # Desktop Apps (3 subcategories)
    DESKTOP_PRODUCTIVITY = "desktop_productivity"
    DESKTOP_MEDIA = "desktop_media"
    DESKTOP_DEVELOPMENT = "desktop_development"
    
    # Mobile Apps (3 subcategories)
    MOBILE_SOCIAL = "mobile_social"
    MOBILE_UTILITY = "mobile_utility"
    MOBILE_GAME = "mobile_game"
    
    # Systems/Infrastructure (4 subcategories)
    SYSTEM_MONITORING = "system_monitoring"
    SYSTEM_AUTOMATION = "system_automation"
    SYSTEM_NETWORKING = "system_networking"
    SYSTEM_SECURITY = "system_security"
    
    # Finance/Business (3 subcategories)
    FINTECH_PAYMENT = "fintech_payment"
    FINTECH_TRADING = "fintech_trading"
    FINTECH_BANKING = "fintech_banking"
    
    # Gaming (2 subcategories)
    GAME_ENGINE = "game_engine"
    GAME_SIMULATION = "game_simulation"
    
    # Blockchain (2 subcategories)
    BLOCKCHAIN_DEFI = "blockchain_defi"
    BLOCKCHAIN_NFT = "blockchain_nft"


class ProjectArchitecture(Enum):
    """Architecture patterns for additional uniqueness"""
    MONOLITHIC = "monolithic"
    MICROSERVICES = "microservices"
    SERVERLESS = "serverless"
    EVENT_DRIVEN = "event_driven"
    LAYERED = "layered"
    CLEAN_ARCHITECTURE = "clean_architecture"
    HEXAGONAL = "hexagonal"
    MVC = "mvc"
    MVVM = "mvvm"
    COMPONENT_BASED = "component_based"


class ProjectTheme(Enum):
    """Project themes for additional variation"""
    BUSINESS = "business"
    EDUCATION = "education"
    HEALTHCARE = "healthcare"
    ENTERTAINMENT = "entertainment"
    PRODUCTIVITY = "productivity"
    SOCIAL = "social"
    UTILITY = "utility"
    CREATIVE = "creative"


@dataclass
class ProjectSpecification:
    """Specification for a synthetic project"""
    unique_id: str                        # Guaranteed unique identifier
    name: str
    description: str
    domain: ProjectDomain
    complexity: ProjectComplexity
    language: str
    architecture: ProjectArchitecture     # Architecture pattern
    theme: ProjectTheme                   # Project theme
    target_file_count: int
    target_token_count: int
    features: List[str]
    architecture_patterns: List[str]
    dependencies: List[str]
    seed: int                            # Deterministic seed for LLM variation
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Convert enums to strings for JSON serialization
        data['domain'] = self.domain.value
        data['complexity'] = self.complexity.value
        data['architecture'] = self.architecture.value
        data['theme'] = self.theme.value
        return data


@dataclass 
class GeneratedFile:
    """A generated source code file"""
    path: str
    content: str
    file_type: str  # 'source', 'config', 'test', 'documentation'
    dependencies: List[str]
    complexity_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SyntheticProject:
    """A complete synthetic software project"""
    specification: ProjectSpecification
    files: List[GeneratedFile]
    file_structure: Dict[str, Any]  # Directory tree structure
    architecture_overview: str
    setup_instructions: str
    test_scenarios: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'specification': self.specification.to_dict(),
            'files': [f.to_dict() for f in self.files],
            'file_structure': self.file_structure,
            'architecture_overview': self.architecture_overview,
            'setup_instructions': self.setup_instructions,
            'test_scenarios': self.test_scenarios
        }


class MultiLLMGenerator:
    """Multi-LLM system for generating diverse, high-quality synthetic projects"""
    
    def __init__(self, config: Config, log_file: str = None):
        self.config = config
        
        # Setup logging
        self.logger = setup_generation_logging(log_file)
        self.logger.info("🚀 MultiLLMGenerator initialized")
        
        self.rate_limiter = APIRateLimitManager(config)
        self.setup_llm_clients()
        
        # Generator specialization (using 3 Elite Models)
        # ✅ OpenAI o3: 43.94s, 13,770 chars | ✅ Claude Sonnet 4: 37.82s, 15,923 chars | ✅ Gemini 2.5 Pro: Confirmed
        self.generators = {
            "requirements": "openai",      # OpenAI o3 - Best for structured requirements
            "architecture": "anthropic",   # Claude Sonnet 4 - Excellent at system design
            "implementation": "openai",    # OpenAI o3 - Strong at code generation
            "scenarios": "anthropic",      # Claude Sonnet 4 - Good at realistic scenarios
            "validation": "google"         # Gemini 2.5 Pro - Alternative perspective
        }
    
    def setup_llm_clients(self):
        """Initialize LLM API clients"""
        # OpenAI
        self.openai_client = openai.AsyncOpenAI(
            api_key=self.config.api.openai_api_key
        )
        
        # Anthropic - try AWS Bedrock first, then direct API
        self.use_bedrock = False
        try:
            # Check if AWS credentials are available
            import os
            if all(key in os.environ for key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']):
                self.bedrock_client = boto3.client(
                    'bedrock-runtime',
                    region_name='us-east-1'  # Default region
                )
                self.use_bedrock = True
                self.logger.info("✅ Using Claude via AWS Bedrock")
            else:
                raise Exception("AWS credentials not found")
        except Exception:
            # Fallback to direct Anthropic API
            self.anthropic_client = anthropic.AsyncAnthropic(
                api_key=self.config.api.anthropic_api_key
            )
            self.use_bedrock = False
            self.logger.info("✅ Using Claude via direct Anthropic API")
        
        # Google
        genai.configure(api_key=self.config.api.google_api_key)
        
        self.logger.info("✅ Multi-LLM generator initialized")
    
    async def generate_with_openai(self, prompt: str, system_prompt: str = None) -> str:
        """Generate content using OpenAI with retry logic and rate limiting"""
        
        async def _make_openai_call():
            if not self.config.api.openai_api_key:
                raise APIError("OpenAI", "AUTH_FAILED", "OpenAI API key not configured")
            
                            # Apply rate limiting
                async with await self.rate_limiter.acquire("openai"):
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})
                    
                    # Handle o3 model special API format
                    if self.config.api.default_model_openai.startswith(("o1", "o3")):
                        response = await self.openai_client.chat.completions.create(
                            model=self.config.api.default_model_openai,
                            messages=messages,
                            max_completion_tokens=50000  # o3 supports up to 100K, using 50K for safety
                        )
                    else:
                        response = await self.openai_client.chat.completions.create(
                            model=self.config.api.default_model_openai,
                            messages=messages,
                            max_tokens=32000,  # Increased for other OpenAI models
                            temperature=0.7
                        )
                
                return response.choices[0].message.content
        
        return await retry_with_backoff(_make_openai_call, provider="OpenAI o3")
    
    async def generate_with_anthropic(self, prompt: str, system_prompt: str = None) -> str:
        """Generate content using Claude Sonnet 4 via AWS Bedrock with retry logic and rate limiting"""
        
        async def _make_anthropic_call():
            if not self.use_bedrock:
                raise APIError("Claude Sonnet 4", "AUTH_FAILED", "AWS Bedrock credentials not configured properly")
            
            # Apply rate limiting
            async with await self.rate_limiter.acquire("anthropic"):
                # Build message content in correct format for Claude Sonnet 4
                user_content = prompt
                if system_prompt:
                    user_content = f"{system_prompt}\n\n{prompt}"
                
                messages = [
                    {
                        "role": "user", 
                        "content": [{"type": "text", "text": user_content}]
                    }
                ]
                
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 32000,  # Increased for Claude Sonnet 4
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "messages": messages
                }
                
                # Elite Claude Sonnet 4 model (confirmed working)
                model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
                
                # Run Bedrock call in thread pool since it's synchronous
                loop = asyncio.get_event_loop()
                request_body = json.dumps(body)
                
                response = await loop.run_in_executor(
                    None,
                    lambda: self.bedrock_client.invoke_model(
                        modelId=model_id,
                        body=request_body,
                        contentType="application/json"
                    )
                )
                
                # Parse response according to AWS documentation
                response_body = json.loads(response['body'].read())
                return response_body['content'][0]['text']
        
        return await retry_with_backoff(_make_anthropic_call, provider="Claude Sonnet 4 (AWS Bedrock)")
    
    async def generate_with_google(self, prompt: str, system_prompt: str = None) -> str:
        """Generate content using Gemini 2.5 Pro with retry logic and rate limiting"""
        
        async def _make_google_call():
            if not self.config.api.google_api_key:
                raise APIError("Gemini 2.5 Pro", "AUTH_FAILED", "Google API key not configured")
            
            # Apply rate limiting
            async with await self.rate_limiter.acquire("google"):
                # Configure generation parameters for high-quality code generation
                generation_config = genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=32000,  # Increased for Gemini 2.5 Pro
                    top_p=0.95,
                    top_k=40
                )
                
                model = genai.GenerativeModel(
                    model_name=self.config.api.default_model_google,
                    generation_config=generation_config
                )
                
                full_prompt = prompt
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                
                # Use synchronous call to avoid async issues with Gemini
                response = model.generate_content(full_prompt)
                return response.text
        
        return await retry_with_backoff(_make_google_call, provider="Gemini 2.5 Pro")
    
    async def generate_with_model(self, model_type: str, prompt: str, system_prompt: str = None) -> str:
        """Generate content with specified model type - NO FALLBACKS"""
        try:
            if model_type == "openai":
                return await self.generate_with_openai(prompt, system_prompt)
            elif model_type == "anthropic":
                return await self.generate_with_anthropic(prompt, system_prompt)
            elif model_type == "google":
                return await self.generate_with_google(prompt, system_prompt)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        except APIError as e:
            # Re-raise APIError with additional context about model assignment
            raise APIError(
                provider=e.provider,
                error_type=e.error_type,
                message=f"Model assignment '{model_type}' failed: {e.message}",
                original_error=e.original_error
            )
        except Exception as e:
            # Convert unexpected errors to APIError
            raise APIError(
                provider=f"{model_type.title()}",
                error_type="UNEXPECTED_ERROR",
                message=f"Unexpected error in {model_type}: {str(e)}",
                original_error=e
            )


class ProjectTemplateManager:
    """Manages project templates and domain-specific patterns"""
    
    def __init__(self):
        # Define base templates for major categories
        self.base_templates = {
            "web": {
                "features": [
                    "user_authentication", "database_integration", "api_endpoints",
                    "frontend_interface", "session_management", "data_validation",
                    "email_notifications", "file_upload", "search_functionality",
                    "admin_panel", "logging_system", "error_handling", "responsive_design",
                    "payment_processing", "social_login", "caching", "ssl_security"
                ],
                "patterns": [
                    "MVC", "REST_API", "Database_ORM", "Authentication_Middleware",
                    "Component_Architecture", "Service_Layer", "Repository_Pattern"
                ],
                "file_types": [
                    "controllers", "models", "views", "routes", "middleware",
                    "services", "utils", "config", "tests", "static", "templates"
                ]
            },
            "api": {
                "features": [
                    "rest_endpoints", "graphql_schema", "authentication", "rate_limiting",
                    "request_validation", "response_caching", "api_documentation",
                    "error_handling", "logging", "monitoring", "versioning", "pagination"
                ],
                "patterns": [
                    "REST_Architecture", "GraphQL_Schema", "Microservices", "API_Gateway",
                    "Repository_Pattern", "Service_Layer", "Command_Query_Separation"
                ],
                "file_types": [
                    "endpoints", "schemas", "models", "validators", "middleware",
                    "services", "utils", "config", "tests", "docs"
                ]
            },
            "data": {
                "features": [
                    "data_ingestion", "data_transformation", "data_validation",
                    "batch_processing", "stream_processing", "data_storage",
                    "monitoring", "error_recovery", "data_quality_checks",
                    "scheduling", "parallel_processing", "data_visualization"
                ],
                "patterns": [
                    "ETL_Pipeline", "Event_Streaming", "Data_Lake", "Microservices",
                    "Observer_Pattern", "Strategy_Pattern", "Pipeline_Pattern"
                ],
                "file_types": [
                    "extractors", "transformers", "loaders", "validators",
                    "schedulers", "monitors", "config", "tests", "pipelines"
                ]
            },
            "ml": {
                "features": [
                    "data_preprocessing", "feature_engineering", "model_training",
                    "model_evaluation", "hyperparameter_tuning", "model_serving",
                    "experiment_tracking", "data_visualization", "model_monitoring",
                    "automated_retraining", "feature_store", "model_versioning"
                ],
                "patterns": [
                    "Pipeline_Pattern", "Factory_Pattern", "Strategy_Pattern",
                    "Observer_Pattern", "MLOps_Architecture", "Model_Registry"
                ],
                "file_types": [
                    "data", "features", "models", "training", "evaluation",
                    "serving", "utils", "config", "tests", "experiments"
                ]
            },
            "desktop": {
                "features": [
                    "gui_interface", "file_management", "settings_configuration",
                    "plugin_system", "auto_updates", "crash_reporting", "user_preferences",
                    "keyboard_shortcuts", "drag_drop", "multi_window", "themes"
                ],
                "patterns": [
                    "MVC", "MVVM", "Observer_Pattern", "Command_Pattern", "Plugin_Architecture",
                    "Event_Driven", "State_Machine"
                ],
                "file_types": [
                    "views", "controllers", "models", "plugins", "resources",
                    "configs", "utils", "tests", "assets", "localization"
                ]
            },
            "mobile": {
                "features": [
                    "responsive_ui", "offline_sync", "push_notifications", "location_services",
                    "camera_integration", "local_storage", "biometric_auth", "social_sharing",
                    "in_app_purchases", "analytics", "crash_reporting"
                ],
                "patterns": [
                    "MVVM", "Repository_Pattern", "Observer_Pattern", "Singleton",
                    "Factory_Pattern", "Adapter_Pattern"
                ],
                "file_types": [
                    "views", "viewmodels", "models", "services", "repositories",
                    "utils", "config", "tests", "resources", "assets"
                ]
            },
            "system": {
                "features": [
                    "system_monitoring", "log_aggregation", "performance_metrics",
                    "alerting", "configuration_management", "deployment_automation",
                    "security_scanning", "backup_recovery", "load_balancing"
                ],
                "patterns": [
                    "Observer_Pattern", "Strategy_Pattern", "Command_Pattern",
                    "Chain_of_Responsibility", "Service_Mesh", "Event_Driven"
                ],
                "file_types": [
                    "monitors", "collectors", "processors", "alerts", "configs",
                    "scripts", "utils", "tests", "dashboards"
                ]
            },
            "fintech": {
                "features": [
                    "payment_processing", "transaction_management", "fraud_detection",
                    "compliance_reporting", "risk_assessment", "encryption", "audit_logging",
                    "multi_currency", "settlement", "kyc_verification", "regulatory_compliance"
                ],
                "patterns": [
                    "Saga_Pattern", "Event_Sourcing", "CQRS", "Microservices",
                    "Security_by_Design", "Audit_Trail"
                ],
                "file_types": [
                    "transactions", "payments", "security", "compliance", "audit",
                    "models", "services", "utils", "config", "tests"
                ]
            },
            "gaming": {
                "features": [
                    "game_engine", "physics_simulation", "graphics_rendering", "audio_system",
                    "input_handling", "ai_behavior", "networking", "save_system",
                    "resource_management", "scripting_system", "level_editor"
                ],
                "patterns": [
                    "Entity_Component_System", "Game_Loop", "State_Machine",
                    "Observer_Pattern", "Object_Pool", "Command_Pattern"
                ],
                "file_types": [
                    "engine", "physics", "graphics", "audio", "input", "ai",
                    "network", "scripts", "assets", "config", "tests"
                ]
            },
            "blockchain": {
                "features": [
                    "smart_contracts", "wallet_integration", "transaction_processing",
                    "consensus_mechanism", "cryptographic_functions", "token_management",
                    "defi_protocols", "nft_minting", "governance", "staking"
                ],
                "patterns": [
                    "Event_Driven", "Factory_Pattern", "Proxy_Pattern",
                    "State_Machine", "Observer_Pattern", "Strategy_Pattern"
                ],
                "file_types": [
                    "contracts", "tokens", "protocols", "wallets", "crypto",
                    "governance", "utils", "config", "tests", "migrations"
                ]
            }
        }
        
        # Map each domain subcategory to its base template
        self.domain_mapping = {
            # Web Applications
            ProjectDomain.WEB_ECOMMERCE: "web",
            ProjectDomain.WEB_SOCIAL: "web", 
            ProjectDomain.WEB_CMS: "web",
            ProjectDomain.WEB_DASHBOARD: "web",
            ProjectDomain.WEB_BLOG: "web",
            ProjectDomain.WEB_PORTFOLIO: "web",
            
            # API Services
            ProjectDomain.API_REST: "api",
            ProjectDomain.API_GRAPHQL: "api",
            ProjectDomain.API_MICROSERVICE: "api",
            ProjectDomain.API_GATEWAY: "api",
            
            # Data Systems
            ProjectDomain.DATA_ANALYTICS: "data",
            ProjectDomain.DATA_ETL: "data",
            ProjectDomain.DATA_WAREHOUSE: "data",
            ProjectDomain.DATA_STREAMING: "data",
            ProjectDomain.DATA_LAKE: "data",
            
            # ML/AI Systems
            ProjectDomain.ML_TRAINING: "ml",
            ProjectDomain.ML_INFERENCE: "ml",
            ProjectDomain.ML_NLP: "ml",
            ProjectDomain.ML_COMPUTER_VISION: "ml",
            
            # Desktop Applications
            ProjectDomain.DESKTOP_PRODUCTIVITY: "desktop",
            ProjectDomain.DESKTOP_MEDIA: "desktop",
            ProjectDomain.DESKTOP_DEVELOPMENT: "desktop",
            
            # Mobile Applications
            ProjectDomain.MOBILE_SOCIAL: "mobile",
            ProjectDomain.MOBILE_UTILITY: "mobile",
            ProjectDomain.MOBILE_GAME: "mobile",
            
            # System Infrastructure
            ProjectDomain.SYSTEM_MONITORING: "system",
            ProjectDomain.SYSTEM_AUTOMATION: "system",
            ProjectDomain.SYSTEM_NETWORKING: "system",
            ProjectDomain.SYSTEM_SECURITY: "system",
            
            # Financial Technology
            ProjectDomain.FINTECH_PAYMENT: "fintech",
            ProjectDomain.FINTECH_TRADING: "fintech",
            ProjectDomain.FINTECH_BANKING: "fintech",
            
            # Gaming & Simulation
            ProjectDomain.GAME_ENGINE: "gaming",
            ProjectDomain.GAME_SIMULATION: "gaming",
            
            # Blockchain Systems
            ProjectDomain.BLOCKCHAIN_DEFI: "blockchain",
            ProjectDomain.BLOCKCHAIN_NFT: "blockchain"
        }
    
    def get_template(self, domain: ProjectDomain, complexity: ProjectComplexity) -> Dict[str, Any]:
        """Get project template based on domain and complexity"""
        # Map the specific domain to its base template category
        template_category = self.domain_mapping.get(domain, "web")  # Default to web if not found
        base_template = self.base_templates[template_category]
        
        # Adjust template based on complexity
        complexity_multipliers = {
            ProjectComplexity.EASY: 0.5,
            ProjectComplexity.MEDIUM: 0.7,
            ProjectComplexity.HARD: 0.9,
            ProjectComplexity.EXPERT: 1.0
        }
        
        multiplier = complexity_multipliers[complexity]
        
        return {
            "features": random.sample(
                base_template["features"], 
                max(3, int(len(base_template["features"]) * multiplier))
            ),
            "patterns": random.sample(
                base_template["patterns"],
                max(2, int(len(base_template["patterns"]) * multiplier))
            ),
            "file_types": base_template["file_types"]
        }


class SyntheticProjectGenerator:
    """Main synthetic project generator"""
    
    def __init__(self, config: Config, log_file: str = None):
        self.config = config
        self.llm_generator = MultiLLMGenerator(config, log_file)
        self.template_manager = ProjectTemplateManager()
        
        # Create output directories
        self.generated_dir = Path(config.data.generated_dir)
        self.generated_dir.mkdir(parents=True, exist_ok=True)
    
    async def generate_project_specification(
        self, 
        domain: ProjectDomain, 
        complexity: ProjectComplexity,
        language: str
    ) -> ProjectSpecification:
        """Generate a detailed project specification"""
        
        template = self.template_manager.get_template(domain, complexity)
        
        # Calculate target metrics based on complexity and config constraints
        complexity_ranges = {
            ProjectComplexity.EASY: {
                "files": (self.config.data.min_files_per_project, min(self.config.data.max_files_per_project, 15)),
                "tokens": tuple(self.config.benchmark.context_ranges["easy"])
            },
            ProjectComplexity.MEDIUM: {
                "files": (max(15, self.config.data.min_files_per_project), min(self.config.data.max_files_per_project, 40)),
                "tokens": tuple(self.config.benchmark.context_ranges["medium"])
            },
            ProjectComplexity.HARD: {
                "files": (max(40, self.config.data.min_files_per_project), min(self.config.data.max_files_per_project, 80)),
                "tokens": tuple(self.config.benchmark.context_ranges["hard"])
            },
            ProjectComplexity.EXPERT: {
                "files": (max(80, self.config.data.min_files_per_project), self.config.data.max_files_per_project),
                "tokens": tuple(self.config.benchmark.context_ranges["expert"])
            }
        }
        
        ranges = complexity_ranges[complexity]
        # Ensure file count respects config constraints
        min_files = max(ranges["files"][0], self.config.data.min_files_per_project)
        max_files = min(ranges["files"][1], self.config.data.max_files_per_project)
        target_file_count = random.randint(min_files, max_files)
        target_token_count = random.randint(*ranges["tokens"])
        
        # Generate specification using LLM
        prompt = f"""
        Generate a detailed specification for a {complexity.value} {domain.value} project in {language}.
        
        Requirements:
        - Domain: {domain.value}
        - Complexity: {complexity.value}
        - Target files: {target_file_count}
        - Target tokens: {target_token_count}
        - Features to include: {', '.join(template['features'])}
        - Architecture patterns: {', '.join(template['patterns'])}
        
        Please provide:
        1. Project name (creative but realistic)
        2. Detailed description (2-3 paragraphs)
        3. List of main dependencies/libraries
        4. Brief explanation of why this complexity level is appropriate
        
        Return as JSON with keys: name, description, dependencies, complexity_justification
        """
        
        system_prompt = """You are a senior software architect specializing in designing realistic software projects for evaluation purposes. Generate specifications that would represent real-world projects of the specified complexity."""
        
        response = await self.llm_generator.generate_with_model(
            self.llm_generator.generators["requirements"],
            prompt,
            system_prompt
        )
        
        try:
            spec_data = json.loads(response)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            spec_data = {
                "name": f"{domain.value.replace('_', ' ').title()} Project",
                "description": f"A {complexity.value} {domain.value} implementation in {language}",
                "dependencies": [],
                "complexity_justification": f"Designed for {complexity.value} complexity level"
            }
        
        return ProjectSpecification(
            unique_id=f"{domain.value}-{complexity.value}-{language}-{random.randint(1000, 9999)}", # Generate a unique ID
            name=spec_data["name"],
            description=spec_data["description"],
            domain=domain,
            complexity=complexity,
            language=language,
            architecture=ProjectArchitecture.MICROSERVICES, # Default to microservices for now
            theme=ProjectTheme.BUSINESS, # Default to business for now
            target_file_count=target_file_count,
            target_token_count=target_token_count,
            features=template["features"],
            architecture_patterns=template["patterns"],
            dependencies=spec_data.get("dependencies", []),
            seed=random.randint(1, 1000000) # Add a seed for deterministic generation
        )
    
    async def generate_project_specification_unique(
        self, 
        domain: ProjectDomain, 
        complexity: ProjectComplexity,
        language: str,
        architecture: ProjectArchitecture,
        theme: ProjectTheme,
        unique_id: str,
        seed: int
    ) -> ProjectSpecification:
        """Generate a unique project specification using all uniqueness factors"""
        
        # Set deterministic seed for consistent but unique generation
        import random
        random.seed(seed)
        
        template = self.template_manager.get_template(domain, complexity)
        
        # Calculate target metrics based on complexity and config constraints
        complexity_ranges = {
            ProjectComplexity.EASY: {
                "files": (self.config.data.min_files_per_project, min(self.config.data.max_files_per_project, 15)),
                "tokens": tuple(self.config.benchmark.context_ranges["easy"])
            },
            ProjectComplexity.MEDIUM: {
                "files": (max(15, self.config.data.min_files_per_project), min(self.config.data.max_files_per_project, 40)),
                "tokens": tuple(self.config.benchmark.context_ranges["medium"])
            },
            ProjectComplexity.HARD: {
                "files": (max(40, self.config.data.min_files_per_project), min(self.config.data.max_files_per_project, 80)),
                "tokens": tuple(self.config.benchmark.context_ranges["hard"])
            },
            ProjectComplexity.EXPERT: {
                "files": (max(80, self.config.data.min_files_per_project), self.config.data.max_files_per_project),
                "tokens": tuple(self.config.benchmark.context_ranges["expert"])
            }
        }
        
        ranges = complexity_ranges[complexity]
        # Add seed-based variation to target counts for uniqueness
        min_files = max(ranges["files"][0], self.config.data.min_files_per_project)
        max_files = min(ranges["files"][1], self.config.data.max_files_per_project)
        target_file_count = random.randint(min_files, max_files)
        target_token_count = random.randint(*ranges["tokens"])
        
        # Generate specification using LLM with unique factors
        prompt = f"""
        Generate a detailed specification for a {complexity.value} {domain.value} project in {language}.
        
        Requirements:
        - Project ID: {unique_id}
        - Domain: {domain.value}
        - Complexity: {complexity.value}
        - Architecture: {architecture.value}
        - Theme: {theme.value}
        - Target files: {target_file_count}
        - Target tokens: {target_token_count}
        - Features to include: {', '.join(template['features'])}
        - Architecture patterns: {', '.join(template['patterns'])}
        
        Create a project that specifically focuses on {theme.value} applications using {architecture.value} architecture.
        The project should be distinctly different from other {domain.value} projects by emphasizing the {theme.value} aspect.
        
        Please provide:
        1. Project name (creative, unique, reflecting the {theme.value} theme)
        2. Detailed description (2-3 paragraphs, emphasizing {theme.value} and {architecture.value})
        3. List of main dependencies/libraries (appropriate for {architecture.value})
        4. Brief explanation of why this {architecture.value} approach fits this {theme.value} project
        
        Return as JSON with keys: name, description, dependencies, architecture_justification
        """
        
        system_prompt = f"""You are a senior software architect specializing in {theme.value} applications using {architecture.value} architecture. 
        Generate specifications for unique, realistic projects that would represent real-world {theme.value} software of the specified complexity.
        Each project should be distinctly different, even within the same domain, by leveraging different aspects of the {theme.value} theme and {architecture.value} patterns."""
        
        response = await self.llm_generator.generate_with_model(
            self.llm_generator.generators["requirements"],
            prompt,
            system_prompt
        )
        
        try:
            spec_data = json.loads(response)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails - make it unique based on factors
            spec_data = {
                "name": f"{theme.value.title()} {domain.value.replace('_', ' ').title()} ({architecture.value.title()})",
                "description": f"A {complexity.value} {domain.value} implementation in {language} focused on {theme.value} applications using {architecture.value} architecture.",
                "dependencies": [],
                "architecture_justification": f"Uses {architecture.value} architecture to support {theme.value} requirements with {complexity.value} complexity."
            }
        
        return ProjectSpecification(
            unique_id=unique_id,
            name=spec_data["name"],
            description=spec_data["description"],
            domain=domain,
            complexity=complexity,
            language=language,
            architecture=architecture,
            theme=theme,
            target_file_count=target_file_count,
            target_token_count=target_token_count,
            features=template["features"],
            architecture_patterns=template["patterns"],
            dependencies=spec_data.get("dependencies", []),
            seed=seed
        )
    
    async def generate_project_architecture(self, spec: ProjectSpecification) -> Tuple[Dict[str, Any], str]:
        """Generate project architecture and file structure"""
        
        prompt = f"""
        Design the complete architecture for this project:
        
        Project: {spec.name}
        Description: {spec.description}
        Language: {spec.language}
        Domain: {spec.domain.value}
        Complexity: {spec.complexity.value}
        Target files: {spec.target_file_count}
        Features: {', '.join(spec.features)}
        Patterns: {', '.join(spec.architecture_patterns)}
        
        Please provide:
        1. Complete directory structure (as nested JSON)
        2. Architectural overview (2-3 paragraphs explaining the design)
        3. Key design decisions and rationale
        
        Return as JSON with keys: file_structure, overview, design_decisions
        """
        
        system_prompt = """You are a senior software architect. Design realistic, well-structured projects that demonstrate good software engineering practices. The file structure should be appropriate for the language and domain."""
        
        response = await self.llm_generator.generate_with_model(
            self.llm_generator.generators["architecture"],
            prompt,
            system_prompt
        )
        
        try:
            arch_data = json.loads(response)
            return arch_data.get("file_structure", {}), arch_data.get("overview", "")
        except json.JSONDecodeError:
            # Fallback structure
            return {"src": {}, "tests": {}, "config": {}}, "Basic project structure"
    
    async def generate_file_content(
        self, 
        file_path: str, 
        spec: ProjectSpecification,
        file_structure: Dict[str, Any],
        dependencies: List[str] = None
    ) -> GeneratedFile:
        """Generate content for a specific file"""
        
        dependencies = dependencies or []
        
        # Determine file type
        file_type = self._classify_file_type(file_path)
        
        prompt = f"""
        Generate realistic, production-quality code for this file:
        
        File path: {file_path}
        Project: {spec.name} ({spec.domain.value})
        Language: {spec.language}
        File type: {file_type}
        Dependencies: {', '.join(dependencies)}
        
        Project context:
        - Description: {spec.description}
        - Features: {', '.join(spec.features[:5])}  # Limit for prompt size
        - Architecture patterns: {', '.join(spec.architecture_patterns)}
        
        Requirements:
        1. Write complete, functional code
        2. Include appropriate imports/dependencies
        3. Add meaningful comments and docstrings
        4. Follow language best practices
        5. Make code realistic and non-trivial
        6. Include error handling where appropriate
        
        Return only the code content, no explanations.
        """
        
        system_prompt = f"""You are an expert {spec.language} developer. Write production-quality code that is realistic, well-structured, and follows best practices. The code should be complex enough to demonstrate real-world software development."""
        
        content = await self.llm_generator.generate_with_model(
            self.llm_generator.generators["implementation"],
            prompt,
            system_prompt
        )
        
        # Calculate complexity score (simple heuristic)
        complexity_score = self._calculate_complexity_score(content)
        
        return GeneratedFile(
            path=file_path,
            content=content,
            file_type=file_type,
            dependencies=dependencies,
            complexity_score=complexity_score
        )
    
    def _classify_file_type(self, file_path: str) -> str:
        """Classify file type based on path"""
        path_lower = file_path.lower()
        
        if any(test_dir in path_lower for test_dir in ['test', 'spec', '__test__']):
            return 'test'
        elif any(config_file in path_lower for config_file in ['config', 'settings', '.env', 'package.json']):
            return 'config'
        elif any(doc_file in path_lower for doc_file in ['readme', 'doc', 'documentation']):
            return 'documentation'
        else:
            return 'source'
    
    def _calculate_complexity_score(self, content: str) -> float:
        """Calculate complexity score for file content"""
        # Simple heuristic based on content characteristics
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Factors contributing to complexity
        line_count = len(non_empty_lines)
        comment_ratio = sum(1 for line in non_empty_lines if line.strip().startswith('#') or '//' in line) / max(line_count, 1)
        avg_line_length = sum(len(line) for line in non_empty_lines) / max(line_count, 1)
        
        # Simple scoring (0-1 scale)
        score = min(1.0, (line_count / 100) * 0.6 + comment_ratio * 0.2 + (avg_line_length / 80) * 0.2)
        
        return round(score, 2)
    
    def _validate_project_constraints(self, generated_files: List[dict], spec: ProjectSpecification):
        """Validate that generated project meets configuration constraints"""
        file_count = len(generated_files)
        
        # Check file count constraints
        if file_count < self.config.data.min_files_per_project:
            logger.warning(
                f"⚠️ Project '{spec.name}' has {file_count} files, "
                f"below minimum of {self.config.data.min_files_per_project}"
            )
        elif file_count > self.config.data.max_files_per_project:
            logger.warning(
                f"⚠️ Project '{spec.name}' has {file_count} files, "
                f"above maximum of {self.config.data.max_files_per_project}"
            )
        
        # Calculate complexity scores for validation
        complexity_scores = []
        documentation_files = 0
        
        for file_info in generated_files:
            complexity_score = self._calculate_complexity_score(file_info['content'])
            complexity_scores.append(complexity_score)
            
            # Check if this is a documentation file
            if file_info['type'] == 'documentation' or any(doc_indicator in file_info['path'].lower() 
                for doc_indicator in ['readme', 'doc', 'documentation']):
                documentation_files += 1
        
        # Check average complexity constraints and filter if needed
        avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0
        
        # Filter out files that don't meet complexity requirements
        if avg_complexity < self.config.data.min_complexity_score:
            logger.warning(
                f"⚠️ Project '{spec.name}' average complexity {avg_complexity:.2f} "
                f"below minimum {self.config.data.min_complexity_score}. "
                f"Consider regenerating or adjusting constraints."
            )
        elif avg_complexity > self.config.data.max_complexity_score:
            logger.warning(
                f"⚠️ Project '{spec.name}' average complexity {avg_complexity:.2f} "
                f"above maximum {self.config.data.max_complexity_score}. "
                f"Consider simplifying or adjusting constraints."
            )
        
        # Check documentation ratio
        documentation_ratio = documentation_files / file_count if file_count > 0 else 0
        if documentation_ratio < self.config.data.min_documentation_ratio:
            logger.warning(
                f"⚠️ Project '{spec.name}' documentation ratio {documentation_ratio:.2f} "
                f"below minimum {self.config.data.min_documentation_ratio}"
            )
        
        logger.debug(
            f"✅ Project '{spec.name}' validation: {file_count} files, "
            f"complexity {avg_complexity:.2f}, docs {documentation_ratio:.2f}"
        )
    
    async def generate_complete_project(
        self, 
        domain: ProjectDomain,
        complexity: ProjectComplexity,
        language: str
    ) -> SyntheticProject:
        """Generate a complete synthetic project"""
        
        logger.info(f"Generating {complexity.value} {domain.value} project in {language}")
        
        # Step 1: Generate specification
        spec = await self.generate_project_specification(domain, complexity, language)
        logger.info(f"Generated specification for '{spec.name}'")
        
        # Step 2: Generate architecture
        file_structure, architecture_overview = await self.generate_project_architecture(spec)
        logger.info("Generated project architecture")
        
        # Step 3: Generate files
        file_paths = self._extract_file_paths(file_structure)
        files = []
        
        for file_path in file_paths:
            try:
                generated_file = await self.generate_file_content(file_path, spec, file_structure)
                files.append(generated_file)
                logger.debug(f"Generated file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to generate file {file_path}: {e}")
                continue
        
        logger.info(f"Generated {len(files)} files")
        
        # Step 4: Generate additional metadata
        setup_instructions = await self._generate_setup_instructions(spec, files)
        test_scenarios = await self._generate_test_scenarios(spec)
        
        return SyntheticProject(
            specification=spec,
            files=files,
            file_structure=file_structure,
            architecture_overview=architecture_overview,
            setup_instructions=setup_instructions,
            test_scenarios=test_scenarios
        )
    
    def _extract_file_paths(self, file_structure: Dict[str, Any], current_path: str = "") -> List[str]:
        """Extract all file paths from nested file structure"""
        paths = []
        
        for name, content in file_structure.items():
            full_path = f"{current_path}/{name}" if current_path else name
            
            if isinstance(content, dict):
                # Directory - recurse
                paths.extend(self._extract_file_paths(content, full_path))
            else:
                # File - add to paths
                paths.append(full_path)
        
        return paths
    
    async def _generate_setup_instructions(self, spec: ProjectSpecification, files: List[GeneratedFile]) -> str:
        """Generate setup and installation instructions"""
        prompt = f"""
        Generate clear setup instructions for this project:
        
        Project: {spec.name}
        Language: {spec.language}
        Dependencies: {', '.join(spec.dependencies)}
        File count: {len(files)}
        
        Include:
        1. Prerequisites and requirements
        2. Installation steps
        3. Configuration needed
        4. How to run the project
        5. Basic usage examples
        
        Keep it concise but complete.
        """
        
        return await self.llm_generator.generate_with_model(
            self.llm_generator.generators["scenarios"],
            prompt,
            "You are a technical writer creating clear, actionable setup instructions."
        )
    
    async def _generate_test_scenarios(self, spec: ProjectSpecification) -> List[str]:
        """Generate test scenarios for the project"""
        prompt = f"""
        Generate 5-10 realistic test scenarios for this project:
        
        Project: {spec.name}
        Domain: {spec.domain.value}
        Features: {', '.join(spec.features)}
        
        Each scenario should be:
        1. Specific and actionable
        2. Cover different aspects of the system
        3. Realistic for the domain
        4. Suitable for automated testing
        
        Return as JSON array of strings.
        """
        
        response = await self.llm_generator.generate_with_model(
            self.llm_generator.generators["scenarios"],
            prompt,
            "You are a QA engineer designing comprehensive test scenarios."
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return [
                "Test basic functionality",
                "Test error handling",
                "Test performance under load",
                "Test security measures",
                "Test integration points"
            ]
    
    async def save_project(self, project: SyntheticProject) -> Path:
        """Save a synthetic project to disk"""
        # Create project directory
        safe_name = project.specification.name.lower().replace(' ', '_').replace('-', '_')
        project_dir = self.generated_dir / safe_name
        project_dir.mkdir(exist_ok=True)
        
        # Save individual files
        for file in project.files:
            file_path = project_dir / file.path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(file.content)
        
        # Save project metadata
        metadata = {
            "specification": {
                "name": project.specification.name,
                "description": project.specification.description,
                "domain": project.specification.domain.value,
                "complexity": project.specification.complexity.value,
                "language": project.specification.language,
                "target_file_count": project.specification.target_file_count,
                "target_token_count": project.specification.target_token_count,
                "features": project.specification.features,
                "architecture_patterns": project.specification.architecture_patterns,
                "dependencies": project.specification.dependencies
            },
            "files": [{"path": f.path, "type": f.file_type} for f in project.files],
            "file_structure": project.file_structure,
            "architecture_overview": project.architecture_overview,
            "setup_instructions": project.setup_instructions,
            "test_scenarios": project.test_scenarios
        }
        
        with open(project_dir / "project_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved project to {project_dir}")
        return project_dir

    async def generate_project_files(self, spec_dict: dict, target_files: int, target_tokens: int) -> List[dict]:
        """Generate actual code files for a project specification from Phase 1"""
        
        # Import Rich console for progress reporting
        from rich.console import Console
        console = Console()
        
        # Convert dict back to ProjectSpecification object
        spec = ProjectSpecification(
            unique_id=f"{ProjectDomain(spec_dict['domain']).value}-{ProjectComplexity(spec_dict['complexity']).value}-{spec_dict['language']}-{random.randint(1000, 9999)}", # Generate a unique ID
            name=spec_dict['name'],
            description=spec_dict['description'],
            domain=ProjectDomain(spec_dict['domain']),
            complexity=ProjectComplexity(spec_dict['complexity']),
            language=spec_dict['language'],
            architecture=ProjectArchitecture.MICROSERVICES, # Default to microservices for now
            theme=ProjectTheme.BUSINESS, # Default to business for now
            target_file_count=target_files,
            target_token_count=target_tokens,
            features=spec_dict.get('features', []),
            architecture_patterns=spec_dict.get('architecture_patterns', []),
            dependencies=spec_dict.get('dependencies', []),
            seed=random.randint(1, 1000000) # Add a seed for deterministic generation
        )
        
        # Generate a proper file structure for this project
        file_structure, _ = await self.generate_project_architecture(spec)
        
        # Extract file paths from the structure
        file_paths = self._extract_file_paths(file_structure)
        
        # Limit to target_files if we have too many
        if len(file_paths) > target_files:
            file_paths = file_paths[:target_files]
        
        # Generate additional files if we need more
        elif len(file_paths) < target_files:
            additional_files = await self._generate_additional_files(spec, target_files - len(file_paths))
            file_paths.extend(additional_files)
        
        # Generate content for each file using our 3 Elite Models
        generated_files = []
        
        console.print(f"      🏭 Generating {len(file_paths)} files...")
        
        for i, file_path in enumerate(file_paths, 1):
            try:
                # Show progress for each file
                console.print(f"      📄 {i}/{len(file_paths)}: {file_path}", end="")
                
                generated_file = await self.generate_file_content(file_path, spec, file_structure)
                
                lines_count = len(generated_file.content.splitlines())
                chars_count = len(generated_file.content)
                
                generated_files.append({
                    'path': file_path,
                    'content': generated_file.content,
                    'type': self._classify_file_type(file_path)
                })
                
                console.print(f" ✅ ({lines_count} lines, {chars_count:,} chars)")
                
            except Exception as e:
                console.print(f" ❌ Error: {str(e)}")
                logger.error(f"Failed to generate file {file_path}: {e}")
                # Create a minimal placeholder file
                placeholder_content = f"# {spec.name}\n# TODO: Implement {file_path}\n# Error: {str(e)}\n"
                generated_files.append({
                    'path': file_path,
                    'content': placeholder_content,
                    'type': self._classify_file_type(file_path)
                })
        
        total_lines = sum(len(f['content'].splitlines()) for f in generated_files)
        total_chars = sum(len(f['content']) for f in generated_files)
        console.print(f"      🎉 [bold green]All files generated![/bold green] {len(generated_files)} files, {total_lines:,} lines, {total_chars:,} chars")
        
        # Validate project constraints
        self._validate_project_constraints(generated_files, spec)
        
        return generated_files
    
    async def _generate_additional_files(self, spec: ProjectSpecification, count: int) -> List[str]:
        """Generate additional file paths if needed to reach target file count"""
        
        lang_extensions = {
            'python': '.py',
            'javascript': '.js', 
            'typescript': '.ts',
            'java': '.java',
            'cpp': '.cpp',
            'go': '.go'
        }
        
        ext = lang_extensions.get(spec.language, '.txt')
        additional_files = []
        
        # Generate common additional files based on project type
        base_files = [
            f"src/utils{ext}",
            f"src/config{ext}",
            f"src/constants{ext}",
            f"tests/test_main{ext}",
            f"tests/test_utils{ext}",
            "README.md",
            "requirements.txt" if spec.language == 'python' else "package.json",
            ".gitignore",
            "Dockerfile",
            f"docs/api{'.md'}",
        ]
        
        # Add files until we reach the target count
        for i, file_path in enumerate(base_files):
            if i >= count:
                break
            additional_files.append(file_path)
        
        # If we still need more files, generate numbered modules
        remaining = count - len(additional_files)
        for i in range(remaining):
            additional_files.append(f"src/module_{i+1}{ext}")
        
        return additional_files


# Example usage and testing
async def main():
    """Example usage of the synthetic generator"""
    config = Config()
    generator = SyntheticProjectGenerator(config)
    
    # Generate a sample project
    project = await generator.generate_complete_project(
        domain=ProjectDomain.WEB_APPLICATION,
        complexity=ProjectComplexity.MEDIUM,
        language="python"
    )
    
    print(f"Generated project: {project.specification.name}")
    print(f"Files: {len(project.files)}")
    print(f"Architecture: {project.architecture_overview[:200]}...")
    
    # Save project
    await generator.save_project(project)


if __name__ == "__main__":
    asyncio.run(main()) 