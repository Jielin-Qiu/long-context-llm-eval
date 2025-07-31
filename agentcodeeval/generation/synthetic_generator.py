"""
Synthetic Project Generator for AgentCodeEval

This module generates realistic multi-file software projects using LLMs.
The generated projects serve as contexts for evaluating long-context LLMs 
in software development agent scenarios.
"""

import asyncio
import json
import logging
import random
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import openai
import anthropic
import google.generativeai as genai
import boto3
import json

from ..core.config import Config

logger = logging.getLogger(__name__)


class ProjectComplexity(Enum):
    """Project complexity levels"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class ProjectDomain(Enum):
    """Software project domains"""
    WEB_APPLICATION = "web_application"
    DATA_PIPELINE = "data_pipeline"
    API_SERVICE = "api_service"
    MACHINE_LEARNING = "machine_learning"
    DESKTOP_APPLICATION = "desktop_application"
    MOBILE_APPLICATION = "mobile_application"
    GAME_ENGINE = "game_engine"
    BLOCKCHAIN = "blockchain"
    IOT_SYSTEM = "iot_system"
    FINTECH_PLATFORM = "fintech_platform"


@dataclass
class ProjectSpecification:
    """Specification for a synthetic project"""
    name: str
    description: str
    domain: ProjectDomain
    complexity: ProjectComplexity
    language: str
    target_file_count: int
    target_token_count: int
    features: List[str]
    architecture_patterns: List[str]
    dependencies: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Convert enums to strings for JSON serialization
        data['domain'] = self.domain.value
        data['complexity'] = self.complexity.value
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
    
    def __init__(self, config: Config):
        self.config = config
        self.setup_llm_clients()
        
        # Generator specialization
        self.generators = {
            "requirements": "openai",      # Best for structured requirements
            "architecture": "anthropic",   # Excellent at system design
            "implementation": "openai",    # Strong at code generation
            "scenarios": "anthropic",      # Good at realistic scenarios
            "validation": "google"         # Alternative perspective
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
                logger.info("✅ Using Claude via AWS Bedrock")
            else:
                raise Exception("AWS credentials not found")
        except Exception:
            # Fallback to direct Anthropic API
            self.anthropic_client = anthropic.AsyncAnthropic(
                api_key=self.config.api.anthropic_api_key
            )
            self.use_bedrock = False
            logger.info("✅ Using Claude via direct Anthropic API")
        
        # Google
        genai.configure(api_key=self.config.api.google_api_key)
        
        logger.info("✅ Multi-LLM generator initialized")
    
    async def generate_with_openai(self, prompt: str, system_prompt: str = None) -> str:
        """Generate content using OpenAI"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.openai_client.chat.completions.create(
            model=self.config.api.default_model_openai,
            messages=messages,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    async def generate_with_anthropic(self, prompt: str, system_prompt: str = None) -> str:
        """Generate content using Anthropic (via Bedrock or direct API)"""
        if self.use_bedrock:
            # Use AWS Bedrock with correct format from official documentation
            # Format according to: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_InvokeModel_AnthropicClaude_section.html
            
            # Build message content in correct format
            # For Claude 3 via Bedrock, include system prompt in user message if provided
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
                "max_tokens": 4000,
                "temperature": 0.7,
                "messages": messages
            }
            
            # Use correct model IDs from AWS documentation
            model_ids = [
                "anthropic.claude-3-sonnet-20240229-v1:0",    # Claude 3 Sonnet  
                "anthropic.claude-3-haiku-20240307-v1:0",     # Claude 3 Haiku 
                "anthropic.claude-v2"                         # Claude V2 (fallback)
            ]
            
            # Run Bedrock call in thread pool since it's synchronous
            import asyncio
            loop = asyncio.get_event_loop()
            
            response = None
            successful_model = None
            
            for model_id in model_ids:
                try:
                    # Special handling for Claude V2 which uses different format
                    if model_id == "anthropic.claude-v2":
                        # Claude V2 uses the older format
                        v2_body = {
                            "prompt": f"\n\nHuman: {system_prompt or ''}\n\n{prompt}\n\nAssistant:",
                            "max_tokens_to_sample": 4000,
                            "temperature": 0.7,
                            "stop_sequences": ["\n\nHuman:"]
                        }
                        request_body = json.dumps(v2_body)
                    else:
                        # Claude 3 uses the messages format
                        request_body = json.dumps(body)
                    
                    response = await loop.run_in_executor(
                        None,
                        lambda: self.bedrock_client.invoke_model(
                            modelId=model_id,
                            body=request_body,
                            contentType="application/json"
                        )
                    )
                    
                    successful_model = model_id
                    logger.debug(f"✅ Successfully used Claude model: {model_id}")
                    break
                    
                except Exception as model_error:
                    logger.debug(f"❌ Failed with model {model_id}: {model_error}")
                    continue
            
            if not response:
                raise Exception("No Claude models available on Bedrock")
            
            # Parse response according to AWS documentation
            response_body = json.loads(response['body'].read())
            
            if successful_model == "anthropic.claude-v2":
                # Claude V2 response format
                return response_body['completion']
            else:
                # Claude 3 response format
                return response_body['content'][0]['text']
        else:
            # Use direct Anthropic API
            response = await self.anthropic_client.messages.create(
                model=self.config.api.default_model_anthropic,
                max_tokens=4000,
                system=system_prompt or "You are a senior software architect.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
    
    async def generate_with_google(self, prompt: str, system_prompt: str = None) -> str:
        """Generate content using Google Gemini"""
        model = genai.GenerativeModel(self.config.api.default_model_google)
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        response = await model.generate_content_async(full_prompt)
        return response.text
    
    async def generate_with_model(self, model_type: str, prompt: str, system_prompt: str = None) -> str:
        """Generate content with specified model type, with fallback support"""
        try:
            if model_type == "openai":
                return await self.generate_with_openai(prompt, system_prompt)
            elif model_type == "anthropic":
                return await self.generate_with_anthropic(prompt, system_prompt)
            elif model_type == "google":
                return await self.generate_with_google(prompt, system_prompt)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        except Exception as e:
            logger.warning(f"Failed to generate with {model_type}: {e}")
            # Fallback to OpenAI if available
            if model_type != "openai":
                logger.info(f"Falling back to OpenAI for {model_type} request")
                try:
                    return await self.generate_with_openai(prompt, system_prompt)
                except Exception as e2:
                    logger.error(f"OpenAI fallback also failed: {e2}")
                    raise e2
            else:
                raise e


class ProjectTemplateManager:
    """Manages project templates and domain-specific patterns"""
    
    def __init__(self):
        self.domain_templates = {
            ProjectDomain.WEB_APPLICATION: {
                "features": [
                    "user_authentication", "database_integration", "api_endpoints",
                    "frontend_interface", "session_management", "data_validation",
                    "email_notifications", "file_upload", "search_functionality",
                    "admin_panel", "logging_system", "error_handling"
                ],
                "patterns": [
                    "MVC", "REST_API", "Database_ORM", "Authentication_Middleware",
                    "Component_Architecture", "Service_Layer"
                ],
                "file_types": [
                    "controllers", "models", "views", "routes", "middleware",
                    "services", "utils", "config", "tests"
                ]
            },
            ProjectDomain.DATA_PIPELINE: {
                "features": [
                    "data_ingestion", "data_transformation", "data_validation",
                    "batch_processing", "stream_processing", "data_storage",
                    "monitoring", "error_recovery", "data_quality_checks",
                    "scheduling", "parallel_processing", "data_visualization"
                ],
                "patterns": [
                    "ETL_Pipeline", "Event_Streaming", "Data_Lake", "Microservices",
                    "Observer_Pattern", "Strategy_Pattern"
                ],
                "file_types": [
                    "extractors", "transformers", "loaders", "validators",
                    "schedulers", "monitors", "config", "tests"
                ]
            },
            ProjectDomain.MACHINE_LEARNING: {
                "features": [
                    "data_preprocessing", "feature_engineering", "model_training",
                    "model_evaluation", "hyperparameter_tuning", "model_serving",
                    "experiment_tracking", "data_visualization", "model_monitoring",
                    "automated_retraining", "feature_store", "model_versioning"
                ],
                "patterns": [
                    "Pipeline_Pattern", "Factory_Pattern", "Strategy_Pattern",
                    "Observer_Pattern", "MLOps_Architecture"
                ],
                "file_types": [
                    "data", "features", "models", "training", "evaluation",
                    "serving", "utils", "config", "tests"
                ]
            }
            # Add more domains as needed
        }
    
    def get_template(self, domain: ProjectDomain, complexity: ProjectComplexity) -> Dict[str, Any]:
        """Get project template based on domain and complexity"""
        base_template = self.domain_templates.get(domain, self.domain_templates[ProjectDomain.WEB_APPLICATION])
        
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
    
    def __init__(self, config: Config):
        self.config = config
        self.llm_generator = MultiLLMGenerator(config)
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
        
        # Calculate target metrics based on complexity
        complexity_ranges = {
            ProjectComplexity.EASY: {"files": (5, 15), "tokens": (10000, 40000)},
            ProjectComplexity.MEDIUM: {"files": (15, 40), "tokens": (40000, 100000)},
            ProjectComplexity.HARD: {"files": (40, 80), "tokens": (100000, 200000)},
            ProjectComplexity.EXPERT: {"files": (80, 150), "tokens": (200000, 500000)}
        }
        
        ranges = complexity_ranges[complexity]
        target_file_count = random.randint(*ranges["files"])
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
            name=spec_data["name"],
            description=spec_data["description"],
            domain=domain,
            complexity=complexity,
            language=language,
            target_file_count=target_file_count,
            target_token_count=target_token_count,
            features=template["features"],
            architecture_patterns=template["patterns"],
            dependencies=spec_data.get("dependencies", [])
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
    
    async def save_project(self, project: SyntheticProject) -> str:
        """Save generated project to disk"""
        project_dir = self.generated_dir / f"{project.specification.name.replace(' ', '_').lower()}"
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Save project metadata
        metadata_file = project_dir / "project_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(project.to_dict(), f, indent=2)
        
        # Save individual files
        for file in project.files:
            file_path = project_dir / file.path.lstrip('/')
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                f.write(file.content)
        
        logger.info(f"Saved project to {project_dir}")
        return str(project_dir)


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