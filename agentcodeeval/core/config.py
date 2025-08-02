"""
Configuration management for AgentCodeEval
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class APIConfig:
    """Configuration for LLM APIs"""
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None 
    google_api_key: Optional[str] = None
    huggingface_token: Optional[str] = None
    
    # Rate limiting settings 
    max_requests_per_minute: int = 600
    max_concurrent_requests: int = 60
    
    # Model configurations
    # 🏆 3 Elite Models Ready for AgentCodeEval:
    default_model_openai: str = "o3"                                  # ✅ Elite: OpenAI o3 (reasoning model)
    default_model_anthropic: str = "claude-sonnet-4-20250514"         # ✅ Elite: Claude Sonnet 4 via AWS Bedrock
    default_model_google: str = "gemini-2.5-pro"                      # ✅ Elite: Gemini 2.5 Pro (latest)


@dataclass
class DataConfig:
    """Configuration for synthetic data generation and storage"""
    # Local storage
    output_dir: str = "./data/output"
    generated_dir: str = "./data/generated"
    
    # Synthetic generation settings
    supported_languages: List[str] = field(default_factory=lambda: [
        "python", "cpp", "java", "c", "csharp", "javascript", "typescript", "go", "rust", "php"
    ])
    
    # Project generation criteria
    min_files_per_project: int = 10  # Fixed: was 5, should match config.yaml
    max_files_per_project: int = 100
    projects_per_language: int = 100  # OPTIMAL: 10 languages × 100 = 1,000 total projects
    
    # Complexity levels for synthetic projects
    complexity_distribution: Dict[str, float] = field(default_factory=lambda: {
        "easy": 0.25,      # 25% easy projects
        "medium": 0.25,    # 25% medium projects  
        "hard": 0.25,      # 25% hard projects
        "expert": 0.25     # 25% expert projects
    })
    
    # Generation quality controls
    min_complexity_score: float = 0.3
    max_complexity_score: float = 0.9
    min_documentation_ratio: float = 0.1


@dataclass 
class BenchmarkConfig:
    """Configuration for benchmark generation"""
    # Scale parameters
    total_instances: int = 8000  # OPTIMIZED: 10 languages × 100 projects × 8 categories
    
    # Task category distribution
    task_distribution: Dict[str, int] = field(default_factory=lambda: {
        "architectural_understanding": 1000,
        "cross_file_refactoring": 1000, 
        "feature_implementation": 1000,
        "bug_investigation": 1000,
        "multi_session_development": 1000,
        "code_comprehension": 1000,
        "integration_testing": 1000,
        "security_analysis": 1000
    })
    
    # Difficulty distribution
    difficulty_distribution: Dict[str, int] = field(default_factory=lambda: {
        "easy": 2000,    # 10K-100K tokens
        "medium": 2000,  # 100K-200K tokens  
        "hard": 2000,    # 200K-500K tokens
        "expert": 2000   # 500K-1M tokens
    })
    
    # Context length ranges
    context_ranges: Dict[str, tuple] = field(default_factory=lambda: {
        "easy": (10000, 100000),
        "medium": (100000, 200000),
        "hard": (200000, 500000), 
        "expert": (500000, 1000000)
    })
    
    # Information coverage requirements
    min_information_coverage: float = 0.7
    target_information_coverage: Dict[str, float] = field(default_factory=lambda: {
        "easy": 0.75,
        "medium": 0.85,
        "hard": 0.9,
        "expert": 0.95
    })


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics and scoring"""
    
    # Metric weights for CADS (Composite Agent Development Score)
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        "architectural_coherence": 0.2,
        "dependency_traversal": 0.2,
        "multi_session_memory": 0.2,
        "cross_file_reasoning": 0.15,
        "incremental_development": 0.15,
        "information_coverage": 0.1
    })
    
    # Scoring thresholds
    score_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "excellent": {"min": 4.0, "max": 5.0},
        "good": {"min": 3.0, "max": 4.0},
        "fair": {"min": 2.0, "max": 3.0}, 
        "poor": {"min": 0.0, "max": 2.0}
    })
    
    # Evaluation timeouts (seconds)
    task_timeout: int = 1800
    session_timeout: int = 3600


class Config:
    """Main configuration class for AgentCodeEval"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration from file or defaults"""
        self.api = APIConfig()
        self.data = DataConfig()
        self.benchmark = BenchmarkConfig()
        self.evaluation = EvaluationConfig()
        
        # Load from environment variables
        self._load_from_env()
        
        # Load from config file if provided
        if config_path:
            self.load_from_file(config_path)
            
        # Create necessary directories
        self._create_directories()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # API keys
        self.api.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.api.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.api.google_api_key = os.getenv("GEMINI_API_KEY")  # Using GEMINI_API_KEY for clarity
        self.api.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        
        # Override data paths if set
        if os.getenv("ACE_OUTPUT_DIR"):
            self.data.output_dir = os.getenv("ACE_OUTPUT_DIR")
    
    def load_from_file(self, config_path: Union[str, Path]):
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Update configurations
        if 'api' in config_data:
            for key, value in config_data['api'].items():
                if hasattr(self.api, key):
                    setattr(self.api, key, value)
                    
        if 'data' in config_data:
            for key, value in config_data['data'].items():
                if hasattr(self.data, key):
                    setattr(self.data, key, value)
                    
        if 'benchmark' in config_data:
            for key, value in config_data['benchmark'].items():
                if hasattr(self.benchmark, key):
                    setattr(self.benchmark, key, value)
                    
        if 'evaluation' in config_data:
            for key, value in config_data['evaluation'].items():
                if hasattr(self.evaluation, key):
                    setattr(self.evaluation, key, value)
    
    def save_to_file(self, config_path: Union[str, Path]):
        """Save current configuration to YAML file"""
        config_path = Path(config_path)
        
        config_data = {
            'api': {
                'openai_api_key': self.api.openai_api_key,
                'anthropic_api_key': self.api.anthropic_api_key,
                'google_api_key': self.api.google_api_key,
                'huggingface_token': self.api.huggingface_token,
                'max_requests_per_minute': self.api.max_requests_per_minute,
                'max_concurrent_requests': self.api.max_concurrent_requests,
                'default_model_openai': self.api.default_model_openai,
                'default_model_anthropic': self.api.default_model_anthropic,
                'default_model_google': self.api.default_model_google,
            },
            'data': {
                'output_dir': self.data.output_dir,
                'generated_dir': self.data.generated_dir,
                'supported_languages': self.data.supported_languages,
                'min_files_per_project': self.data.min_files_per_project,
                'max_files_per_project': self.data.max_files_per_project,
                'projects_per_language': self.data.projects_per_language,
                'complexity_distribution': self.data.complexity_distribution,
                'min_complexity_score': self.data.min_complexity_score,
                'max_complexity_score': self.data.max_complexity_score,
                'min_documentation_ratio': self.data.min_documentation_ratio,
            },
            'benchmark': {
                'total_instances': self.benchmark.total_instances,
                'task_distribution': self.benchmark.task_distribution,
                'difficulty_distribution': self.benchmark.difficulty_distribution,
                'context_ranges': self.benchmark.context_ranges,
                'min_information_coverage': self.benchmark.min_information_coverage,
                'target_information_coverage': self.benchmark.target_information_coverage,
            },
            'evaluation': {
                'metric_weights': self.evaluation.metric_weights,
                'score_thresholds': self.evaluation.score_thresholds,
                'task_timeout': self.evaluation.task_timeout,
                'session_timeout': self.evaluation.session_timeout,
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.data.output_dir,
            self.data.generated_dir,
            f"{self.data.output_dir}/scenarios",
            f"{self.data.output_dir}/validation",
            f"{self.data.output_dir}/references"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Check API keys (including AWS Bedrock for Claude)
        import os
        aws_configured = all([
            os.getenv('AWS_ACCESS_KEY_ID'),
            os.getenv('AWS_SECRET_ACCESS_KEY'), 
            os.getenv('AWS_SESSION_TOKEN')
        ])
        anthropic_available = self.api.anthropic_api_key or aws_configured
        
        if not any([
            self.api.openai_api_key,
            anthropic_available,  # Direct Anthropic API or AWS Bedrock
            self.api.google_api_key
        ]):
            errors.append("At least one API key must be provided (OpenAI, Anthropic/AWS Bedrock, or Google)")
            
        # Check directory permissions
        try:
            Path(self.data.output_dir).mkdir(parents=True, exist_ok=True)
            Path(self.data.generated_dir).mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            errors.append(f"Permission error creating directories: {e}")
            
        # Validate benchmark distribution
        total_distributed = sum(self.benchmark.task_distribution.values())
        if total_distributed != self.benchmark.total_instances:
            errors.append(
                f"Task distribution total ({total_distributed}) "
                f"doesn't match total_instances ({self.benchmark.total_instances})"
            )
            
        difficulty_total = sum(self.benchmark.difficulty_distribution.values())
        if difficulty_total != self.benchmark.total_instances:
            errors.append(
                f"Difficulty distribution total ({difficulty_total}) "
                f"doesn't match total_instances ({self.benchmark.total_instances})"
            )
            
        # Validate metric weights sum to 1.0
        weight_sum = sum(self.evaluation.metric_weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            errors.append(f"Metric weights sum to {weight_sum}, should be 1.0")
            
        return errors
    
    def summary(self) -> str:
        """Return a summary of the configuration"""
        return f"""
AgentCodeEval Configuration Summary
==================================

Benchmark Scale:
- Total instances: {self.benchmark.total_instances:,}
- Task categories: {len(self.benchmark.task_distribution)}
- Difficulty levels: {len(self.benchmark.difficulty_distribution)}

Context Ranges:
- Easy: {self.benchmark.context_ranges['easy'][0]:,} - {self.benchmark.context_ranges['easy'][1]:,} tokens
- Medium: {self.benchmark.context_ranges['medium'][0]:,} - {self.benchmark.context_ranges['medium'][1]:,} tokens  
- Hard: {self.benchmark.context_ranges['hard'][0]:,} - {self.benchmark.context_ranges['hard'][1]:,} tokens
- Expert: {self.benchmark.context_ranges['expert'][0]:,} - {self.benchmark.context_ranges['expert'][1]:,} tokens

Data Sources:
- Languages: {', '.join(self.data.supported_languages)}
- Files per project: {self.data.min_files_per_project} - {self.data.max_files_per_project}

API Configuration:
- OpenAI: {'✓' if self.api.openai_api_key else '✗'}
- Anthropic: {'✓' if self.api.anthropic_api_key else '✗'}
- Google: {'✓' if self.api.google_api_key else '✗'}
- HuggingFace: {'✓' if self.api.huggingface_token else '✗'}
""" 