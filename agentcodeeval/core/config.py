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
    
    # Rate limiting
    max_requests_per_minute: int = 60
    max_concurrent_requests: int = 10
    
    # Model configurations
    default_model_openai: str = "gpt-4o"
    default_model_anthropic: str = "claude-3-5-sonnet-20241022"
    default_model_google: str = "gemini-1.5-pro-latest"


@dataclass
class DataConfig:
    """Configuration for synthetic data generation and storage"""
    # Local storage
    cache_dir: str = "./data/cache"
    output_dir: str = "./data/output"
    benchmark_dir: str = "./data/benchmark"
    generated_dir: str = "./data/generated"
    templates_dir: str = "./data/templates"
    
    # Synthetic generation settings
    supported_languages: List[str] = field(default_factory=lambda: [
        "python", "javascript", "typescript", "java", "cpp", "go"
    ])
    
    # Project generation criteria
    min_files_per_project: int = 5
    max_files_per_project: int = 100
    projects_per_language: int = 200  # For 12,000 total instances
    
    # Complexity levels for synthetic projects
    complexity_distribution: Dict[str, float] = field(default_factory=lambda: {
        "easy": 0.25,      # 25% easy projects
        "medium": 0.40,    # 40% medium projects  
        "hard": 0.25,      # 25% hard projects
        "expert": 0.10     # 10% expert projects
    })
    
    # Generation quality controls
    min_complexity_score: float = 0.3
    max_complexity_score: float = 0.9
    min_documentation_ratio: float = 0.1


@dataclass 
class BenchmarkConfig:
    """Configuration for benchmark generation"""
    # Scale parameters
    total_instances: int = 12000
    
    # Task category distribution
    task_distribution: Dict[str, int] = field(default_factory=lambda: {
        "architectural_understanding": 1500,
        "cross_file_refactoring": 1500, 
        "feature_implementation": 1900,
        "bug_investigation": 1600,
        "multi_session_development": 1200,
        "code_comprehension": 1600,
        "integration_testing": 1400,
        "security_analysis": 1300
    })
    
    # Difficulty distribution
    difficulty_distribution: Dict[str, int] = field(default_factory=lambda: {
        "easy": 3200,    # 10K-40K tokens
        "medium": 4500,  # 40K-100K tokens  
        "hard": 3600,    # 100K-200K tokens
        "expert": 700    # 200K+ tokens
    })
    
    # Context length ranges
    context_ranges: Dict[str, tuple] = field(default_factory=lambda: {
        "easy": (10000, 40000),
        "medium": (40000, 100000),
        "hard": (100000, 200000), 
        "expert": (200000, 500000)
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
    task_timeout: int = 300
    session_timeout: int = 1800
    
    # Validation settings
    human_validation_ratio: float = 0.05  # 5% manual validation
    inter_rater_agreement_threshold: float = 0.8


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
        self.api.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.api.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        
        # Override data paths if set
        if os.getenv("ACE_CACHE_DIR"):
            self.data.cache_dir = os.getenv("ACE_CACHE_DIR")
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
                'stack_v2_path': self.data.stack_v2_path,
                'starcoder_data_path': self.data.starcoder_data_path,
                'codenet_path': self.data.codenet_path,
                'cache_dir': self.data.cache_dir,
                'output_dir': self.data.output_dir,
                'benchmark_dir': self.data.benchmark_dir,
                'min_stars': self.data.min_stars,
                'min_files': self.data.min_files,
                'max_files': self.data.max_files,
                'supported_languages': self.data.supported_languages,
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
                'human_validation_ratio': self.evaluation.human_validation_ratio,
                'inter_rater_agreement_threshold': self.evaluation.inter_rater_agreement_threshold,
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.data.cache_dir,
            self.data.output_dir,
            self.data.benchmark_dir,
            f"{self.data.cache_dir}/repositories",
            f"{self.data.cache_dir}/analysis",
            f"{self.data.output_dir}/tasks",
            f"{self.data.output_dir}/evaluations",
            f"{self.data.benchmark_dir}/instances",
            f"{self.data.benchmark_dir}/references"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Check API keys
        if not any([
            self.api.openai_api_key,
            self.api.anthropic_api_key, 
            self.api.google_api_key
        ]):
            errors.append("At least one API key must be provided")
            
        # Check directory permissions
        try:
            Path(self.data.cache_dir).mkdir(parents=True, exist_ok=True)
            Path(self.data.output_dir).mkdir(parents=True, exist_ok=True)
            Path(self.data.benchmark_dir).mkdir(parents=True, exist_ok=True)
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
- Min stars: {self.data.min_stars}
- File range: {self.data.min_files} - {self.data.max_files}

API Configuration:
- OpenAI: {'✓' if self.api.openai_api_key else '✗'}
- Anthropic: {'✓' if self.api.anthropic_api_key else '✗'}
- Google: {'✓' if self.api.google_api_key else '✗'}
- HuggingFace: {'✓' if self.api.huggingface_token else '✗'}
""" 