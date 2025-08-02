# AgentCodeEval

**A Novel Benchmark for Evaluating Long-Context Language Models in Software Development Agent Tasks**

AgentCodeEval is a comprehensive benchmark designed to evaluate long-context language models (LLMs) in realistic software development agent scenarios. Unlike existing code evaluation benchmarks that focus on single-function completion or isolated code understanding, AgentCodeEval addresses the critical gap in evaluating LLMs' ability to perform complex, multi-file software development tasks.

## 🎯 Key Features

- **Unprecedented Scale**: 12,000 evaluation instances (5x larger than SWE-Bench)
- **Synthetic Generation**: Clean, uncontaminated benchmark data using elite LLMs
- **Long-Context Focus**: 10K-200K+ token contexts with high information coverage (IC > 0.7)
- **Agent-Specific Evaluation**: Multi-session development workflows and architectural understanding
- **Novel Metrics**: 6 specialized metrics for software development agent capabilities
- **Production Ready**: Robust error handling, automated validation, and comprehensive testing

## 📊 Benchmark Overview

### Task Categories (8 categories, 12,000 instances)

1. **Architectural Understanding** (1,500 instances) - Design pattern recognition, dependency analysis
2. **Cross-file Refactoring** (1,500 instances) - Multi-file restructuring and pattern application  
3. **Feature Implementation** (1,800 instances) - Adding functionality to existing systems
4. **Bug Investigation** (1,500 instances) - Systematic debugging across complex codebases
5. **Multi-session Development** (1,200 instances) - Context persistence across development sessions
6. **Code Comprehension** (1,500 instances) - Deep understanding and explanation capabilities
7. **Integration Testing** (1,300 instances) - System-level testing and validation
8. **Security Analysis** (1,200 instances) - Security-aware development practices

### Difficulty Levels

- **Easy** (3,200 instances): 10K-40K tokens, basic architectural patterns
- **Medium** (4,500 instances): 40K-100K tokens, moderate complexity
- **Hard** (3,600 instances): 100K-200K tokens, complex architectures  
- **Expert** (700 instances): 200K+ tokens, enterprise-scale systems

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- API access to at least one LLM provider:
  - **OpenAI o3** (recommended for best performance)
  - **Claude Sonnet 4** (via AWS Bedrock)
  - **Gemini 2.5 Pro** (via Google AI Studio)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/AgentCodeEval/AgentCodeEval.git
cd AgentCodeEval
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
# OR install in development mode
pip install -e .
```

3. **Set up API keys**:
```bash
# Copy the template and add your real API keys
cp api.sh.template api.sh
# Edit api.sh with your actual API keys (at least one required)
nano api.sh  # or your preferred editor
# Apply the configuration
source api.sh
```

## 📖 Complete Usage Guide

### Step 1: Verify Setup

Check that your configuration is working:
```bash
agentcodeeval status
```

This will show you:
- ✅ Which API keys are configured
- 📁 Directory structure 
- ⚙️ Current configuration settings

### Step 2: Generate Benchmark Data

AgentCodeEval uses a 4-phase generation pipeline:

#### Phase 1: Project Specification Generation
Generate synthetic project blueprints:
```bash
# Quick test with minimal data
agentcodeeval generate --phase 1 --config-path test_config.yaml

# Full generation (12,000 instances)
agentcodeeval generate --phase 1
```

#### Phase 2: Synthetic Codebase Generation  
Create actual multi-file codebases:
```bash
agentcodeeval generate --phase 2 --config-path test_config.yaml
```

#### Phase 3: Evaluation Scenario Creation
Generate evaluation tasks from the codebases:
```bash
agentcodeeval generate --phase 3 --config-path test_config.yaml
```

#### Phase 4: Test Suite Generation
Create automated validation frameworks:
```bash
agentcodeeval generate --phase 4 --config-path test_config.yaml
```

#### Generate All Phases at Once
```bash
# For testing (small dataset)
agentcodeeval generate --phase all --config-path test_config.yaml

# For full benchmark generation
agentcodeeval generate --phase all
```

### Step 3: Evaluate LLMs

Once you have generated scenarios, evaluate models:

#### Single Model Evaluation
```bash
# Evaluate OpenAI o3 (auto-saves to evaluation_results/openaio3_allcats_alldiff_TIMESTAMP_evaluation_results.json)
agentcodeeval evaluate --model openai-o3 --task-category feature_implementation

# Evaluate Claude Sonnet 4 on architectural understanding
agentcodeeval evaluate --model anthropic-claude-sonnet-4 --task-category architectural_understanding

# Evaluate with custom config
agentcodeeval evaluate --config-path test_config.yaml --model openai-o3

# Custom output file
agentcodeeval evaluate --model openai-o3 --output-file my_results.json

# Display only (no file saving)
agentcodeeval evaluate --model openai-o3 --no-save
```

#### Multiple Models Comparison
```bash
# Compare all models (auto-saves to evaluation_results/3models_allcats_alldiff_TIMESTAMP_evaluation_results.json)
agentcodeeval evaluate --model openai-o3 --model anthropic-claude-sonnet-4 --model google-gemini-2.5-pro

# Specific difficulty level (auto-saves with descriptive filename)
agentcodeeval evaluate --model openai-o3 --difficulty medium
```

### Step 4: View Results

The evaluation will output:
- 📊 **Performance Summary**: Overall scores and grades
- 📈 **Category Breakdown**: Performance by task category  
- 🎯 **Novel Metrics**: ACS, DTA, MMR, CFRD, IDC, ICU scores
- 📋 **Detailed Analysis**: Per-scenario results

#### 💾 Comprehensive JSON Results (Auto-Saved)

Results are automatically saved to `evaluation_results/` with comprehensive data:

```json
{
  "metadata": {
    "evaluation_timestamp": "2025-01-01T15:30:45",
    "models_evaluated": ["openai-o3"],
    "total_scenarios": 12,
    "category_distribution": {...},
    "system_info": {...}
  },
  "configuration": {
    "api_settings": {...},
    "evaluation_weights": {...},
    "benchmark_settings": {...}
  },
  "analysis": {
    "model_comparison": {...},
    "performance_ranking": [...],
    "category_performance": {...}
  },
  "summaries": {
    "openai-o3": {
      "avg_total_score": 0.468,
      "category_results": {...}
    }
  },
  "detailed_results": {
    "openai-o3": [
      {
        "scenario_id": "web_app_arch_001",
        "detailed_results": {
          "agent_metrics": {
            "architectural_coherence_score": 0.75,
            "dependency_traversal_accuracy": 0.68,
            "multi_session_memory_retention": 0.55,
            "cross_file_reasoning_depth": 0.72,
            "incremental_development_capability": 0.58,
            "information_coverage_utilization": 0.61
          },
          "compilation": {...},
          "code_quality": {...},
          "style_analysis": {...}
        }
      }
    ]
  },
  "scenario_lookup": {
    "web_app_arch_001": {
      "models_evaluated": ["openai-o3"],
      "results": {...}
    }
  }
}
```

Perfect for:
- 📈 **Research Analysis**: Load into pandas/R for statistical analysis
- 🔬 **Deep Investigation**: Per-scenario detailed metrics breakdown
- 📊 **Visualization**: Generate custom charts and comparisons
- 🔄 **Longitudinal Studies**: Compare performance over time

## 🏗️ Development Pipeline

AgentCodeEval follows a 4-phase **synthetic generation pipeline** to eliminate data contamination:

### Phase 1: Project Specification Generation
- **🏆 3 Elite Models**: OpenAI o3, Claude Sonnet 4, Gemini 2.5 Pro for maximum quality  
- **Domain Coverage**: Web apps, data pipelines, ML systems, APIs, games, blockchain
- **Realistic Complexity**: Easy (5-15 files) to Expert (80-150 files)
- **Output**: Unique project specifications (200 per language)

### Phase 2: Synthetic Codebase Generation  
- **Architecture-First**: System design → File structure → Implementation
- **Production Quality**: Real dependencies, tests, documentation, error handling
- **Cross-File Consistency**: Proper imports, shared utilities, API contracts
- **Output**: Complete multi-file projects (10K-500K tokens each)

### Phase 3: Agent Evaluation Scenario Creation
- **8 Task Categories**: Architecture understanding, refactoring, debugging, etc.
- **Progressive Complexity**: Context length from 10K to 500K tokens
- **Realistic Workflows**: Multi-session development, incremental building
- **Output**: Evaluation instances with ground truth solutions

### Phase 4: Automated Test-Driven Validation
- **No LLM Bias**: Purely automated validation using compilation, testing, metrics
- **Real Code Analysis**: Actual compilation, security scanning, quality analysis
- **Novel Agent Metrics**: 6 specialized metrics for agent capabilities
- **Output**: Production-ready benchmark with objective evaluation

## 📈 Evaluation Metrics

AgentCodeEval introduces 6 novel agent-specific metrics:

1. **Architectural Coherence Score (ACS)** - Consistency with design patterns
2. **Dependency Traversal Accuracy (DTA)** - Navigation of complex dependencies  
3. **Multi-Session Memory Retention (MMR)** - Context persistence across sessions
4. **Cross-File Reasoning Depth (CFRD)** - Understanding multi-file relationships
5. **Incremental Development Capability (IDC)** - Building on previous work
6. **Information Coverage Utilization (ICU)** - Effective context usage

**Composite Agent Development Score (CADS)**: Weighted combination of all metrics (0-5 scale)

### Evaluation Weights
- **Functional Correctness**: 40% (compilation, tests, functionality)
- **Agent-Specific Metrics**: 30% (ACS, DTA, MMR, CFRD, IDC, ICU)
- **Code Quality**: 20% (complexity, maintainability, security)
- **Style & Practices**: 10% (formatting, conventions, documentation)

## ⚙️ Configuration

### Quick Configuration for Testing
Use `test_config.yaml` for faster testing with smaller datasets:
```bash
# Generate small test dataset (16 instances total)
agentcodeeval generate --phase all --config-path test_config.yaml

# Evaluate on test dataset  
agentcodeeval evaluate --config-path test_config.yaml --model openai-o3
```

### Full Production Configuration
Edit `config.yaml` for full-scale generation:
```yaml
# config.yaml
api:
  max_requests_per_minute: 60
  default_model_openai: "o3"  # 🏆 Elite model
  default_model_anthropic: "claude-sonnet-4-20250514"
  default_model_google: "gemini-2.5-pro"

benchmark:
  total_instances: 12000
  min_information_coverage: 0.7

evaluation:
  metric_weights:
    architectural_coherence: 0.20
    dependency_traversal: 0.20
    multi_session_memory: 0.20
    cross_file_reasoning: 0.15
    incremental_development: 0.15
    information_coverage: 0.10
```

## 🔧 Advanced Usage

### Force Regeneration
```bash
# Regenerate all phases (overwrites existing data)
agentcodeeval generate --phase all --force

# Regenerate specific phase
agentcodeeval generate --phase 3 --force
```

### Custom Output Directory
```bash
# Set custom output directory
export ACE_OUTPUT_DIR="/path/to/custom/output"
agentcodeeval generate --phase 1
```

### Debugging and Monitoring
```bash
# Check generation status
agentcodeeval status

# Monitor progress (generation shows real-time progress)
agentcodeeval generate --phase 2 --config-path test_config.yaml
```

## 📁 Directory Structure

After running the pipeline, your directory structure will be:
```
AgentCodeEval/
├── data/
│   ├── generated/          # Phase 1-2: Synthetic projects
│   └── output/             # Phase 3-4: Scenarios & validation
│       ├── scenarios/      # Evaluation scenarios (.json)
│       ├── validation/     # Test suites & validation data
│       └── references/     # Reference solutions (if generated)
├── evaluation_results/     # Auto-saved evaluation results (.json)
│   ├── openai3_allcats_alldiff_20250101_123045_evaluation_results.json
│   ├── 3models_allcats_medium_20250101_140532_evaluation_results.json
│   └── ...                 # Timestamped comprehensive results
├── agentcodeeval/          # Main framework code
├── config.yaml             # Production configuration
├── test_config.yaml        # Testing configuration
└── api.sh                  # API key configuration (not in git)
```

## 🚨 Important Notes

- **API Costs**: Full generation (12,000 instances) requires significant API usage (~$100-300)
- **Time Requirements**: Full generation takes 6-12 hours depending on API rate limits
- **Storage**: Generated data requires ~1-5GB of storage
- **Testing First**: Always use `test_config.yaml` first to verify your setup

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Configuration Reference](docs/configuration.md)
- [API Documentation](docs/api.md)
- [Evaluation Metrics](docs/metrics.md)
- [Contributing Guidelines](docs/contributing.md)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) for details.

## 📧 Contact

- **Research Paper**: [arXiv:XXXX.XXXXX](https://arxiv.org)
- **Website**: [https://agentcodeeval.github.io](https://agentcodeeval.github.io)
- **Issues**: [GitHub Issues](https://github.com/AgentCodeEval/AgentCodeEval/issues)

## 🙏 Acknowledgments

- **BigCode Community** for advancing open code AI research
- **HuggingFace** for datasets and evaluation framework infrastructure
- **Tree-sitter** community for robust multi-language parsing
- Research community for foundational work in code evaluation

---

**AgentCodeEval**: Advancing the evaluation of AI-powered software development agents through comprehensive, large-scale benchmarking. 