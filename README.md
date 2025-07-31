# AgentCodeEval

**A Novel Benchmark for Evaluating Long-Context Language Models in Software Development Agent Tasks**

AgentCodeEval is a comprehensive benchmark designed to evaluate long-context language models (LLMs) in realistic software development agent scenarios. Unlike existing code evaluation benchmarks that focus on single-function completion or isolated code understanding, AgentCodeEval addresses the critical gap in evaluating LLMs' ability to perform complex, multi-file software development tasks.

## 🎯 Key Features

- **Unprecedented Scale**: 12,000 evaluation instances (5x larger than SWE-Bench)
- **Real-World Relevance**: Tasks derived from production codebases in The Stack v2
- **Long-Context Focus**: 10K-200K+ token contexts with high information coverage (IC > 0.7)
- **Agent-Specific Evaluation**: Multi-session development workflows and architectural understanding
- **Novel Metrics**: 6 specialized metrics for software development agent capabilities
- **Automated Generation**: Scalable pipeline using tree-sitter and LLM APIs

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
- API access to at least one LLM provider (OpenAI, Anthropic, or Google)
- HuggingFace account for dataset access

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
export OPENAI_API_KEY="your-openai-key-here"
export ANTHROPIC_API_KEY="your-anthropic-key-here"
export GOOGLE_API_KEY="your-google-key-here"
export HUGGINGFACE_TOKEN="your-hf-token-here"
```

4. **Initialize configuration**:
```bash
agentcodeeval setup --save-config my_config.yaml
```

### Basic Usage

1. **Check system status**:
```bash
agentcodeeval status
```

2. **Generate benchmark instances (Phase 1)**:
```bash
# Dry run to see what would happen
agentcodeeval generate --phase 1 --dry-run

# Actually run Phase 1: Repository Analysis
agentcodeeval generate --phase 1
```

3. **Generate all phases**:
```bash
agentcodeeval generate --phase all
```

4. **Evaluate models**:
```bash
agentcodeeval evaluate --model gpt-4o --task-category architectural_understanding
```

## 🏗️ Development Pipeline

AgentCodeEval follows a 4-phase automated generation pipeline:

### Phase 1: Repository Analysis and Selection
- Filter 50,000+ repositories from The Stack v2
- Extract AST and dependency information using tree-sitter
- Calculate architectural complexity and quality metrics
- **Output**: 50,000 analyzed repositories with metadata

### Phase 2: Scenario Template Generation  
- Create 15 scenario templates per task category (120 total)
- Design progressive complexity scaling
- Implement Information Coverage (IC) optimization
- **Output**: 120 validated scenario templates

### Phase 3: Large-Scale Instance Generation
- Match templates with appropriate repositories
- Generate 100 variations per template using LLM APIs
- Quality assurance and deduplication pipeline
- **Output**: 12,000 unique evaluation instances

### Phase 4: Reference Solution Generation
- Multi-LLM approach (GPT-4o, Claude, Gemini)
- Generate 2-3 reference solutions per instance
- Automated validation and evaluation rubrics
- **Output**: Complete benchmark with reference solutions

## 📈 Evaluation Metrics

AgentCodeEval introduces 6 novel agent-specific metrics:

1. **Architectural Coherence Score (ACS)** - Consistency with design patterns
2. **Dependency Traversal Accuracy (DTA)** - Navigation of complex dependencies  
3. **Multi-Session Memory Retention (MMR)** - Context persistence across sessions
4. **Cross-File Reasoning Depth (CFRD)** - Understanding multi-file relationships
5. **Incremental Development Capability (IDC)** - Building on previous work
6. **Information Coverage Utilization (ICU)** - Effective context usage

**Composite Agent Development Score (CADS)**: Weighted combination of all metrics (0-5 scale)

## 🔧 Configuration

AgentCodeEval uses YAML configuration files. Example:

```yaml
# config.yaml
api:
  max_requests_per_minute: 60
  default_model_openai: "gpt-4o"

benchmark:
  total_instances: 12000
  min_information_coverage: 0.7

evaluation:
  metric_weights:
    architectural_coherence: 0.20
    dependency_traversal: 0.20
    multi_session_memory: 0.20
    # ... other weights
```

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

- **The Stack v2** by BigCode for providing high-quality code datasets
- **HuggingFace** for datasets and evaluation framework infrastructure
- **Tree-sitter** community for robust multi-language parsing
- Research community for foundational work in code evaluation

---

**AgentCodeEval**: Advancing the evaluation of AI-powered software development agents through comprehensive, large-scale benchmarking. 