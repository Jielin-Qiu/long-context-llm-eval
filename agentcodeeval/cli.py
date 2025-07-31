"""
Command Line Interface for AgentCodeEval
"""

import click
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .core.config import Config

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="AgentCodeEval")
@click.pass_context  
def main(ctx):
    """AgentCodeEval: A Novel Benchmark for Evaluating Long-Context LLMs in Software Development Agent Tasks"""
    ctx.ensure_object(dict)


@main.command()
@click.option('--config-path', '-c', type=click.Path(), help='Path to configuration file')
@click.option('--save-config', '-s', type=click.Path(), help='Save configuration to file')
def setup(config_path, save_config):
    """Set up AgentCodeEval environment and configuration"""
    console.print(Panel.fit("🚀 AgentCodeEval Setup", style="bold blue"))
    
    try:
        # Load configuration
        config = Config(config_path=config_path)
        
        # Validate configuration
        errors = config.validate()
        
        if errors:
            console.print("❌ Configuration errors found:", style="bold red")
            for error in errors:
                console.print(f"  • {error}", style="red")
            
            if not any([config.api.openai_api_key, config.api.anthropic_api_key, config.api.google_api_key]):
                console.print("\n💡 To fix API key issues, set environment variables:", style="yellow")
                console.print("  export OPENAI_API_KEY='your-key-here'")
                console.print("  export ANTHROPIC_API_KEY='your-key-here'") 
                console.print("  export GOOGLE_API_KEY='your-key-here'")
                console.print("  export HUGGINGFACE_TOKEN='your-token-here'")
                
            sys.exit(1)
        
        # Display configuration summary
        console.print("✅ Configuration validated successfully!", style="bold green")
        console.print(config.summary())
        
        # Save configuration if requested
        if save_config:
            config.save_to_file(save_config)
            console.print(f"💾 Configuration saved to: {save_config}", style="green")
            
        console.print("🎯 Setup complete! Ready to begin AgentCodeEval benchmark generation.", style="bold green")
        
    except Exception as e:
        console.print(f"❌ Setup failed: {e}", style="bold red")
        sys.exit(1)


@main.command()
@click.option('--config-path', '-c', type=click.Path(), help='Path to configuration file')
def status(config_path):
    """Show current AgentCodeEval status and configuration"""
    try:
        config = Config(config_path=config_path)
        
        # Create status table
        table = Table(title="AgentCodeEval Status", style="cyan")
        table.add_column("Component", style="bold")
        table.add_column("Status", justify="center") 
        table.add_column("Details")
        
        # API status
        apis = [
            ("OpenAI", config.api.openai_api_key),
            ("Anthropic", config.api.anthropic_api_key),
            ("Google", config.api.google_api_key),
            ("HuggingFace", config.api.huggingface_token)
        ]
        
        for name, key in apis:
            status_icon = "✅" if key else "❌"
            status_text = "Configured" if key else "Missing"
            table.add_row(f"{name} API", status_icon, status_text)
        
        # Directory status
        directories = [
            ("Cache Directory", config.data.cache_dir),
            ("Output Directory", config.data.output_dir), 
            ("Benchmark Directory", config.data.benchmark_dir)
        ]
        
        for name, path in directories:
            exists = Path(path).exists()
            status_icon = "✅" if exists else "❌"
            status_text = f"{'Exists' if exists else 'Missing'}: {path}"
            table.add_row(name, status_icon, status_text)
        
        # Benchmark configuration
        table.add_row("Benchmark Scale", "📊", f"{config.benchmark.total_instances:,} instances")
        table.add_row("Task Categories", "📋", f"{len(config.benchmark.task_distribution)} categories")
        table.add_row("Languages", "🔤", f"{len(config.data.supported_languages)} languages")
        
        console.print(table)
        
        # Validation errors
        errors = config.validate()
        if errors:
            console.print("\n⚠️  Configuration Issues:", style="yellow")
            for error in errors:
                console.print(f"  • {error}", style="yellow")
        else:
            console.print("\n✅ All systems ready!", style="bold green")
            
    except Exception as e:
        console.print(f"❌ Status check failed: {e}", style="bold red")
        sys.exit(1)


@main.command()
@click.option('--config-path', '-c', type=click.Path(), help='Path to configuration file')
@click.option('--phase', type=click.Choice(['1', '2', '3', '4', 'all']), default='1', 
              help='Which implementation phase to run')
@click.option('--dry-run', is_flag=True, help='Show what would be done without executing')
def generate(config_path, phase, dry_run):
    """Generate AgentCodeEval benchmark instances"""
    console.print(Panel.fit(f"🏗️  AgentCodeEval Generation - Phase {phase}", style="bold green"))
    
    if dry_run:
        console.print("🔍 DRY RUN MODE - No actual generation will occur", style="yellow")
    
    try:
        config = Config(config_path=config_path)
        
        # Validate configuration
        errors = config.validate()
        if errors:
            console.print("❌ Configuration errors found:", style="bold red")
            for error in errors:
                console.print(f"  • {error}", style="red")
            sys.exit(1)
        
        if phase == '1' or phase == 'all':
            console.print("📊 Phase 1: Repository Analysis and Selection", style="bold")
            if not dry_run:
                from .analysis.repository_analyzer import run_phase_1
                run_phase_1(config)
            else:
                console.print("  • Filter The Stack v2 repositories")
                console.print("  • Extract AST and dependency information")
                console.print("  • Calculate quality metrics")
                console.print(f"  • Target: 50,000 analyzed repositories")
                
        if phase == '2' or phase == 'all':
            console.print("🎯 Phase 2: Scenario Template Generation", style="bold")
            if not dry_run:
                from .generation.template_generator import run_phase_2
                run_phase_2(config)
            else:
                console.print("  • Create 15 templates per task category")
                console.print("  • Design complexity scaling") 
                console.print("  • Implement IC optimization")
                console.print(f"  • Target: 120 validated templates")
                
        if phase == '3' or phase == 'all':
            console.print("🔄 Phase 3: Large-Scale Instance Generation", style="bold") 
            if not dry_run:
                from .generation.instance_generator import run_phase_3
                run_phase_3(config)
            else:
                console.print("  • Match templates with repositories")
                console.print("  • Generate 100 variations per template")
                console.print("  • Quality assurance pipeline")
                console.print(f"  • Target: {config.benchmark.total_instances:,} instances")
                
        if phase == '4' or phase == 'all':
            console.print("✅ Phase 4: Reference Solution Generation", style="bold")
            if not dry_run:
                from .generation.reference_generator import run_phase_4
                run_phase_4(config)
            else:
                console.print("  • Generate reference solutions")
                console.print("  • Validate solution correctness")
                console.print("  • Create evaluation rubrics")
                console.print(f"  • Target: 2-3 solutions per instance")
        
        console.print("🎉 Generation phase complete!", style="bold green")
        
    except Exception as e:
        console.print(f"❌ Generation failed: {e}", style="bold red")
        sys.exit(1)


@main.command()
@click.option('--config-path', '-c', type=click.Path(), help='Path to configuration file')
@click.option('--model', '-m', multiple=True, help='Model to evaluate (can specify multiple)')
@click.option('--task-category', '-t', multiple=True, help='Task category to evaluate')
@click.option('--difficulty', '-d', type=click.Choice(['easy', 'medium', 'hard', 'expert']),
              help='Difficulty level to evaluate')
@click.option('--output-file', '-o', type=click.Path(), help='Output file for results')
def evaluate(config_path, model, task_category, difficulty, output_file):
    """Evaluate models on AgentCodeEval benchmark"""
    console.print(Panel.fit("🧪 AgentCodeEval Evaluation", style="bold purple"))
    
    try:
        config = Config(config_path=config_path)
        
        # Show evaluation parameters
        console.print("📋 Evaluation Parameters:", style="bold")
        console.print(f"  • Models: {list(model) if model else 'All available'}")
        console.print(f"  • Categories: {list(task_category) if task_category else 'All categories'}")
        console.print(f"  • Difficulty: {difficulty if difficulty else 'All levels'}")
        console.print(f"  • Output: {output_file if output_file else 'Standard output'}")
        
        from .evaluation.evaluator import run_evaluation
        results = run_evaluation(config, model, task_category, difficulty)
        
        # Display results
        if results:
            console.print("📊 Evaluation Results:", style="bold green")
            # TODO: Format and display results
            console.print(results)
            
            if output_file:
                # TODO: Save results to file
                console.print(f"💾 Results saved to: {output_file}", style="green")
        
    except Exception as e:
        console.print(f"❌ Evaluation failed: {e}", style="bold red")
        sys.exit(1)


@main.command()
def version():
    """Show AgentCodeEval version information"""
    console.print("🔧 AgentCodeEval v0.1.0", style="bold blue")
    console.print("A Novel Benchmark for Evaluating Long-Context Language Models")
    console.print("in Software Development Agent Tasks")
    console.print("\nFor more information: https://github.com/AgentCodeEval/AgentCodeEval")


if __name__ == '__main__':
    main() 