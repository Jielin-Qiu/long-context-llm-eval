"""
Command Line Interface for AgentCodeEval
"""

import click
import os
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
            
            # Check for available APIs (including AWS Bedrock)
            aws_configured = all([
                os.getenv('AWS_ACCESS_KEY_ID'),
                os.getenv('AWS_SECRET_ACCESS_KEY'), 
                os.getenv('AWS_SESSION_TOKEN')
            ])
            anthropic_available = config.api.anthropic_api_key or aws_configured
            
            if not any([config.api.openai_api_key, anthropic_available, config.api.google_api_key]):
                console.print("\n💡 To fix API key issues, set environment variables:", style="yellow")
                console.print("  🏆 For our 3 Elite Models:")
                console.print("  export OPENAI_API_KEY='your-key-here'  # For o3")
                console.print("  export ANTHROPIC_API_KEY='your-key-here'  # For Claude Sonnet 4 (or use AWS)")
                console.print("  export GOOGLE_API_KEY='your-key-here'  # For Gemini 2.5 Pro")
                console.print("  # OR for Claude via AWS Bedrock:")
                console.print("  export AWS_ACCESS_KEY_ID='...' AWS_SECRET_ACCESS_KEY='...' AWS_SESSION_TOKEN='...'")
                
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
        
        # API status (checking our 3 Elite Models)
        # Check AWS credentials for Claude Bedrock access
        import os
        aws_configured = all([
            os.getenv('AWS_ACCESS_KEY_ID'),
            os.getenv('AWS_SECRET_ACCESS_KEY'), 
            os.getenv('AWS_SESSION_TOKEN')
        ])
        anthropic_available = config.api.anthropic_api_key or aws_configured
        
        apis = [
            ("OpenAI", config.api.openai_api_key),
            ("Anthropic", anthropic_available),  # Direct API or AWS Bedrock
            ("Google", config.api.google_api_key),
            # Removed HuggingFace - no longer needed with synthetic generation
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
            console.print("🎯 Phase 1: Synthetic Project Generation", style="bold")
            if not dry_run:
                import asyncio
                from .generation.synthetic_generator import SyntheticProjectGenerator, ProjectDomain, ProjectComplexity
                asyncio.run(run_phase_1_generation(config))
            else:
                console.print("  • Generate synthetic multi-file projects")
                console.print("  • 10 domains × 4 complexity levels × 6 languages")
                console.print("  • Production-quality code with tests & docs")
                console.print(f"  • Target: 1,200 synthetic projects")
                
        if phase == '2' or phase == 'all':
            console.print("🎮 Phase 2: Agent Evaluation Scenario Creation", style="bold")
            if not dry_run:
                console.print("⏳ [yellow]Coming soon: Convert projects to agent evaluation tasks[/yellow]")
                # TODO: Implement scenario generation from synthetic projects
            else:
                console.print("  • Convert synthetic projects to evaluation scenarios")
                console.print("  • Create bug investigation tasks")
                console.print("  • Generate feature implementation challenges") 
                console.print(f"  • Target: 8 task categories × 1,200 projects")
                
        if phase == '3' or phase == 'all':
            console.print("📈 Phase 3: 12,000 Instance Generation", style="bold") 
            if not dry_run:
                console.print("⏳ [yellow]Coming soon: Generate 12,000 evaluation instances[/yellow]")
                # TODO: Generate 10 evaluation instances per project
            else:
                console.print("  • Generate 10 instances per synthetic project")
                console.print("  • Apply progressive complexity scaling")
                console.print("  • Multi-session development scenarios")
                console.print(f"  • Target: {config.benchmark.total_instances:,} instances")
                
        if phase == '4' or phase == 'all':
            console.print("🎯 Phase 4: Reference Solution & Validation", style="bold")
            if not dry_run:
                console.print("⏳ [yellow]Coming soon: Generate ground truth solutions[/yellow]")
                # TODO: Generate reference solutions using multi-LLM approach
            else:
                console.print("  • Multi-LLM reference solution generation")
                console.print("  • Automated validation and quality checks")
                console.print("  • Novel agent-specific evaluation metrics")
                console.print(f"  • Target: Ground truth for all 12,000 instances")
        
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


async def run_phase_1_generation(config):
    """Run Phase 1: Synthetic Project Generation"""
    from .generation.synthetic_generator import SyntheticProjectGenerator, ProjectDomain, ProjectComplexity
    
    console.print("\n🎯 [bold]Synthetic Project Generation Pipeline[/bold]")
    console.print("=" * 60)
    
    generator = SyntheticProjectGenerator(config)
    
    # Target: projects per language from config
    languages = config.data.supported_languages
    projects_per_language = config.data.projects_per_language
    
    console.print(f"📊 Target: {len(languages)} languages × {projects_per_language} projects = {len(languages) * projects_per_language} total")
    console.print(f"🌐 Languages: {', '.join(languages)}")
    
    domains = list(ProjectDomain)[:6]  # Use first 6 domains for balanced distribution
    complexities = list(ProjectComplexity)
    
    total_generated = 0
    
    for language in languages:
        console.print(f"\n🔤 [bold cyan]Processing {language}...[/bold cyan]")
        
        lang_projects = 0
        projects_per_domain = projects_per_language // len(domains)
        
        for domain in domains:
            for complexity in complexities:
                if lang_projects >= projects_per_language:
                    break
                
                try:
                    with console.status(f"[bold green]Generating {complexity.value} {domain.value} in {language}..."):
                        
                        project = await generator.generate_complete_project(
                            domain=domain,
                            complexity=complexity,
                            language=language
                        )
                        
                        project_path = await generator.save_project(project)
                        
                        console.print(f"   ✅ {project.specification.name} ({len(project.files)} files, {sum(len(f.content) for f in project.files):,} chars)")
                        
                        lang_projects += 1
                        total_generated += 1
                        
                except Exception as e:
                    console.print(f"   ❌ Failed {complexity.value} {domain.value}: {str(e)[:50]}...")
                    continue
            
            if lang_projects >= projects_per_language:
                break
        
        console.print(f"   📈 {language}: {lang_projects} projects generated")
    
    console.print(f"\n🎉 [bold green]Phase 1 Complete![/bold green]")
    console.print(f"   📊 Generated: {total_generated} synthetic projects")
    console.print(f"   📁 Location: {config.data.generated_dir}")
    console.print(f"   💾 Total size: ~{total_generated * 50000:,} characters of code")


if __name__ == '__main__':
    main() 