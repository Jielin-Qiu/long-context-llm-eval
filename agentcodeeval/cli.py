"""
Command Line Interface for AgentCodeEval
"""

import click
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import List, Dict, Any

from .core.config import Config
from .generation.synthetic_generator import CriticalAuthError

console = Console()


def save_progress(progress_file: Path, completed_projects: List[Dict[str, Any]], phase: str):
    """Save progress to a JSON file for resumability"""
    with open(progress_file, 'w') as f:
        json.dump({
            'phase': phase,
            'timestamp': str(datetime.now()),
            'completed_projects': completed_projects,
            'total_completed': len(completed_projects)
        }, f, indent=2)

def load_progress(progress_file: Path) -> List[Dict[str, Any]]:
    """Load progress from a JSON file"""
    if not progress_file.exists():
        return []
    
    try:
        with open(progress_file, 'r') as f:
            data = json.load(f)
            return data.get('completed_projects', [])
    except Exception as e:
        console.print(f"⚠️ Warning: Could not load progress file: {e}")
        return []


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
    
            ("Output Directory", config.data.output_dir), 
            ("Generated Directory", config.data.generated_dir)
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
@click.option('--force', is_flag=True, help='Force regeneration of already completed projects')
@click.option('--max-concurrent', '-j', type=int, default=3, 
              help='Maximum concurrent operations (default: 3, recommended: 3-10)')
def generate(config_path, phase, dry_run, force, max_concurrent):
    """Generate AgentCodeEval benchmark instances"""
    console.print(Panel.fit(f"🏗️  AgentCodeEval Generation - Phase {phase}", style="bold green"))
    
    if dry_run:
        console.print("🔍 DRY RUN MODE - No actual generation will occur", style="yellow")
    
    if max_concurrent > 1:
        console.print(f"🚀 Parallel mode: {max_concurrent} concurrent operations", style="bold blue")
    
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
                asyncio.run(run_phase_1_generation(config, max_concurrent))
            else:
                console.print("  • Generate synthetic multi-file projects")
                console.print("  • 10 domains × 4 complexity levels × 6 languages")
                console.print("  • Production-quality code with tests & docs")
                console.print(f"  • Target: 1,200 synthetic projects")
                
        if phase == '2' or phase == 'all':
            console.print("🎯 Phase 2: Synthetic Codebase Generation", style="bold")
            if not dry_run:
                import asyncio
                asyncio.run(run_phase_2_generation(config, force, max_concurrent))
            else:
                console.print("  • Generate actual code files from specifications")
                console.print("  • Multi-file projects with realistic complexity")
                console.print("  • Tests, documentation, and error handling")
                
        if phase == '3' or phase == 'all':
            console.print("🎯 Phase 3: Agent Evaluation Scenario Creation", style="bold")
            if not dry_run:
                import asyncio
                asyncio.run(run_phase_3_generation(config, force, max_concurrent))
            else:
                console.print("  • Create evaluation scenarios from generated code")
                console.print("  • 8 task categories × varying difficulties")
                console.print("  • Context-rich scenarios for agent testing")
                
        if phase == '4' or phase == 'all':
            console.print("🎯 Phase 4: Automated Test-Driven Validation", style="bold")
            if not dry_run:
                import asyncio
                asyncio.run(run_phase_4_generation(config, force, max_concurrent))
            else:
                console.print("  • Generate automated test suites")
                console.print("  • Compilation, unit tests, integration tests")
                console.print("  • 6 novel agent-specific metrics (ACS, DTA, MMR, CFRD, IDC, ICU)")
                console.print("  • Implement 6 novel agent-specific metrics (ACS, DTA, MMR, CFRD, IDC, ICU)")
                console.print("  • Security analysis and code quality validation")
                
        console.print("\n✅ Generation complete!", style="bold green")
        console.print("Next steps:")
        console.print("  • Run evaluation: agentcodeeval evaluate")
        console.print("  • Check status: agentcodeeval status")
        
    except Exception as e:
        console.print(f"❌ Generation failed: {e}", style="bold red")
        sys.exit(1)


@main.command()
@click.option('--config-path', '-c', type=click.Path(), help='Path to configuration file')
@click.option('--model', '-m', multiple=True, help='Model to evaluate (can specify multiple)')
@click.option('--task-category', '-t', multiple=True, help='Task category to evaluate')
@click.option('--difficulty', '-d', type=click.Choice(['easy', 'medium', 'hard', 'expert']),
              help='Difficulty level to evaluate')
@click.option('--output-file', '-o', type=click.Path(), help='Output file for results (auto-generated if not specified)')
@click.option('--no-save', is_flag=True, help='Skip saving results to file (display only)')
def evaluate(config_path, model, task_category, difficulty, output_file, no_save):
    """Evaluate models on AgentCodeEval benchmark"""
    console.print(Panel.fit("🧪 AgentCodeEval Evaluation", style="bold purple"))
    
    try:
        config = Config(config_path=config_path)
        
        from .evaluation.evaluator import run_evaluation
        evaluation_data = run_evaluation(config, model, task_category, difficulty)
        
        # Check if evaluation succeeded
        if not evaluation_data.get('success', False):
            console.print(f"❌ Evaluation failed: {evaluation_data.get('error', 'Unknown error')}", style="bold red")
            return
        
        # Extract results
        evaluator = evaluation_data['evaluator']
        results = evaluation_data['results']
        summaries = evaluation_data['summaries']
        
        # Auto-generate output filename if not provided
        if not output_file and not no_save:
            from datetime import datetime
            from pathlib import Path
            
            # Build descriptive filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Model names part
            model_list = list(model) if model else ['all-models']
            models_part = "_".join([m.replace('-', '').replace('_', '').lower() for m in model_list])
            if len(models_part) > 30:  # Limit length
                models_part = f"{len(model_list)}models"
            
            # Category part
            if task_category:
                categories_part = "_".join([c.replace('_', '') for c in task_category])
                if len(categories_part) > 20:
                    categories_part = f"{len(task_category)}cats"
            else:
                categories_part = "allcats"
            
            # Difficulty part
            difficulty_part = difficulty if difficulty else "alldiff"
            
            # Construct filename
            output_file = f"{models_part}_{categories_part}_{difficulty_part}_{timestamp}_evaluation_results.json"
            
            # Ensure results directory exists
            results_dir = Path("evaluation_results")
            results_dir.mkdir(exist_ok=True)
            output_file = results_dir / output_file
        
        # Show evaluation parameters (including auto-generated filename)
        console.print("📋 Evaluation Parameters:", style="bold")
        console.print(f"  • Models: {list(model) if model else 'All available'}")
        console.print(f"  • Categories: {list(task_category) if task_category else 'All categories'}")
        console.print(f"  • Difficulty: {difficulty if difficulty else 'All levels'}")
        if no_save:
            console.print(f"  • Output: Display only (saving disabled)")
        else:
            console.print(f"  • Output: {output_file}")
        
        # Display formatted results
        if summaries:
            console.print("\n📊 Evaluation Completed!", style="bold green")
            evaluator.display_results(summaries)
            
            # Save comprehensive results (unless explicitly disabled)
            if not no_save:
                from pathlib import Path
                output_path = Path(output_file)
                evaluator.save_results(results, summaries, output_path)
        else:
            console.print("❌ No evaluation results generated", style="bold red")
        
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


async def run_phase_1_generation(config, max_concurrent=3):
    """Run Phase 1: Synthetic Project Generation with Guaranteed Uniqueness"""
    from .generation.synthetic_generator import (
        SyntheticProjectGenerator, ProjectDomain, ProjectComplexity,
        ProjectArchitecture, ProjectTheme
    )
    import asyncio
    from asyncio import Semaphore
    
    console.print("\n🎯 [bold]Synthetic Project Generation Pipeline (Uniqueness Guaranteed)[/bold]")
    console.print("=" * 75)
    
    generator = SyntheticProjectGenerator(config)
    
    # Target: projects per language from config
    languages = config.data.supported_languages
    projects_per_language = config.data.projects_per_language
    total_projects = len(languages) * projects_per_language
    
    # Get all available factors for uniqueness
    domains = list(ProjectDomain)
    complexities = list(ProjectComplexity)
    architectures = list(ProjectArchitecture)
    themes = list(ProjectTheme)
    
    console.print(f"📊 Target: {len(languages)} languages × {projects_per_language} projects = {total_projects} total")
    console.print(f"🌐 Languages: {', '.join(languages)}")
    console.print(f"🏗️ Uniqueness factors:")
    console.print(f"   • {len(domains)} domains × {len(complexities)} complexities × {len(architectures)} architectures × {len(themes)} themes")
    console.print(f"   • = {len(domains) * len(complexities) * len(architectures) * len(themes):,} possible combinations")
    console.print(f"   • + unique seeds = guaranteed uniqueness for {projects_per_language} projects per language ✅")
    
    if max_concurrent > 1:
        console.print(f"🚀 [bold blue]Parallel mode: {max_concurrent} concurrent specifications[/bold blue]")
    
    console.print("🏗️ Generating unique project specifications...")
    
    # Create complexity selection pool based on config distribution
    import random
    complexity_pool = []
    for complexity_name, ratio in config.data.complexity_distribution.items():
        complexity_enum = getattr(ProjectComplexity, complexity_name.upper())
        count = int(projects_per_language * len(languages) * ratio)
        complexity_pool.extend([complexity_enum] * count)
    
    # Ensure we have exactly the right number of complexities
    while len(complexity_pool) < total_projects:
        complexity_pool.append(random.choice(complexities))
    while len(complexity_pool) > total_projects:
        complexity_pool.pop()
    
    # Shuffle for random distribution
    random.shuffle(complexity_pool)
    
    # Generate unique combinations for each language
    spec_tasks = []
    global_index = 0
    
    for language in languages:
        console.print(f"🔧 [cyan]Planning {projects_per_language} unique projects for {language}...[/cyan]")
        
        # Create unique combinations for this language
        language_combinations = []
        
        for i in range(projects_per_language):
            # Use different distribution strategies to ensure uniqueness
            domain = domains[i % len(domains)]
            complexity = complexity_pool[global_index]
            architecture = architectures[i % len(architectures)]
            theme = themes[i % len(themes)]
            
            # Create unique seed for deterministic but varied LLM generation
            unique_seed = hash(f"{language}-{domain.value}-{complexity.value}-{architecture.value}-{theme.value}-{i}") % 1000000
            
            # Generate unique project ID
            unique_id = f"{language}_{domain.value}_{complexity.value}_{i:03d}"
            
            language_combinations.append({
                'unique_id': unique_id,
                'language': language,
                'domain': domain,
                'complexity': complexity,
                'architecture': architecture,
                'theme': theme,
                'index': i,
                'seed': unique_seed,
                'global_index': global_index
            })
            
            global_index += 1
        
        # Add to spec tasks
        spec_tasks.extend(language_combinations)
        
        # Verify uniqueness for this language
        unique_combinations = set()
        for combo in language_combinations:
            combination_key = (combo['domain'].value, combo['complexity'].value, 
                             combo['architecture'].value, combo['theme'].value)
            unique_combinations.add(combination_key)
        
        console.print(f"   ✅ [green]{len(unique_combinations)} unique factor combinations for {language}[/green]")
    
    console.print(f"🎯 Generated {len(spec_tasks)} unique project specifications...")
    
    # Verify global uniqueness
    all_unique_ids = set(task['unique_id'] for task in spec_tasks)
    console.print(f"🔍 Uniqueness verification: {len(all_unique_ids)} unique IDs for {len(spec_tasks)} projects ✅")
    
    # Semaphore for parallel generation
    semaphore = Semaphore(max_concurrent)
    
    # Statistics tracking
    projects_generated = 0
    projects_failed = 0
    
    async def generate_single_spec(task_info, task_index):
        """Generate a single project specification with guaranteed uniqueness"""
        async with semaphore:
            unique_id = task_info['unique_id']
            language = task_info['language']
            domain = task_info['domain']
            complexity = task_info['complexity']
            architecture = task_info['architecture']
            theme = task_info['theme']
            seed = task_info['seed']
            
            try:
                console.print(f"🔨 [bold cyan]Generating {task_index}/{len(spec_tasks)}: {unique_id}[/bold cyan]")
                console.print(f"     {language} | {domain.value} | {complexity.value} | {architecture.value} | {theme.value}")
                
                # Start timing
                import time
                start_time = time.time()
                
                # Set random seed for deterministic variation
                random.seed(seed)
                
                # Generate project specification with unique factors
                spec = await generator.generate_project_specification_unique(
                    domain, complexity, language, architecture, theme, unique_id, seed
                )
                
                generation_time = time.time() - start_time
                
                # Save specification to project directory  
                project_name = unique_id
                
                # Create project directory and save specification
                project_dir = generator.generated_dir / project_name
                project_dir.mkdir(exist_ok=True)
                
                # Save specification metadata
                metadata = {
                    "specification": spec.to_dict(),
                    "generated_timestamp": time.time(),
                    "phase_1_complete": True,
                    "uniqueness_factors": {
                        "domain": domain.value,
                        "complexity": complexity.value, 
                        "architecture": architecture.value,
                        "theme": theme.value,
                        "seed": seed
                    }
                }
                
                with open(project_dir / "project_metadata.json", 'w') as f:
                    import json
                    json.dump(metadata, f, indent=2)
                
                console.print(f"   ✅ [green]Generated {project_name}![/green] {spec.target_file_count} files, ~{spec.target_token_count:,} tokens ({generation_time:.1f}s)")
                
                return {
                    'success': True,
                    'project_name': project_name,
                    'unique_id': unique_id,
                    'language': language,
                    'domain': domain.value,
                    'complexity': complexity.value,
                    'architecture': architecture.value,
                    'theme': theme.value
                }
                
            except Exception as e:
                console.print(f"   ❌ [red]Failed {unique_id}: {str(e)}[/red]")
                return {
                    'success': False,
                    'error': str(e),
                    'unique_id': unique_id,
                    'language': language,
                    'domain': domain.value
                }
    
    # Execute all specification generation tasks in parallel
    console.print(f"\n🚀 [bold]Starting parallel specification generation for {len(spec_tasks)} projects...[/bold]")
    
    # Create asyncio tasks
    tasks = []
    for i, task_info in enumerate(spec_tasks, 1):
        task = generate_single_spec(task_info, i)
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    successful_projects = []
    failed_projects = []
    
    for result in results:
        if isinstance(result, Exception):
            failed_projects.append(f"Exception: {str(result)}")
            projects_failed += 1
        elif result and result['success']:
            successful_projects.append(result)
            projects_generated += 1
        else:
            failed_projects.append(f"{result['language']} {result['domain']}" if result else "Unknown project")
            projects_failed += 1
    
    # Final summary
    console.print(f"\n📊 [bold]Phase 1 Summary:[/bold]")
    console.print(f"   ✅ Generated: {projects_generated} project specifications")
    console.print(f"   ❌ Failed: {projects_failed} specifications")
    console.print(f"   📁 Specifications saved to: {generator.generated_dir}")
    
    if failed_projects:
        console.print(f"\n⚠️  [yellow]Failed specifications:[/yellow]")
        for failed in failed_projects[:5]:  # Show first 5 failures
            console.print(f"   • {failed}")
        if len(failed_projects) > 5:
            console.print(f"   • ... and {len(failed_projects) - 5} more")
    
    console.print(f"\n💡 [dim]Next: Run Phase 2 to generate actual code files[/dim]")


async def run_phase_2_generation(config, force_regenerate=False, max_concurrent=3):
    """Run Phase 2: Synthetic Codebase Generation with parallel processing and resumability"""
    from .generation.synthetic_generator import SyntheticProjectGenerator
    from pathlib import Path
    import json
    import asyncio
    from asyncio import Semaphore
    
    console.print("\n💻 [bold]Synthetic Codebase Generation Pipeline[/bold]")
    console.print("=" * 60)
    
    # Setup progress tracking
    progress_file = Path("logs/phase2_progress.json")
    progress_file.parent.mkdir(exist_ok=True)
    completed_projects = load_progress(progress_file)
    completed_project_names = {p.get('project_name', '') for p in completed_projects}
    
    generator = SyntheticProjectGenerator(config, log_file="logs/phase2_generation.log")
    generated_dir = Path(config.data.generated_dir)
    
    # Find all project metadata files from Phase 1
    project_dirs = [d for d in generated_dir.iterdir() if d.is_dir()]
    
    console.print(f"📂 Found {len(project_dirs)} projects from Phase 1")
    console.print(f"📋 Resume state: {len(completed_projects)} projects previously completed")
    
    if force_regenerate:
        console.print("🔄 [yellow]Force mode: Regenerating ALL projects[/yellow]")
        completed_project_names = set()  # Clear resume state
    else:
        console.print("🧠 [cyan]Smart resume: Checking for completed projects...[/cyan]")
    
    if max_concurrent > 1:
        console.print(f"🚀 [bold blue]Parallel mode: {max_concurrent} concurrent projects[/bold blue]")
    
    console.print("🏭 Generating production-quality code with 3 Elite Models...")
    
    # Prepare projects for processing
    projects_to_process = []
    projects_skipped = 0
    
    for project_dir in project_dirs:
        metadata_file = project_dir / "project_metadata.json"
        
        if not metadata_file.exists():
            console.print(f"⚠️  Skipping {project_dir.name} - no metadata found")
            continue
            
        # Load project specification
        with open(metadata_file, 'r') as f:
            project_data = json.load(f)
        
        project_name = f"{project_data['specification']['name']} ({project_data['specification']['language']})"
        
        # Check if project is already completed (unless force regeneration)
        if not force_regenerate and (
            project_name in completed_project_names or 
            'generated_stats' in project_data
        ):
            stats = project_data.get('generated_stats', {})
            # Also verify files actually exist on disk
            expected_files = project_data.get('files', [])
            all_files_exist = all((project_dir / f['path']).exists() for f in expected_files)
            
            if all_files_exist and stats.get('files_count', 0) > 0:
                console.print(f"✅ [green]{project_name} - Already completed![/green]")
                projects_skipped += 1
                continue
        
        projects_to_process.append((project_dir, project_data))
    
    if not projects_to_process:
        console.print("✅ All projects already completed! Use --force to regenerate.")
        return
    
    console.print(f"🎯 Processing {len(projects_to_process)} projects ({projects_skipped} skipped)")
    
    # Semaphore to limit concurrent project generation
    semaphore = Semaphore(max_concurrent)
    
    # Statistics tracking
    total_files_generated = 0
    total_lines_generated = 0
    projects_completed = 0
    
    async def generate_single_project(project_info, project_index):
        """Generate a single project with semaphore control"""
        project_dir, project_data = project_info
        
        async with semaphore:  # Acquire semaphore slot
            spec = project_data['specification']
            project_name = f"{spec['name']} ({spec['language']})"
            
            try:
                console.print(f"🔨 [bold cyan]Starting {project_index}/{len(projects_to_process)}: {project_name}[/bold cyan]")
                
                # Extract target metrics
                target_files = spec.get('target_file_count', 10)
                target_tokens = spec.get('target_token_count', 20000)
                
                console.print(f"   🎯 Target: {target_files} files, ~{target_tokens:,} tokens")
                console.print("   🤖 3 Elite Models working...")
                
                # Start timing
                import time
                start_time = time.time()
                
                # Generate project files
                project_files = await generator.generate_project_files(spec, target_files, target_tokens)
                
                generation_time = time.time() - start_time
                console.print(f"   ⏱️  Generated in {generation_time:.1f}s")
                
                # Save generated files to project directory
                files_created = 0
                lines_created = 0
                
                console.print(f"   💾 Saving {len(project_files)} files...")
                for file_data in project_files:
                    file_path = project_dir / file_data['path']
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(file_data['content'])
                    
                    file_lines = len(file_data['content'].splitlines())
                    files_created += 1
                    lines_created += file_lines
                
                # Update project metadata with generated files
                project_data['files'] = [{'path': f['path'], 'type': f['type']} for f in project_files]
                project_data['generated_stats'] = {
                    'files_count': files_created,
                    'lines_count': lines_created,
                    'generation_time': generation_time,
                    'timestamp': time.time()
                }
                
                # Save updated metadata
                with open(project_dir / "project_metadata.json", 'w') as f:
                    json.dump(project_data, f, indent=2)
                
                console.print(f"   ✅ [green]Completed {project_name}![/green] {files_created} files, {lines_created:,} lines")
                
                # Save progress for successful completion
                import time
                current_progress = {
                    'project_name': project_name,
                    'status': 'completed',
                    'files_created': files_created,
                    'lines_created': lines_created,
                    'timestamp': time.time()
                }
                completed_projects.append(current_progress)
                save_progress(progress_file, completed_projects, "2")
                
                return {
                    'success': True,
                    'files_created': files_created,
                    'lines_created': lines_created,
                    'project_name': project_name
                }
                
            except CriticalAuthError as e:
                # Critical auth errors should stop the entire process
                console.print(f"   🚨 [bold red]CRITICAL AUTH FAILURE in {project_name}[/bold red]")
                console.print(f"   🔑 {e.provider}: {e.message}")
                console.print("   🛑 [yellow]Stopping generation to fix authentication...[/yellow]")
                
                # Save current progress before stopping
                current_progress = {
                    'project_name': project_name,
                    'status': 'auth_failed',
                    'error': str(e),
                    'timestamp': time.time()
                }
                completed_projects.append(current_progress)
                save_progress(progress_file, completed_projects, "2")
                
                # Re-raise to stop the entire process
                raise e
                
            except Exception as e:
                console.print(f"   ❌ [red]Failed {project_name}: {str(e)}[/red]")
                return {
                    'success': False,
                    'error': str(e),
                    'project_name': project_name
                }
    
    # Execute all projects in parallel with progress tracking
    console.print(f"\n🚀 [bold]Starting parallel generation of {len(projects_to_process)} projects...[/bold]")
    
    try:
        # Create tasks for all projects
        tasks = []
        for i, project_info in enumerate(projects_to_process, 1):
            task = generate_single_project(project_info, i)
            tasks.append(task)
        
        # Wait for all projects to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for CriticalAuthError in results
        for result in results:
            if isinstance(result, CriticalAuthError):
                raise result
        
        # Process results
        successful_projects = []
        failed_projects = []
        
        for result in results:
            if isinstance(result, Exception):
                failed_projects.append(f"Exception: {str(result)}")
            elif result and result['success']:
                successful_projects.append(result)
                total_files_generated += result['files_created']
                total_lines_generated += result['lines_created']
                projects_completed += 1
            else:
                failed_projects.append(result['project_name'] if result else "Unknown project")
        
        # Final summary
        console.print(f"\n📊 [bold]Phase 2 Summary:[/bold]")
        console.print(f"   ✅ Completed: {projects_completed} projects")
        console.print(f"   ⚠️  Skipped: {projects_skipped} projects (already done)")
        console.print(f"   ❌ Failed: {len(failed_projects)} projects")
        console.print(f"   📄 Total files generated: {total_files_generated:,}")
        console.print(f"   📝 Total lines generated: {total_lines_generated:,}")
        
        if failed_projects:
            console.print(f"\n⚠️  [yellow]Failed projects:[/yellow]")
            for failed in failed_projects[:10]:  # Show first 10
                console.print(f"     • {failed}")
            if len(failed_projects) > 10:
                console.print(f"     ... and {len(failed_projects) - 10} more")
                
    except CriticalAuthError as e:
        console.print(f"\n🚨 [bold red]CRITICAL AUTHENTICATION FAILURE[/bold red]")
        console.print(f"🔑 Provider: {e.provider}")
        console.print(f"💬 Error: {e.message}")
        console.print(f"\n📋 Progress saved to: {progress_file}")
        console.print(f"✅ {len(completed_projects)} projects completed before failure")
        console.print(f"\n🔧 [bold yellow]Next steps:[/bold yellow]")
        console.print("   1. Update your API credentials (check api.sh)")
        console.print("   2. Run: source api.sh")
        console.print("   3. Resume with: agentcodeeval generate --phase 2")
        console.print("   4. The pipeline will automatically resume from where it stopped")
        
        # Exit with error code
        import sys
        sys.exit(1)


async def run_phase_3_generation(config, force_regenerate=False, max_concurrent=3):
    """Run Phase 3: Agent Evaluation Scenario Creation with parallel processing"""
    from .generation.scenario_generator import ScenarioGenerator
    from .core.task import TaskCategory
    from pathlib import Path
    import json
    import asyncio
    from asyncio import Semaphore
    
    console.print("\n🎮 [bold]Agent Evaluation Scenario Creation Pipeline[/bold]")
    console.print("=" * 60)
    
    generator = ScenarioGenerator(config)
    generated_dir = Path(config.data.generated_dir)
    scenarios_dir = Path(config.data.output_dir) / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all completed projects from Phase 2
    project_dirs = [d for d in generated_dir.iterdir() if d.is_dir()]
    completed_projects = []
    
    for project_dir in project_dirs:
        metadata_file = project_dir / "project_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                project_data = json.load(f)
            # Check if project has generated code files
            if 'generated_stats' in project_data and project_data['generated_stats'].get('files_count', 0) > 0:
                completed_projects.append((project_dir, project_data))
    
    console.print(f"📂 Found {len(completed_projects)} completed projects from Phase 2")
    
    if len(completed_projects) == 0:
        console.print("⚠️  [yellow]No completed projects found. Run Phase 2 first![/yellow]")
        return
    
    if force_regenerate:
        console.print("🔄 [yellow]Force mode: Regenerating ALL scenarios[/yellow]")
    else:
        console.print("🧠 [cyan]Smart resume: Checking for completed scenarios...[/cyan]")
    
    if max_concurrent > 1:
        console.print(f"🚀 [bold blue]Parallel mode: {max_concurrent} concurrent scenario generations[/bold blue]")
    
    console.print("🎯 Creating evaluation scenarios with 3 Elite Models...")
    
    # Calculate scenario distribution based on config task_distribution
    task_categories = list(TaskCategory)
    
    # Validate that all configured task categories exist in our enum
    config_categories = set(config.benchmark.task_distribution.keys())
    enum_categories = {cat.value for cat in task_categories}
    missing_categories = config_categories - enum_categories
    if missing_categories:
        console.print(f"⚠️  [yellow]Warning: Config contains unknown task categories: {missing_categories}[/yellow]")
    
    # Create distribution map from config
    task_instance_counts = {}
    total_projects_all_languages = len(completed_projects)
    
    for task_category in task_categories:
        if task_category.value in config.benchmark.task_distribution:
            # Use configured count
            target_count = config.benchmark.task_distribution[task_category.value]
            instances_per_project = max(1, target_count // total_projects_all_languages)
            task_instance_counts[task_category] = instances_per_project
        else:
            # Fallback for missing categories
            console.print(f"⚠️  [yellow]Warning: {task_category.value} not in config task_distribution, using default[/yellow]")
            task_instance_counts[task_category] = 2
    
    # Calculate total scenarios
    total_scenarios_planned = sum(task_instance_counts.values()) * total_projects_all_languages
    
    console.print(f"📋 Task Distribution (from config):")
    for task_category, instances_per_project in task_instance_counts.items():
        total_for_category = instances_per_project * total_projects_all_languages
        console.print(f"  • {task_category.value}: {instances_per_project} per project × {total_projects_all_languages} projects = {total_for_category} total")
    
    console.print(f"🎯 Total scenarios to generate: {total_scenarios_planned}")
    
    # Prepare scenario generation tasks
    scenario_tasks = []
    scenarios_skipped = 0
    
    for project_dir, project_data in completed_projects:
        for task_category in task_categories:
            # Check if scenarios already exist for this project+category
            scenario_file = scenarios_dir / f"{project_dir.name}_{task_category.value}.json"
            
            if not force_regenerate and scenario_file.exists():
                console.print(f"✅ [green]{project_dir.name} - {task_category.value} scenarios already exist[/green]")
                scenarios_skipped += 1
                continue
            
            scenario_tasks.append({
                'project_dir': project_dir,
                'project_data': project_data,
                'task_category': task_category,
                'target_instances': task_instance_counts[task_category],
                'scenario_file': scenario_file
            })
    
    if not scenario_tasks:
        console.print("✅ All scenarios already completed! Use --force to regenerate.")
        return
    
    console.print(f"🎯 Processing {len(scenario_tasks)} scenario generation tasks ({scenarios_skipped} skipped)")
    
    # Semaphore to limit concurrent scenario generation
    semaphore = Semaphore(max_concurrent)
    
    # Statistics tracking
    total_scenarios_generated = 0
    tasks_completed = 0
    
    async def generate_scenarios_for_category(task_info, task_index):
        """Generate scenarios for one project+category combination"""
        async with semaphore:  # Acquire semaphore slot
            project_dir = task_info['project_dir']
            project_data = task_info['project_data']
            task_category = task_info['task_category']
            target_instances = task_info['target_instances']
            scenario_file = task_info['scenario_file']
            
            project_name = project_data['specification']['name']
            category_name = task_category.value
            
            try:
                console.print(f"🔨 [bold cyan]Starting {task_index}/{len(scenario_tasks)}: {project_name} - {category_name}[/bold cyan]")
                
                # Start timing
                import time
                start_time = time.time()
                
                # Generate scenarios for this project+category
                scenarios = await generator.generate_task_scenarios(
                    project_dir, project_data, task_category, target_instances
                )
                
                generation_time = time.time() - start_time
                
                # Save scenarios to file
                scenario_data = {
                    'project_name': project_name,
                    'project_id': project_dir.name,
                    'task_category': category_name,
                    'generated_timestamp': time.time(),
                    'generation_time': generation_time,
                    'scenarios': scenarios
                }
                
                with open(scenario_file, 'w') as f:
                    json.dump(scenario_data, f, indent=2)
                
                console.print(f"   ✅ [green]Completed {project_name} - {category_name}![/green] {len(scenarios)} scenarios in {generation_time:.1f}s")
                
                return {
                    'success': True,
                    'scenarios_generated': len(scenarios),
                    'project_name': project_name,
                    'category': category_name,
                    'generation_time': generation_time
                }
                
            except Exception as e:
                console.print(f"   ❌ [red]Failed {project_name} - {category_name}: {str(e)}[/red]")
                return {
                    'success': False,
                    'error': str(e),
                    'project_name': project_name,
                    'category': category_name
                }
    
    # Execute all scenario generation tasks in parallel
    console.print(f"\n🚀 [bold]Starting parallel scenario generation for {len(scenario_tasks)} tasks...[/bold]")
    
    # Create asyncio tasks
    tasks = []
    for i, task_info in enumerate(scenario_tasks, 1):
        task = generate_scenarios_for_category(task_info, i)
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    successful_tasks = []
    failed_tasks = []
    
    for result in results:
        if isinstance(result, Exception):
            failed_tasks.append(f"Exception: {str(result)}")
        elif result and result['success']:
            successful_tasks.append(result)
            total_scenarios_generated += result['scenarios_generated']
            tasks_completed += 1
        else:
            failed_tasks.append(f"{result['project_name']} - {result['category']}" if result else "Unknown task")
    
    # Final summary
    console.print(f"\n📊 [bold]Phase 3 Summary:[/bold]")
    console.print(f"   ✅ Completed: {tasks_completed} scenario generation tasks")
    console.print(f"   ⚠️  Skipped: {scenarios_skipped} tasks (already done)")
    console.print(f"   ❌ Failed: {len(failed_tasks)} tasks")
    console.print(f"   🎯 Total scenarios generated: {total_scenarios_generated}")
    console.print(f"   📁 Scenarios saved to: {scenarios_dir}")
    
    if failed_tasks:
        console.print(f"\n⚠️  [yellow]Failed tasks:[/yellow]")
        for failed in failed_tasks[:5]:  # Show first 5 failures
            console.print(f"   • {failed}")
        if len(failed_tasks) > 5:
            console.print(f"   • ... and {len(failed_tasks) - 5} more")
    
    console.print(f"\n💡 [dim]Tip: Use --force to regenerate all scenarios[/dim]")


async def run_phase_4_generation(config, force_regenerate=False, max_concurrent=3):
    """Run Phase 4: Automated Test-Driven Validation Framework with parallel processing"""
    from .generation.validation_framework import AutomatedValidator
    from .core.task import TaskCategory
    from pathlib import Path
    import json
    import asyncio
    from asyncio import Semaphore
    
    console.print("\n🧪 [bold]Automated Test-Driven Validation Framework[/bold]")
    console.print("=" * 60)
    
    validator = AutomatedValidator(config)
    scenarios_dir = Path(config.data.output_dir) / "scenarios"
    
    if not scenarios_dir.exists():
        console.print("⚠️  [yellow]No scenarios found. Run Phase 3 first![/yellow]")
        return
    
    # Find all scenario files
    scenario_files = list(scenarios_dir.glob("*.json"))
    
    if len(scenario_files) == 0:
        console.print("⚠️  [yellow]No scenario files found. Run Phase 3 first![/yellow]")
        return
    
    console.print(f"📂 Found {len(scenario_files)} scenario files from Phase 3")
    
    if force_regenerate:
        console.print("🔄 [yellow]Force mode: Regenerating ALL test suites[/yellow]")
    else:
        console.print("🧠 [cyan]Smart resume: Checking for completed test suites...[/cyan]")
    
    if max_concurrent > 1:
        console.print(f"🚀 [bold blue]Parallel mode: {max_concurrent} concurrent test suite generations[/bold blue]")
    
    console.print("🎯 Creating automated test suites for evaluation...")
    console.print(f"⚖️  Evaluation weights: Functional (40%) | Agent Metrics (30%) | Quality (20%) | Style (10%)")
    
    # Prepare test suite generation tasks
    validation_tasks = []
    test_suites_skipped = 0
    
    for scenario_file in scenario_files:
        # Check if test suite already exists
        validation_dir = Path(config.data.output_dir) / "validation" / "test_suites"
        validation_dir.mkdir(parents=True, exist_ok=True)
        
        test_suite_file = validation_dir / f"{scenario_file.stem}_test_suite.json"
        
        if not force_regenerate and test_suite_file.exists():
            console.print(f"✅ [green]{scenario_file.name} - test suite already exists[/green]")
            test_suites_skipped += 1
            continue
        
        validation_tasks.append({
            'scenario_file': scenario_file,
            'test_suite_file': test_suite_file
        })
    
    if not validation_tasks:
        console.print("✅ All test suites already completed! Use --force to regenerate.")
        return
    
    console.print(f"🎯 Processing {len(validation_tasks)} test suite generation tasks ({test_suites_skipped} skipped)")
    
    # Semaphore to limit concurrent test suite generation
    semaphore = Semaphore(max_concurrent)
    
    # Statistics tracking
    total_test_suites_generated = 0
    tasks_completed = 0
    
    async def generate_test_suite_for_scenarios(task_info, task_index):
        """Generate test suite for one scenario file"""
        async with semaphore:  # Acquire semaphore slot
            scenario_file = task_info['scenario_file']
            test_suite_file = task_info['test_suite_file']
            
            try:
                console.print(f"🔨 [bold cyan]Starting {task_index}/{len(validation_tasks)}: {scenario_file.name}[/bold cyan]")
                
                # Load scenarios
                with open(scenario_file, 'r') as f:
                    scenario_data = json.load(f)
                
                scenarios = scenario_data.get('scenarios', [])
                if not scenarios:
                    console.print(f"   ⚠️  [yellow]No scenarios found in {scenario_file.name}[/yellow]")
                    return {'success': True, 'test_suites_generated': 0, 'scenario_file': scenario_file.name}
                
                # Start timing
                import time
                start_time = time.time()
                
                # Generate test suites for all scenarios in this file
                test_suites = []
                for scenario in scenarios:
                    test_suite = await validator.generate_test_suite(scenario)
                    test_suites.append({
                        'scenario_id': scenario.get('id', 'unknown'),
                        'test_suite': test_suite.to_dict()  # Convert to dict for JSON serialization
                    })
                
                generation_time = time.time() - start_time
                
                # Save test suites
                test_suite_data = {
                    'source_file': scenario_file.name,
                    'generated_timestamp': time.time(),
                    'generation_time': generation_time,
                    'test_suites': test_suites
                }
                
                with open(test_suite_file, 'w') as f:
                    json.dump(test_suite_data, f, indent=2)
                
                console.print(f"   ✅ [green]Completed {scenario_file.name}![/green] {len(test_suites)} test suites in {generation_time:.1f}s")
                
                return {
                    'success': True,
                    'test_suites_generated': len(test_suites),
                    'scenario_file': scenario_file.name,
                    'generation_time': generation_time
                }
                
            except Exception as e:
                console.print(f"   ❌ [red]Failed {scenario_file.name}: {str(e)}[/red]")
                return {
                    'success': False,
                    'error': str(e),
                    'scenario_file': scenario_file.name
                }
    
    # Execute all test suite generation tasks in parallel
    console.print(f"\n🚀 [bold]Starting parallel test suite generation for {len(validation_tasks)} tasks...[/bold]")
    
    # Create asyncio tasks
    tasks = []
    for i, task_info in enumerate(validation_tasks, 1):
        task = generate_test_suite_for_scenarios(task_info, i)
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    successful_tasks = []
    failed_tasks = []
    
    for result in results:
        if isinstance(result, Exception):
            failed_tasks.append(f"Exception: {str(result)}")
        elif result and result['success']:
            successful_tasks.append(result)
            total_test_suites_generated += result['test_suites_generated']
            tasks_completed += 1
        else:
            failed_tasks.append(result['scenario_file'] if result else "Unknown task")
    
    # Final summary
    console.print(f"\n📊 [bold]Phase 4 Summary:[/bold]")
    console.print(f"   ✅ Completed: {tasks_completed} test suite generation tasks")
    console.print(f"   ⚠️  Skipped: {test_suites_skipped} tasks (already done)")
    console.print(f"   ❌ Failed: {len(failed_tasks)} tasks")
    console.print(f"   🧪 Total test suites generated: {total_test_suites_generated}")
    console.print(f"   📁 Test suites saved to: {Path(config.data.output_dir) / 'validation' / 'test_suites'}")
    
    if failed_tasks:
        console.print(f"\n⚠️  [yellow]Failed tasks:[/yellow]")
        for failed in failed_tasks[:5]:  # Show first 5 failures
            console.print(f"   • {failed}")
        if len(failed_tasks) > 5:
            console.print(f"   • ... and {len(failed_tasks) - 5} more")
    
    console.print(f"\n💡 [dim]Tip: Use --force to regenerate all test suites[/dim]")


if __name__ == '__main__':
    main() 