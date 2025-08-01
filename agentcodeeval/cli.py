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
@click.option('--force', is_flag=True, help='Force regeneration of already completed projects')
def generate(config_path, phase, dry_run, force):
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
            console.print("💻 Phase 2: Synthetic Codebase Generation", style="bold")
            if not dry_run:
                import asyncio
                asyncio.run(run_phase_2_generation(config, force_regenerate=force))
            else:
                console.print("  • Generate actual code files from project specifications")
                console.print("  • Architecture-first implementation approach")
                console.print("  • Production quality: dependencies, tests, docs")
                console.print("  • Cross-file consistency and proper imports")
                if force:
                    console.print("  • Force mode: Regenerate all projects (--force)")
                else:
                    console.print("  • Smart resume: Skip already completed projects")
                
        if phase == '3' or phase == 'all':
            console.print("🎮 Phase 3: Agent Evaluation Scenario Creation", style="bold") 
            if not dry_run:
                import asyncio
                asyncio.run(run_phase_3_generation(config, force_regenerate=force))
            else:
                console.print("  • Convert synthetic projects to evaluation scenarios")
                console.print("  • 8 task categories: debug, refactor, feature-add, test-write, etc.")
                console.print("  • Progressive complexity: 10K-500K token contexts")
                console.print(f"  • Target: 10 instances per project")
                if force:
                    console.print("  • Force mode: Regenerate all scenarios (--force)")
                else:
                    console.print("  • Smart resume: Skip already completed scenarios")
                
        if phase == '4' or phase == 'all':
            console.print("📈 Phase 4: Quality Validation & Reference Solutions", style="bold")
            if not dry_run:
                import asyncio
                asyncio.run(run_phase_4_generation(config, force))
            else:
                console.print("  • Generate automated test suites for each scenario (no LLM bias)")
                console.print("  • Implement 6 novel agent-specific metrics (ACS, DTA, MMR, CFRD, IDC, ICU)")
                console.print("  • Create functional correctness tests (compilation, unit, integration)")
                console.print("  • Automated code quality and security analysis")
                console.print("  • Evaluation weights: Functional (40%) + Agent Metrics (30%) + Quality (20%) + Style (10%)")
                if force:
                    console.print("  • Force mode: Regenerate all test suites")
                else:
                    console.print("  • Smart resume: Skip already completed test suites")
        
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
        evaluation_data = run_evaluation(config, model, task_category, difficulty)
        
        # Check if evaluation succeeded
        if not evaluation_data.get('success', False):
            console.print(f"❌ Evaluation failed: {evaluation_data.get('error', 'Unknown error')}", style="bold red")
            return
        
        # Extract results
        evaluator = evaluation_data['evaluator']
        results = evaluation_data['results']
        summaries = evaluation_data['summaries']
        
        # Display formatted results
        if summaries:
            console.print("\n📊 Evaluation Completed!", style="bold green")
            evaluator.display_results(summaries)
            
            # Save results if output file specified
            if output_file:
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


async def run_phase_2_generation(config, force_regenerate=False):
    """Run Phase 2: Synthetic Codebase Generation"""
    from .generation.synthetic_generator import SyntheticProjectGenerator
    from pathlib import Path
    import json
    
    console.print("\n💻 [bold]Synthetic Codebase Generation Pipeline[/bold]")
    console.print("=" * 60)
    
    generator = SyntheticProjectGenerator(config)
    generated_dir = Path(config.data.generated_dir)
    
    # Find all project metadata files from Phase 1
    project_dirs = [d for d in generated_dir.iterdir() if d.is_dir()]
    
    console.print(f"📂 Found {len(project_dirs)} projects from Phase 1")
    
    if force_regenerate:
        console.print("🔄 [yellow]Force mode: Regenerating ALL projects[/yellow]")
    else:
        console.print("🧠 [cyan]Smart resume: Checking for completed projects...[/cyan]")
    
    console.print("🏭 Generating production-quality code with 3 Elite Models...")
    
    total_files_generated = 0
    total_lines_generated = 0
    projects_completed = 0
    projects_skipped = 0
    
    for i, project_dir in enumerate(project_dirs, 1):
        metadata_file = project_dir / "project_metadata.json"
        
        if not metadata_file.exists():
            console.print(f"⚠️  Skipping {project_dir.name} - no metadata found")
            continue
            
        # Load project specification
        with open(metadata_file, 'r') as f:
            project_data = json.load(f)
        
        spec = project_data['specification']
        console.print(f"\n🔨 [bold cyan]Project {i}/{len(project_dirs)}: {spec['name']} ({spec['language']})[/bold cyan]")
        
        # Check if project is already completed (unless force regeneration)
        if not force_regenerate and 'generated_stats' in project_data:
            stats = project_data['generated_stats']
            # Also verify files actually exist on disk
            expected_files = project_data.get('files', [])
            all_files_exist = all((project_dir / f['path']).exists() for f in expected_files)
            
            if all_files_exist and stats.get('files_count', 0) > 0:
                console.print(f"   ✅ [green]Already completed![/green] {stats['files_count']} files, {stats['lines_count']:,} lines")
                console.print(f"   ⏭️  [dim]Skipping (use --force to regenerate)[/dim]")
                projects_skipped += 1
                continue
            else:
                console.print(f"   ⚠️  [yellow]Incomplete - some files missing, regenerating...[/yellow]")
        
        # Generate the actual code files
        try:
            # Extract target metrics
            target_files = spec.get('target_file_count', 10)
            target_tokens = spec.get('target_token_count', 20000)
            
            console.print(f"   🎯 Target: {target_files} files, ~{target_tokens:,} tokens")
            console.print("   🤖 3 Elite Models working...")
            
            # Start timing
            import time
            start_time = time.time()
            
            with console.status(f"[bold green]🏭 Generating code files for {spec['name']}..."):
                project_files = await generator.generate_project_files(spec, target_files, target_tokens)
            
            generation_time = time.time() - start_time
            console.print(f"   ⏱️  Generated in {generation_time:.1f}s")
            
            # Save generated files to project directory with progress
            files_created = 0
            lines_created = 0
            
            console.print(f"   💾 Saving {len(project_files)} files...")
            for j, file_data in enumerate(project_files, 1):
                file_path = project_dir / file_data['path']
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(file_data['content'])
                
                file_lines = len(file_data['content'].splitlines())
                files_created += 1
                lines_created += file_lines
                
                # Show progress for larger projects
                if len(project_files) > 5 and j % max(1, len(project_files) // 5) == 0:
                    console.print(f"      📄 Saved {j}/{len(project_files)} files...")
            
            # Update project metadata with generated files
            project_data['files'] = [{'path': f['path'], 'type': f['type']} for f in project_files]
            project_data['generated_stats'] = {
                'files_count': files_created,
                'lines_count': lines_created,
                'characters_count': sum(len(f['content']) for f in project_files),
                'generation_time_seconds': round(generation_time, 2)
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(project_data, f, indent=2)
            
            console.print(f"   ✅ [bold green]Success![/bold green] {files_created} files, {lines_created:,} lines, {sum(len(f['content']) for f in project_files):,} chars")
            
            total_files_generated += files_created
            total_lines_generated += lines_created
            projects_completed += 1
            
        except Exception as e:
            console.print(f"   ❌ [bold red]Failed to generate {spec['name']}: {e}[/bold red]")
            import traceback
            console.print(f"   🔍 Debug: {traceback.format_exc()}")
            continue
    
    console.print(f"\n🎉 [bold green]Phase 2 Complete![/bold green]")
    console.print(f"   📊 Generated: {total_files_generated} files across {projects_completed} projects")
    console.print(f"   ⏭️  Skipped: {projects_skipped} already completed projects")
    console.print(f"   📝 Total lines: {total_lines_generated:,}")
    console.print(f"   📁 Location: {generated_dir}")
    console.print(f"   💾 Production-quality codebases with tests & docs")
    
    if projects_skipped > 0 and not force_regenerate:
        console.print(f"\n💡 [dim]Tip: Use --force to regenerate all projects[/dim]")


async def run_phase_3_generation(config, force_regenerate=False):
    """Run Phase 3: Agent Evaluation Scenario Creation"""
    from .generation.scenario_generator import ScenarioGenerator
    from .core.task import TaskCategory
    from pathlib import Path
    import json
    
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
    
    console.print("🎯 Creating evaluation scenarios with 3 Elite Models...")
    
    # Task categories from enum
    task_categories = list(TaskCategory)
    
    # Calculate target instances per project based on configuration
    total_projects_all_languages = len(completed_projects)
    target_instances_per_project = config.benchmark.total_instances // (total_projects_all_languages * len(task_categories))
    if target_instances_per_project == 0:
        target_instances_per_project = 1  # At least 1 instance per category per project
    
    console.print(f"🎯 Target: {target_instances_per_project} instances per task category per project")
    console.print(f"📊 Total target: {len(completed_projects)} projects × {len(task_categories)} categories × {target_instances_per_project} = {len(completed_projects) * len(task_categories) * target_instances_per_project} scenarios")
    
    total_scenarios_generated = 0
    scenarios_skipped = 0
    
    for i, (project_dir, project_data) in enumerate(completed_projects, 1):
        spec = project_data['specification']
        console.print(f"\n🔨 [bold cyan]Project {i}/{len(completed_projects)}: {spec['name']} ({spec['language']})[/bold cyan]")
        
        for j, task_category in enumerate(task_categories, 1):
            console.print(f"   📋 [bold cyan]Task {j}/{len(task_categories)}: {task_category.value}[/bold cyan]")
            
            # Check if scenarios already exist (unless force regeneration)
            scenario_file = scenarios_dir / f"{project_dir.name}_{task_category.value}.json"
            
            if not force_regenerate and scenario_file.exists():
                with open(scenario_file, 'r') as f:
                    existing_scenarios = json.load(f)
                if len(existing_scenarios.get('scenarios', [])) >= target_instances_per_project:
                    existing_count = len(existing_scenarios['scenarios'])
                    console.print(f"      ✅ [green]Already completed![/green] {existing_count} scenarios")
                    console.print(f"      ⏭️  [dim]Skipping (use --force to regenerate)[/dim]")
                    scenarios_skipped += existing_count
                    continue
                else:
                    console.print(f"      ⚠️  [yellow]Incomplete - found {len(existing_scenarios.get('scenarios', []))} scenarios, need {target_instances_per_project}[/yellow]")
            
            try:
                console.print(f"      🎯 Target: {target_instances_per_project} scenarios")
                console.print(f"      🤖 3 Elite Models working...")
                
                # Start timing
                import time
                start_time = time.time()
                
                with console.status(f"[bold green]🏭 Generating {task_category.value} scenarios..."):
                    scenarios = await generator.generate_task_scenarios(
                        project_dir=project_dir,
                        project_data=project_data,
                        task_category=task_category,
                        num_instances=target_instances_per_project
                    )
                
                generation_time = time.time() - start_time
                avg_context_length = sum(s.get('context_length', 0) for s in scenarios) // len(scenarios) if scenarios else 0
                
                console.print(f"      ⏱️  Generated in {generation_time:.1f}s")
                console.print(f"      📊 Avg context: {avg_context_length:,} chars")
                
                # Save scenarios
                scenario_data = {
                    "project_info": {
                        "name": spec['name'],
                        "language": spec['language'],
                        "domain": spec['domain'],
                        "complexity": spec['complexity']
                    },
                    "task_category": task_category.value,
                    "scenarios": scenarios,
                    "generation_stats": {
                        "instances_count": len(scenarios),
                        "generation_time_seconds": round(generation_time, 2),
                        "avg_context_length": avg_context_length
                    }
                }
                
                with open(scenario_file, 'w') as f:
                    json.dump(scenario_data, f, indent=2)
                
                console.print(f"      ✅ [bold green]Success![/bold green] {len(scenarios)} scenarios, {avg_context_length:,} avg chars")
                total_scenarios_generated += len(scenarios)
                
            except Exception as e:
                console.print(f"      ❌ [bold red]Failed to generate {task_category.value}: {e}[/bold red]")
                import traceback
                console.print(f"      🔍 Debug: {traceback.format_exc()}")
                continue
    
    console.print(f"\n🎉 [bold green]Phase 3 Complete![/bold green]")
    console.print(f"   📊 Generated: {total_scenarios_generated} scenarios")
    console.print(f"   ⏭️  Skipped: {scenarios_skipped} already completed scenarios")
    console.print(f"   📁 Location: {scenarios_dir}")
    console.print(f"   🎯 8 task categories across {len(completed_projects)} projects")
    
    if scenarios_skipped > 0 and not force_regenerate:
        console.print(f"\n💡 [dim]Tip: Use --force to regenerate all scenarios[/dim]")


async def run_phase_4_generation(config, force_regenerate=False):
    """Run Phase 4: Automated Test-Driven Validation Framework"""
    from .generation.validation_framework import AutomatedValidator
    from .core.task import TaskCategory
    from pathlib import Path
    import json
    
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
    
    console.print("🎯 Creating automated test suites for evaluation...")
    console.print(f"⚖️  Evaluation weights: Functional (40%) | Agent Metrics (30%) | Quality (20%) | Style (10%)")
    
    total_test_suites_generated = 0
    test_suites_skipped = 0
    
    for i, scenario_file in enumerate(scenario_files, 1):
        console.print(f"\n🔨 [bold]File {i}/{len(scenario_files)}: {scenario_file.stem}[/bold]")
        
        # Check if test suite already exists
        test_suite_file = validator.test_suites_dir / f"{scenario_file.stem}_tests.json"
        if test_suite_file.exists() and not force_regenerate:
            console.print(f"   ⏭️  [dim]Test suite exists - skipping[/dim]")
            test_suites_skipped += 1
            continue
        
        try:
            # Load scenario to get task category
            with open(scenario_file, 'r') as f:
                scenario_data = json.load(f)
            
            task_category_str = scenario_data['task_category']
            task_category = TaskCategory(task_category_str)
            scenario_count = len(scenario_data['scenarios'])
            
            console.print(f"   📋 Task: {task_category.value}")
            console.print(f"   🎯 Scenarios: {scenario_count}")
            console.print(f"   🧪 Generating automated test suites...")
            
            start_time = time.time()
            
            # Generate test suites for each scenario in the file
            total_scenarios_in_file = 0
            for scenario in scenario_data['scenarios']:
                test_suite = await validator.generate_test_suite(scenario)
                total_scenarios_in_file += 1
            
            generation_time = time.time() - start_time
            
            console.print(f"   ⏱️  Generated in {generation_time:.1f}s")
            console.print(f"   📊 Test Categories: Compilation, Unit, Integration, Performance, Security")
            console.print(f"   ✅ [bold green]Success![/bold green] {total_scenarios_in_file} test suites generated")
            
            total_test_suites_generated += total_scenarios_in_file
            
        except Exception as e:
            console.print(f"   ❌ [bold red]Failed to generate test suites: {e}[/bold red]")
            import traceback
            console.print(f"   🔍 Debug: {traceback.format_exc()}")
            continue
    
    console.print(f"\n🎉 [bold green]Phase 4 Complete![/bold green]")
    console.print(f"   📊 Generated: {total_test_suites_generated} automated test suites")
    console.print(f"   ⏭️  Skipped: {test_suites_skipped} already completed files")
    console.print(f"   📁 Location: {validator.test_suites_dir}")
    console.print(f"   🧪 Test-Driven: No LLM bias - 100% automated validation")
    console.print(f"   📈 Novel Metrics: ACS, DTA, MMR, CFRD, IDC, ICU algorithms")
    console.print(f"   ⚖️  Evaluation Ready: Functional (40%) + Agent Metrics (30%) + Quality (20%) + Style (10%)")
    
    if test_suites_skipped > 0 and not force_regenerate:
        console.print(f"\n💡 [dim]Tip: Use --force to regenerate all test suites[/dim]")


if __name__ == '__main__':
    main() 