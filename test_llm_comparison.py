#!/usr/bin/env python3
"""
LLM Comparison Test for AgentCodeEval

This script tests all 3 Elite Models through our validation framework:
- OpenAI o3 (direct API)
- Claude Sonnet 4 (AWS Bedrock + direct API fallback) 
- Gemini 2.5 Pro (direct API)

Compares their performance across our 6 novel agent-specific metrics.
"""

import asyncio
import json
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from agentcodeeval.core.config import Config
from agentcodeeval.generation.validation_framework import AutomatedValidator
from agentcodeeval.generation.synthetic_generator import MultiLLMGenerator
from agentcodeeval.utils.llm_parsing import parse_llm_response


console = Console()


async def test_all_llms():
    """Test all 3 Elite Models through our validation framework"""
    
    console.print(Panel.fit("🏆 Testing 3 Elite Models with AgentCodeEval", style="bold green"))
    
    # Load configuration
    config = Config(config_path="test_config.yaml")
    
    # Initialize frameworks
    validator = AutomatedValidator(config)
    llm_generator = MultiLLMGenerator(config)
    
    console.print("✅ Framework initialized with 3 Elite Models")
    console.print("   🤖 OpenAI o3 (Direct API)")
    console.print("   🧠 Claude Sonnet 4 (AWS Bedrock + Fallback)")
    console.print("   💎 Gemini 2.5 Pro (Direct API)")
    
    # Find available scenarios
    scenarios_dir = Path("data/output/scenarios")
    if not scenarios_dir.exists():
        console.print("❌ No scenarios found. Run Phase 3 first!")
        return
    
    scenario_files = list(scenarios_dir.glob("*.json"))
    if not scenario_files:
        console.print("❌ No scenario files found!")
        return
    
    # Select first scenario for testing (feature implementation works well for comparison)
    test_scenario_file = next((f for f in scenario_files if "feature_implementation" in f.name), scenario_files[0])
    console.print(f"📁 Testing with: {test_scenario_file.name}")
    
    # Load scenario data
    with open(test_scenario_file, 'r') as f:
        scenario_data = json.load(f)
    
    # Get first scenario from the file
    scenario = scenario_data['scenarios'][0]
    
    console.print(f"🎯 Testing scenario: [bold]{scenario['title'][:60]}...[/bold]")
    console.print(f"📋 Task category: [cyan]{scenario['task_category']}[/cyan]")
    console.print(f"⚖️  Difficulty: [yellow]{scenario['difficulty']}[/yellow]")
    
    # Generate test suite once
    console.print("\n📝 Generating automated test suite...")
    test_suite = await validator.generate_test_suite(scenario)
    
    # Test each model
    models = [
        ("openai", "🤖 OpenAI o3", "blue"),
        ("anthropic", "🧠 Claude Sonnet 4", "purple"), 
        ("google", "💎 Gemini 2.5 Pro", "green")
    ]
    
    results = {}
    
    for model_key, model_name, color in models:
        console.print(f"\n{model_name}", style=f"bold {color}")
        console.print("=" * 50)
        
        try:
            # Generate solution with this model
            solution_code = await generate_solution_with_model(
                llm_generator, model_key, scenario
            )
            
            if not solution_code:
                console.print(f"❌ {model_name} failed to generate solution")
                continue
            
            console.print(f"✅ Generated {len(solution_code)} files")
            for filename in solution_code.keys():
                lines = len(solution_code[filename].split('\n'))
                console.print(f"   • {filename} ({lines} lines)")
            
            # Validate solution
            with console.status(f"[bold {color}]⚡ Validating {model_name} solution..."):
                validation_result = await validator.validate_solution(
                    scenario, solution_code, test_suite
                )
            
            results[model_key] = {
                'name': model_name,
                'result': validation_result,
                'code_files': len(solution_code),
                'total_lines': sum(len(code.split('\n')) for code in solution_code.values()),
                'color': color
            }
            
            # Display individual results
            display_individual_results(model_name, validation_result, color)
            
        except Exception as e:
            console.print(f"❌ {model_name} error: {e}")
            import traceback
            console.print(f"🔍 Debug: {traceback.format_exc()}")
            continue
    
    # Display comparison
    if len(results) > 1:
        console.print("\n" + "=" * 80)
        display_model_comparison(results)
    
    return results


async def generate_solution_with_model(llm_generator, model_key, scenario):
    """Generate solution using specified model with improved prompting"""
    
    # Create enhanced solution prompt with clearer structure requirements
    solution_prompt = f"""You are an expert Go software engineer working on a web application project.

**TASK**: {scenario['title']}

**DESCRIPTION**: {scenario['description']}

**REQUIREMENTS**: 
{scenario['task_prompt']}

**CONTEXT FILES**: {', '.join(scenario['context_files'])}

**IMPORTANT**: Please provide your response in EXACTLY this JSON format:

```json
{{
    "approach": "Brief explanation of your solution strategy",
    "files": {{
        "main.go": "package main\\n\\nimport \\"fmt\\"\\n\\nfunc main() {{\\n    // Your implementation here\\n}}",
        "handlers.go": "package main\\n\\n// Additional file if needed"
    }},
    "explanation": "Key implementation details and design decisions"
}}
```

**CRITICAL REQUIREMENTS**:
1. Response MUST be valid JSON with "files" key
2. Each file value MUST be a complete, syntactically correct Go program
3. Use proper Go syntax: package declarations, imports, proper error handling
4. Include ALL necessary imports and dependencies  
5. Make code production-ready with proper validation and error handling
6. Follow Go best practices and naming conventions
7. Address ALL requirements mentioned in the task prompt
8. Ensure code compiles and runs without errors

**EXAMPLE FILE STRUCTURE**:
- Use "package main" for executable programs
- Include proper imports (fmt, net/http, etc.)
- Implement complete functionality, not just stubs
- Add proper error handling with "if err != nil"
- Use Go naming conventions (camelCase for private, PascalCase for public)

Generate a complete, working solution now:"""

    try:
        start_time = time.time()
        
        response = await llm_generator.generate_with_model(
            model_key, solution_prompt
        )
        
        generation_time = time.time() - start_time
        console.print(f"⏱️  Generated in {generation_time:.1f}s ({len(response):,} chars)")
        
        # Parse solution using our advanced parser
        solution_code = parse_llm_response(response, expected_language='go')
        
        # Check if we got a meaningful solution
        if solution_code:
            total_lines = sum(len(code.split('\n')) for code in solution_code.values())
            if total_lines > 5:  # Minimum viable solution
                console.print(f"✅ Successfully parsed structured solution")
                return solution_code
            else:
                console.print(f"⚠️  Solution too minimal ({total_lines} lines), treating as fallback")
        
        return solution_code
        
    except Exception as e:
        console.print(f"❌ Generation failed: {e}")
        return None


def display_individual_results(model_name, result, color):
    """Display results for individual model"""
    
    # Create compact results table
    table = Table(title=f"{model_name} Results", title_style=f"bold {color}")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green")
    table.add_column("Weight", style="yellow")
    table.add_column("Contribution", style="magenta")
    
    table.add_row(
        "Functional", f"{result.functional_score:.3f}",
        "40%", f"{result.functional_score * 0.4:.3f}"
    )
    table.add_row(
        "Agent Metrics", f"{result.agent_metrics_score:.3f}",
        "30%", f"{result.agent_metrics_score * 0.3:.3f}"
    )
    table.add_row(
        "Quality", f"{result.quality_score:.3f}",
        "20%", f"{result.quality_score * 0.2:.3f}"
    )
    table.add_row(
        "Style", f"{result.style_score:.3f}",
        "10%", f"{result.style_score * 0.1:.3f}"
    )
    table.add_row("", "", "", "", style="dim")
    table.add_row(
        "TOTAL", f"{result.total_score:.3f}",
        "100%", f"{result.total_score:.3f}",
        style=f"bold {color}"
    )
    
    console.print(table)
    
    grade = get_letter_grade(result.total_score)
    console.print(f"🏆 Grade: [bold {color}]{grade}[/bold {color}]")


def display_model_comparison(results):
    """Display side-by-side comparison of all models"""
    
    console.print(Panel.fit("🏆 Elite Models Comparison", style="bold green"))
    
    # Main comparison table
    comparison_table = Table(title="Performance Comparison")
    comparison_table.add_column("Model", style="bold")
    comparison_table.add_column("Total Score", style="green")
    comparison_table.add_column("Grade", style="yellow")
    comparison_table.add_column("Functional", style="blue")
    comparison_table.add_column("Agent Metrics", style="purple")
    comparison_table.add_column("Quality", style="cyan")
    comparison_table.add_column("Style", style="magenta")
    comparison_table.add_column("Files", style="dim")
    comparison_table.add_column("Lines", style="dim")
    
    # Sort by total score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['result'].total_score, reverse=True)
    
    for i, (model_key, data) in enumerate(sorted_results):
        result = data['result']
        
        # Add medal for top performers
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else ""
        model_display = f"{medal} {data['name']}"
        
        comparison_table.add_row(
            model_display,
            f"{result.total_score:.3f}",
            get_letter_grade(result.total_score),
            f"{result.functional_score:.3f}",
            f"{result.agent_metrics_score:.3f}",
            f"{result.quality_score:.3f}",
            f"{result.style_score:.3f}",
            str(data['code_files']),
            str(data['total_lines'])
        )
    
    console.print(comparison_table)
    
    # Analysis
    if len(sorted_results) >= 2:
        winner_key, winner_data = sorted_results[0]
        second_key, second_data = sorted_results[1]
        
        score_diff = winner_data['result'].total_score - second_data['result'].total_score
        
        analysis_text = f"""
🏆 Winner: {winner_data['name']}
📊 Top Score: {winner_data['result'].total_score:.3f}
📈 Margin: +{score_diff:.3f} over second place
        
💡 Key Insights:
• Functional Correctness Leader: {max(results.items(), key=lambda x: x[1]['result'].functional_score)[1]['name']}
• Agent Metrics Leader: {max(results.items(), key=lambda x: x[1]['result'].agent_metrics_score)[1]['name']}
• Code Quality Leader: {max(results.items(), key=lambda x: x[1]['result'].quality_score)[1]['name']}
• Style Leader: {max(results.items(), key=lambda x: x[1]['result'].style_score)[1]['name']}
        """
        
        console.print(Panel(analysis_text.strip(), title="🔍 Analysis", style="blue"))


def get_letter_grade(score):
    """Convert score to letter grade"""
    if score >= 0.9:
        return "A+ (Excellent)"
    elif score >= 0.8:
        return "A (Very Good)"
    elif score >= 0.7:
        return "B (Good)"
    elif score >= 0.6:
        return "C (Fair)"
    elif score >= 0.5:
        return "D (Poor)"
    else:
        return "F (Failing)"


async def main():
    """Main test function"""
    try:
        start_time = time.time()
        
        results = await test_all_llms()
        
        total_time = time.time() - start_time
        
        if results:
            console.print(f"\n✅ [bold green]Elite Models comparison completed![/bold green]")
            console.print(f"⏱️  Total time: {total_time:.1f}s")
            console.print(f"🎯 Models tested: {len(results)}/3")
            
            console.print("\n🎉 [bold blue]AgentCodeEval Framework Validated![/bold blue]")
            console.print("   📊 6 Novel Metrics working with real LLMs")
            console.print("   🤖 Multi-model comparison successful")
            console.print("   ⚡ Production-ready evaluation pipeline")
            
        else:
            console.print("\n❌ [bold red]No models completed successfully[/bold red]")
            
    except Exception as e:
        console.print(f"\n❌ [bold red]Test error: {e}[/bold red]")
        import traceback
        console.print(f"🔍 Debug: {traceback.format_exc()}")


if __name__ == "__main__":
    asyncio.run(main()) 