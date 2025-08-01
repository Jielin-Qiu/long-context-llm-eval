#!/usr/bin/env python3
"""
Category Evaluation Test - Test AgentCodeEval across all 8 task categories

This script systematically tests our evaluation framework across all available
task categories to ensure the metrics work correctly for different types of tasks.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from agentcodeeval.core.config import Config
from agentcodeeval.core.task import TaskCategory
from agentcodeeval.evaluation.evaluator import AgentEvaluator
from agentcodeeval.generation.synthetic_generator import MultiLLMGenerator
from agentcodeeval.utils.llm_parsing import parse_llm_response

console = Console()

async def test_category_evaluation():
    """Test evaluation across all available task categories"""
    
    # Load configuration
    config = Config("test_config.yaml")
    evaluator = AgentEvaluator(config)
    
    # Find all scenario files
    scenarios_dir = Path("data/output/scenarios")
    scenario_files = list(scenarios_dir.glob("*.json"))
    
    console.print(f"🎯 Found {len(scenario_files)} scenario files")
    
    # Group scenarios by category
    category_scenarios = {}
    all_scenarios = []
    
    for scenario_file in scenario_files:
        with open(scenario_file, 'r') as f:
            scenario_data = json.load(f)
            scenarios = scenario_data.get('scenarios', [])
            
            for scenario in scenarios:
                category = scenario.get('task_category', 'unknown')
                if category not in category_scenarios:
                    category_scenarios[category] = []
                category_scenarios[category].append(scenario)
                all_scenarios.append(scenario)
    
    console.print(f"📊 Found scenarios for {len(category_scenarios)} categories:")
    
    # Display available categories
    category_table = Table(title="Available Task Categories")
    category_table.add_column("Category", style="bold")
    category_table.add_column("Count", style="green")
    category_table.add_column("Example Title", style="dim")
    
    for category, scenarios in category_scenarios.items():
        example_title = scenarios[0].get('title', 'N/A')[:50] + "..." if len(scenarios) > 0 else "N/A"
        category_table.add_row(
            category.replace('_', ' ').title(),
            str(len(scenarios)),
            example_title
        )
    
    console.print(category_table)
    
    # Test with one model across categories
    test_model = 'openai-o3'
    console.print(f"\n🤖 Testing with model: [bold]{test_model}[/bold]")
    
    # Results storage
    category_results = {}
    total_start_time = time.time()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        main_task = progress.add_task("Testing categories", total=len(category_scenarios))
        
        for category, scenarios in category_scenarios.items():
            progress.update(main_task, description=f"🧪 Testing {category}...")
            
            # Test with first scenario from each category (for speed)
            test_scenario = scenarios[0]
            
            try:
                # Evaluate the scenario
                result = await evaluator.evaluate_model_on_scenario(test_model, test_scenario)
                
                if result:
                    category_results[category] = {
                        'success': True,
                        'total_score': result.total_score,
                        'functional_score': result.functional_score,
                        'agent_metrics_score': result.agent_metrics_score,
                        'quality_score': result.quality_score,
                        'style_score': result.style_score,
                        'generation_time': result.generation_time,
                        'scenario_title': result.scenario_title
                    }
                    
                    console.print(f"  ✅ {category}: {result.total_score:.3f}")
                else:
                    category_results[category] = {
                        'success': False,
                        'error': 'Failed to generate or evaluate solution'
                    }
                    console.print(f"  ❌ {category}: Failed")
                    
            except Exception as e:
                category_results[category] = {
                    'success': False,
                    'error': str(e)
                }
                console.print(f"  ❌ {category}: Error - {str(e)[:50]}...")
            
            progress.advance(main_task)
    
    total_time = time.time() - total_start_time
    
    # Display results summary
    console.print(f"\n🏆 Category Evaluation Summary ({total_time:.1f}s total)")
    
    results_table = Table(title="Category Performance Results")
    results_table.add_column("Category", style="bold")
    results_table.add_column("Status", style="green")
    results_table.add_column("Total Score", style="blue")
    results_table.add_column("Agent Metrics", style="purple")
    results_table.add_column("Functional", style="cyan")
    results_table.add_column("Quality", style="yellow")
    results_table.add_column("Style", style="magenta")
    results_table.add_column("Time", style="dim")
    
    successful_tests = 0
    total_score_sum = 0.0
    
    for category, result in category_results.items():
        if result.get('success', False):
            successful_tests += 1
            total_score_sum += result['total_score']
            
            results_table.add_row(
                category.replace('_', ' ').title(),
                "✅ Success",
                f"{result['total_score']:.3f}",
                f"{result['agent_metrics_score']:.3f}",
                f"{result['functional_score']:.3f}",
                f"{result['quality_score']:.3f}",
                f"{result['style_score']:.3f}",
                f"{result['generation_time']:.1f}s"
            )
        else:
            results_table.add_row(
                category.replace('_', ' ').title(),
                "❌ Failed",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A"
            )
    
    console.print(results_table)
    
    # Summary statistics
    if successful_tests > 0:
        avg_score = total_score_sum / successful_tests
        success_rate = successful_tests / len(category_scenarios)
        
        console.print(Panel.fit(f"""
🎯 [bold]Category Testing Results[/bold]

✅ Successful categories: {successful_tests}/{len(category_scenarios)} ({success_rate:.1%})
📊 Average score across categories: {avg_score:.3f}
⏱️  Total testing time: {total_time:.1f} seconds
🧠 Model tested: {test_model}

[green]The evaluation framework successfully handles multiple task categories![/green]
        """, style="bold green"))
        
        # Category-specific insights
        if successful_tests >= 3:  # Only show insights if we have enough data
            console.print("\n📊 Category Performance Insights:")
            
            sorted_results = [(cat, res) for cat, res in category_results.items() if res.get('success', False)]
            sorted_results.sort(key=lambda x: x[1]['total_score'], reverse=True)
            
            best_category = sorted_results[0] if sorted_results else None
            worst_category = sorted_results[-1] if sorted_results else None
            
            if best_category and worst_category:
                console.print(f"🥇 Best performing category: [bold]{best_category[0].replace('_', ' ').title()}[/bold] ({best_category[1]['total_score']:.3f})")
                console.print(f"📉 Lowest performing category: [bold]{worst_category[0].replace('_', ' ').title()}[/bold] ({worst_category[1]['total_score']:.3f})")
                
                # Agent metrics analysis
                agent_metrics_scores = [res['agent_metrics_score'] for cat, res in sorted_results]
                avg_agent_metrics = sum(agent_metrics_scores) / len(agent_metrics_scores)
                console.print(f"🎯 Average agent metrics score: {avg_agent_metrics:.3f}")
    
    else:
        console.print(Panel.fit("""
❌ [bold red]Category Testing Failed[/bold red]

No categories were successfully tested. This indicates a fundamental issue
with the evaluation pipeline that needs to be addressed.
        """, style="bold red"))
    
    # Save detailed results
    output_file = Path("category_evaluation_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model_tested': test_model,
                'total_categories': len(category_scenarios),
                'successful_categories': successful_tests,
                'total_time_seconds': total_time
            },
            'category_results': category_results,
            'summary': {
                'success_rate': successful_tests / len(category_scenarios) if category_scenarios else 0,
                'average_score': total_score_sum / successful_tests if successful_tests > 0 else 0
            }
        }, f, indent=2)
    
    console.print(f"💾 Detailed results saved to: {output_file}")
    
    return category_results

async def main():
    """Main function"""
    console.print(Panel.fit("🧪 AgentCodeEval Category Testing", style="bold blue"))
    
    try:
        results = await test_category_evaluation()
        
        # Show next steps
        console.print("\n🚀 Next Steps:")
        console.print("1. Review any failed categories and investigate issues")
        console.print("2. Run full evaluation with multiple models: agentcodeeval evaluate")
        console.print("3. Scale up to test all scenarios in each category")
        
    except Exception as e:
        console.print(f"❌ Category testing failed: {e}", style="bold red")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 