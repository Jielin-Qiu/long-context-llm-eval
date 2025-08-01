"""
AgentCodeEval Evaluation Pipeline

This module provides comprehensive evaluation capabilities for testing LLMs
on agent-specific development tasks using our automated validation framework.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID, TimeElapsedColumn

from ..core.config import Config
from ..core.task import TaskCategory, DifficultyLevel
from ..generation.validation_framework import AutomatedValidator, ValidationResult
from ..generation.synthetic_generator import MultiLLMGenerator
from ..utils.llm_parsing import parse_llm_response

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class ModelEvaluationResult:
    """Results for a single model on a single scenario"""
    model_name: str
    scenario_id: str
    scenario_title: str
    task_category: str
    difficulty: str
    
    # Core scores (from ValidationResult)
    functional_score: float
    agent_metrics_score: float
    quality_score: float
    style_score: float
    total_score: float
    
    # Additional metrics
    generation_time: float
    code_files_generated: int
    total_lines_generated: int
    parsing_success: bool
    
    # Detailed breakdown
    detailed_results: Dict[str, Any]
    timestamp: str


@dataclass
class EvaluationSummary:
    """Summary statistics for model evaluation"""
    model_name: str
    total_scenarios: int
    completed_scenarios: int
    failed_scenarios: int
    
    # Average scores
    avg_functional_score: float
    avg_agent_metrics_score: float
    avg_quality_score: float
    avg_style_score: float
    avg_total_score: float
    
    # Performance stats
    avg_generation_time: float
    total_evaluation_time: float
    parsing_success_rate: float
    
    # Category breakdown
    category_results: Dict[str, Dict[str, float]]
    difficulty_results: Dict[str, Dict[str, float]]


class AgentEvaluator:
    """Main evaluator for AgentCodeEval benchmark"""
    
    def __init__(self, config: Config):
        self.config = config
        self.validator = AutomatedValidator(config)
        self.llm_generator = MultiLLMGenerator(config)
        self.results: List[ModelEvaluationResult] = []
        
    async def evaluate_model_on_scenario(self, model_name: str, scenario: Dict[str, Any]) -> Optional[ModelEvaluationResult]:
        """Evaluate a single model on a single scenario"""
        
        scenario_id = scenario.get('id', 'unknown')
        
        try:
            # Generate solution with the model
            start_time = time.time()
            solution_code = await self._generate_solution(model_name, scenario)
            generation_time = time.time() - start_time
            
            if not solution_code:
                logger.warning(f"Model {model_name} failed to generate solution for scenario {scenario_id}")
                return None
            
            # Parse solution statistics
            code_files_count = len(solution_code)
            total_lines = sum(len(code.split('\n')) for code in solution_code.values())
            parsing_success = code_files_count > 0 and total_lines > 5  # Minimum viable solution
            
            # Generate test suite for this scenario
            test_suite = await self.validator.generate_test_suite(scenario)
            
            # Validate the solution using our framework
            validation_result = await self.validator.validate_solution(
                scenario, solution_code, test_suite
            )
            
            # Create evaluation result
            result = ModelEvaluationResult(
                model_name=model_name,
                scenario_id=scenario_id,
                scenario_title=scenario.get('title', 'Unknown'),
                task_category=scenario.get('task_category', 'unknown'),
                difficulty=scenario.get('difficulty', 'unknown'),
                
                functional_score=validation_result.functional_score,
                agent_metrics_score=validation_result.agent_metrics_score,
                quality_score=validation_result.quality_score,
                style_score=validation_result.style_score,
                total_score=validation_result.total_score,
                
                generation_time=generation_time,
                code_files_generated=code_files_count,
                total_lines_generated=total_lines,
                parsing_success=parsing_success,
                
                detailed_results=validation_result.detailed_results,
                timestamp=datetime.now().isoformat()
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Evaluation failed for model {model_name} on scenario {scenario_id}: {e}")
            return None

    async def evaluate_models(self, model_names: List[str], scenarios: List[Dict[str, Any]], 
                            task_categories: Optional[List[str]] = None,
                            difficulty_levels: Optional[List[str]] = None) -> Dict[str, List[ModelEvaluationResult]]:
        """Evaluate multiple models on multiple scenarios"""
        
        # Filter scenarios based on criteria
        filtered_scenarios = self._filter_scenarios(scenarios, task_categories, difficulty_levels)
        
        console.print(f"🎯 Evaluating {len(model_names)} models on {len(filtered_scenarios)} scenarios")
        
        results = {}
        
        for model_name in model_names:
            console.print(f"\n🤖 Evaluating model: [bold]{model_name}[/bold]")
            
            model_results = []
            failed_count = 0
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                
                task = progress.add_task(f"Evaluating {model_name}", total=len(filtered_scenarios))
                
                for i, scenario in enumerate(filtered_scenarios):
                    scenario_title = scenario.get('title', 'Unknown')[:50]
                    progress.update(task, description=f"🧪 {scenario_title}...")
                    
                    result = await self.evaluate_model_on_scenario(model_name, scenario)
                    
                    if result:
                        model_results.append(result)
                        grade = self._get_letter_grade(result.total_score)
                        console.print(f"  ✅ {scenario_title}: {result.total_score:.3f} ({grade})")
                    else:
                        failed_count += 1
                        console.print(f"  ❌ {scenario_title}: Failed")
                    
                    progress.advance(task)
            
            results[model_name] = model_results
            
            # Show model summary
            if model_results:
                avg_score = sum(r.total_score for r in model_results) / len(model_results)
                console.print(f"📊 {model_name} Summary: {len(model_results)} completed, {failed_count} failed, avg score: {avg_score:.3f}")
        
        return results

    def generate_evaluation_summary(self, results: Dict[str, List[ModelEvaluationResult]]) -> Dict[str, EvaluationSummary]:
        """Generate comprehensive evaluation summaries"""
        
        summaries = {}
        
        for model_name, model_results in results.items():
            if not model_results:
                continue
            
            # Calculate averages
            total_scenarios = len(model_results)
            completed_scenarios = len([r for r in model_results if r.parsing_success])
            failed_scenarios = total_scenarios - completed_scenarios
            
            avg_functional = sum(r.functional_score for r in model_results) / total_scenarios
            avg_agent_metrics = sum(r.agent_metrics_score for r in model_results) / total_scenarios
            avg_quality = sum(r.quality_score for r in model_results) / total_scenarios
            avg_style = sum(r.style_score for r in model_results) / total_scenarios
            avg_total = sum(r.total_score for r in model_results) / total_scenarios
            
            avg_generation_time = sum(r.generation_time for r in model_results) / total_scenarios
            parsing_success_rate = completed_scenarios / total_scenarios
            
            # Category breakdown
            category_results = {}
            for category in TaskCategory:
                category_name = category.value
                category_scores = [r for r in model_results if r.task_category == category_name]
                if category_scores:
                    category_results[category_name] = {
                        'count': len(category_scores),
                        'avg_total_score': sum(r.total_score for r in category_scores) / len(category_scores),
                        'avg_agent_metrics': sum(r.agent_metrics_score for r in category_scores) / len(category_scores)
                    }
            
            # Difficulty breakdown
            difficulty_results = {}
            for difficulty in DifficultyLevel:
                difficulty_name = difficulty.value
                difficulty_scores = [r for r in model_results if r.difficulty == difficulty_name]
                if difficulty_scores:
                    difficulty_results[difficulty_name] = {
                        'count': len(difficulty_scores),
                        'avg_total_score': sum(r.total_score for r in difficulty_scores) / len(difficulty_scores),
                        'avg_agent_metrics': sum(r.agent_metrics_score for r in difficulty_scores) / len(difficulty_scores)
                    }
            
            summary = EvaluationSummary(
                model_name=model_name,
                total_scenarios=total_scenarios,
                completed_scenarios=completed_scenarios,
                failed_scenarios=failed_scenarios,
                
                avg_functional_score=avg_functional,
                avg_agent_metrics_score=avg_agent_metrics,
                avg_quality_score=avg_quality,
                avg_style_score=avg_style,
                avg_total_score=avg_total,
                
                avg_generation_time=avg_generation_time,
                total_evaluation_time=sum(r.generation_time for r in model_results),
                parsing_success_rate=parsing_success_rate,
                
                category_results=category_results,
                difficulty_results=difficulty_results
            )
            
            summaries[model_name] = summary
        
        return summaries

    def display_results(self, summaries: Dict[str, EvaluationSummary]):
        """Display formatted evaluation results"""
        
        if not summaries:
            console.print("❌ No evaluation results to display")
            return
        
        # Overall comparison table
        console.print(Panel.fit("🏆 AgentCodeEval Results", style="bold green"))
        
        comparison_table = Table(title="Model Performance Comparison")
        comparison_table.add_column("Model", style="bold")
        comparison_table.add_column("Total Score", style="green")
        comparison_table.add_column("Grade", style="yellow")
        comparison_table.add_column("Agent Metrics", style="purple")
        comparison_table.add_column("Functional", style="blue")
        comparison_table.add_column("Quality", style="cyan")
        comparison_table.add_column("Style", style="magenta")
        comparison_table.add_column("Success Rate", style="dim")
        
        # Sort by total score
        sorted_summaries = sorted(summaries.items(), key=lambda x: x[1].avg_total_score, reverse=True)
        
        for i, (model_name, summary) in enumerate(sorted_summaries):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else ""
            model_display = f"{medal} {model_name}"
            
            comparison_table.add_row(
                model_display,
                f"{summary.avg_total_score:.3f}",
                self._get_letter_grade(summary.avg_total_score),
                f"{summary.avg_agent_metrics_score:.3f}",
                f"{summary.avg_functional_score:.3f}",
                f"{summary.avg_quality_score:.3f}",
                f"{summary.avg_style_score:.3f}",
                f"{summary.parsing_success_rate:.1%}"
            )
        
        console.print(comparison_table)
        
        # Category breakdown
        if len(summaries) > 0:
            first_summary = next(iter(summaries.values()))
            if first_summary.category_results:
                self._display_category_breakdown(summaries)

    def save_results(self, results: Dict[str, List[ModelEvaluationResult]], 
                    summaries: Dict[str, EvaluationSummary], 
                    output_file: Path):
        """Save evaluation results to file"""
        
        output_data = {
            'metadata': {
                'evaluation_timestamp': datetime.now().isoformat(),
                'framework_version': '1.0.0',
                'total_models': len(results),
                'total_scenarios': sum(len(model_results) for model_results in results.values())
            },
            'summaries': {model: asdict(summary) for model, summary in summaries.items()},
            'detailed_results': {
                model: [asdict(result) for result in model_results] 
                for model, model_results in results.items()
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        console.print(f"💾 Results saved to: {output_file}")

    # Helper methods
    
    async def _generate_solution(self, model_name: str, scenario: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Generate solution using specified model"""
        
        # Create solution prompt
        solution_prompt = f"""You are an expert Go software engineer working on: {scenario.get('title', 'Development Task')}

**TASK DESCRIPTION**: {scenario.get('description', '')}

**REQUIREMENTS**: 
{scenario.get('task_prompt', '')}

**CONTEXT FILES**: {', '.join(scenario.get('context_files', []))}

Please provide your response in EXACTLY this JSON format:

```json
{{
    "approach": "Brief explanation of your solution strategy",
    "files": {{
        "main.go": "package main\\n\\nimport \\"fmt\\"\\n\\nfunc main() {{\\n    // Your implementation here\\n}}",
        "handler.go": "package main\\n\\n// Additional file if needed"
    }},
    "explanation": "Key implementation details and design decisions"
}}
```

**REQUIREMENTS**:
1. Response MUST be valid JSON with "files" key
2. Each file MUST be complete, syntactically correct Go code
3. Use proper Go syntax, imports, and error handling
4. Follow Go best practices and naming conventions
5. Address ALL requirements in the task prompt
6. Make code production-ready and well-documented
"""

        try:
            # Map model names to our generator keys
            model_key_mapping = {
                'openai': 'openai',
                'claude': 'anthropic',
                'gemini': 'google',
                'openai-o3': 'openai',
                'claude-sonnet-4': 'anthropic',
                'gemini-2.5-pro': 'google'
            }
            
            model_key = model_key_mapping.get(model_name.lower(), 'openai')
            
            response = await self.llm_generator.generate_with_model(model_key, solution_prompt)
            
            # Parse the response using our enhanced parser
            solution_code = parse_llm_response(response, expected_language='go')
            
            return solution_code
            
        except Exception as e:
            logger.error(f"Solution generation failed for {model_name}: {e}")
            return None

    def _filter_scenarios(self, scenarios: List[Dict[str, Any]], 
                         task_categories: Optional[List[str]], 
                         difficulty_levels: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Filter scenarios based on criteria"""
        
        filtered = scenarios
        
        if task_categories:
            filtered = [s for s in filtered if s.get('task_category') in task_categories]
        
        if difficulty_levels:
            filtered = [s for s in filtered if s.get('difficulty') in difficulty_levels]
        
        return filtered

    def _get_letter_grade(self, score: float) -> str:
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

    def _display_category_breakdown(self, summaries: Dict[str, EvaluationSummary]):
        """Display category-wise performance breakdown"""
        
        console.print(Panel.fit("📊 Category Performance Breakdown", style="bold blue"))
        
        # Get all categories
        all_categories = set()
        for summary in summaries.values():
            all_categories.update(summary.category_results.keys())
        
        for category in sorted(all_categories):
            category_table = Table(title=f"{category.replace('_', ' ').title()} Category")
            category_table.add_column("Model", style="bold")
            category_table.add_column("Count", style="dim")
            category_table.add_column("Avg Score", style="green")
            category_table.add_column("Agent Metrics", style="purple")
            
            for model_name, summary in summaries.items():
                if category in summary.category_results:
                    data = summary.category_results[category]
                    category_table.add_row(
                        model_name,
                        str(data['count']),
                        f"{data['avg_total_score']:.3f}",
                        f"{data['avg_agent_metrics']:.3f}"
                    )
            
            console.print(category_table)


def run_evaluation(config: Config, models: Optional[List[str]] = None, 
                  categories: Optional[List[str]] = None, 
                  difficulty: Optional[str] = None) -> Dict[str, Any]:
    """Main evaluation function called by CLI"""
    
    async def _async_evaluation():
        evaluator = AgentEvaluator(config)
        
        # Load scenarios from Phase 3
        scenarios_dir = Path(config.data.output_dir) / "scenarios"
        if not scenarios_dir.exists():
            raise FileNotFoundError("No scenarios found. Run Phase 3 first!")
        
        # Load all scenarios
        all_scenarios = []
        for scenario_file in scenarios_dir.glob("*.json"):
            with open(scenario_file, 'r') as f:
                scenario_data = json.load(f)
                all_scenarios.extend(scenario_data.get('scenarios', []))
        
        if not all_scenarios:
            raise ValueError("No scenarios found in scenario files!")
        
        # Default models if none specified
        if not models:
            available_models = ['openai-o3', 'claude-sonnet-4', 'gemini-2.5-pro']
        else:
            available_models = list(models)
        
        # Convert difficulty to list if specified
        difficulty_levels = [difficulty] if difficulty else None
        
        # Run evaluation
        results = await evaluator.evaluate_models(
            available_models, all_scenarios, categories, difficulty_levels
        )
        
        # Generate summaries
        summaries = evaluator.generate_evaluation_summary(results)
        
        return {
            'evaluator': evaluator,
            'results': results,
            'summaries': summaries,
            'success': True
        }
    
    # Run the async evaluation
    try:
        return asyncio.run(_async_evaluation())
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'results': {},
            'summaries': {}
        } 