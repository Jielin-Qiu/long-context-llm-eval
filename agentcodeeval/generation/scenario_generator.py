"""
Scenario Generator for AgentCodeEval Phase 3
Converts completed projects into agent evaluation scenarios
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ..core.task import TaskCategory, DifficultyLevel
from ..core.config import Config
from .synthetic_generator import MultiLLMGenerator

logger = logging.getLogger(__name__)


@dataclass
class EvaluationScenario:
    """Represents a single evaluation scenario"""
    id: str
    task_category: TaskCategory
    difficulty: DifficultyLevel
    title: str
    description: str
    context_files: List[str]
    context_length: int
    task_prompt: str
    expected_approach: str
    ground_truth: str
    evaluation_criteria: List[str]
    metadata: Dict[str, Any]


class ScenarioGenerator:
    """Main scenario generator for Phase 3"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm_generator = MultiLLMGenerator(config)
        
        # Create output directories
        self.scenarios_dir = Path(config.data.output_dir) / "scenarios"
        self.scenarios_dir.mkdir(parents=True, exist_ok=True)
    
    async def generate_task_scenarios(
        self,
        project_dir: Path,
        project_data: Dict[str, Any],
        task_category: TaskCategory,
        num_instances: int = 1
    ) -> List[Dict[str, Any]]:
        """Generate evaluation scenarios for a specific task category"""
        
        spec = project_data['specification']
        generated_stats = project_data.get('generated_stats', {})
        
        # Load project files for context
        project_files = self._load_project_files(project_dir, project_data)
        
        scenarios = []
        
        # Import Rich console for progress reporting
        from rich.console import Console
        console = Console()
        
        console.print(f"         🔄 Generating {num_instances} {task_category.value} scenarios...")
        
        for i in range(num_instances):
            scenario_id = f"{project_dir.name}_{task_category.value}_{i+1:02d}"
            
            try:
                console.print(f"         📝 Scenario {i+1}/{num_instances}: {scenario_id}", end="")
                
                scenario = await self._generate_single_scenario(
                    scenario_id=scenario_id,
                    task_category=task_category,
                    project_spec=spec,
                    project_files=project_files,
                    project_stats=generated_stats
                )
                
                # Show context info
                context_length = scenario.get('context_length', 0)
                files_count = len(scenario.get('context_files', []))
                difficulty = scenario.get('difficulty', 'unknown')
                
                console.print(f" ✅ ({difficulty}, {files_count} files, {context_length:,} chars)")
                
                scenarios.append(scenario)
                logger.info(f"Generated scenario {scenario_id}")
                
            except Exception as e:
                console.print(f" ❌ Error: {str(e)}")
                logger.error(f"Failed to generate scenario {scenario_id}: {e}")
                continue
        
        console.print(f"         🎉 [bold green]Completed {len(scenarios)}/{num_instances} scenarios[/bold green]")
        
        return scenarios
    
    def _load_project_files(self, project_dir: Path, project_data: Dict[str, Any]) -> Dict[str, str]:
        """Load all project files into memory"""
        files = {}
        
        # Load files listed in metadata
        for file_info in project_data.get('files', []):
            file_path = project_dir / file_info['path']
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        files[file_info['path']] = f.read()
                except Exception as e:
                    logger.warning(f"Could not read file {file_path}: {e}")
        
        return files
    
    async def _generate_single_scenario(
        self,
        scenario_id: str,
        task_category: TaskCategory,
        project_spec: Dict[str, Any],
        project_files: Dict[str, str],
        project_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a single evaluation scenario"""
        
        # Select files for context based on task category
        context_files = self._select_context_files(task_category, project_files)
        context_length = sum(len(content) for content in context_files.values())
        
        # Determine difficulty based on context length and complexity
        difficulty = self._determine_difficulty(context_length, project_spec.get('complexity', 'medium'))
        
        # Generate scenario using LLM
        scenario_data = await self._generate_scenario_content(
            task_category=task_category,
            difficulty=difficulty,
            project_spec=project_spec,
            context_files=context_files,
            scenario_id=scenario_id
        )
        
        return {
            "id": scenario_id,
            "task_category": task_category.value,
            "difficulty": difficulty.value,
            "title": scenario_data.get('title', f"{task_category.value.replace('_', ' ').title()} Task"),
            "description": scenario_data.get('description', ''),
            "context_files": list(context_files.keys()),
            "context_length": context_length,
            "task_prompt": scenario_data.get('task_prompt', ''),
            "expected_approach": scenario_data.get('expected_approach', ''),
            "ground_truth": scenario_data.get('ground_truth', ''),
            "evaluation_criteria": scenario_data.get('evaluation_criteria', []),
            "metadata": {
                "project_language": project_spec.get('language'),
                "project_domain": project_spec.get('domain'),
                "project_features": project_spec.get('features', []),
                "files_count": len(context_files),
                "generation_timestamp": self._get_timestamp()
            }
        }
    
    def _select_context_files(self, task_category: TaskCategory, project_files: Dict[str, str]) -> Dict[str, str]:
        """Select relevant files for the task context"""
        
        if not project_files:
            return {}
        
        # Different strategies based on task category
        if task_category == TaskCategory.ARCHITECTURAL_UNDERSTANDING:
            # Focus on main implementation files
            return self._select_files_by_pattern(project_files, ['src/', 'main.', 'app.', 'server.'])
        
        elif task_category == TaskCategory.CROSS_FILE_REFACTORING:
            # Include multiple related files
            return self._select_random_subset(project_files, min_files=3, max_files=8)
        
        elif task_category == TaskCategory.FEATURE_IMPLEMENTATION:
            # Core files where new features would be added
            return self._select_files_by_pattern(project_files, ['src/', 'lib/', 'core/'])
        
        elif task_category == TaskCategory.BUG_INVESTIGATION:
            # Mix of implementation and test files
            return self._select_files_by_pattern(project_files, ['src/', 'test', 'spec'])
        
        elif task_category == TaskCategory.MULTI_SESSION_DEVELOPMENT:
            # Broader context for multi-session work
            return self._select_random_subset(project_files, min_files=5, max_files=12)
        
        elif task_category == TaskCategory.CODE_COMPREHENSION:
            # Focus on complex implementation files
            return self._select_files_by_complexity(project_files, target_count=4)
        
        elif task_category == TaskCategory.INTEGRATION_TESTING:
            # Test files and integration points
            return self._select_files_by_pattern(project_files, ['test', 'spec', 'integration', 'api/'])
        
        elif task_category == TaskCategory.SECURITY_ANALYSIS:
            # Security-relevant files
            return self._select_files_by_pattern(project_files, ['auth', 'security', 'config', 'env'])
        
        # Default: random selection
        return self._select_random_subset(project_files, min_files=2, max_files=6)
    
    def _select_files_by_pattern(self, project_files: Dict[str, str], patterns: List[str]) -> Dict[str, str]:
        """Select files matching any of the given patterns"""
        selected = {}
        
        for file_path, content in project_files.items():
            if any(pattern.lower() in file_path.lower() for pattern in patterns):
                selected[file_path] = content
        
        # If no matches, return a random subset
        if not selected:
            return self._select_random_subset(project_files, min_files=2, max_files=4)
        
        return selected
    
    def _select_random_subset(self, project_files: Dict[str, str], min_files: int = 2, max_files: int = 6) -> Dict[str, str]:
        """Select a random subset of files"""
        file_list = list(project_files.items())
        count = min(max_files, max(min_files, len(file_list)))
        count = min(count, len(file_list))
        
        selected_items = random.sample(file_list, count)
        return dict(selected_items)
    
    def _select_files_by_complexity(self, project_files: Dict[str, str], target_count: int = 4) -> Dict[str, str]:
        """Select files based on complexity (length, lines, etc.)"""
        # Sort files by length (simple complexity metric)
        sorted_files = sorted(project_files.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Take the most complex files up to target count
        selected_count = min(target_count, len(sorted_files))
        return dict(sorted_files[:selected_count])
    
    def _determine_difficulty(self, context_length: int, project_complexity: str) -> DifficultyLevel:
        """Determine difficulty level based on context and project complexity"""
        
        # Base difficulty on context length
        if context_length < 20000:
            base_difficulty = DifficultyLevel.EASY
        elif context_length < 60000:
            base_difficulty = DifficultyLevel.MEDIUM
        elif context_length < 150000:
            base_difficulty = DifficultyLevel.HARD
        else:
            base_difficulty = DifficultyLevel.EXPERT
        
        # Adjust based on project complexity
        complexity_adjustment = {
            'easy': -1,
            'medium': 0,
            'hard': 1,
            'expert': 2
        }
        
        difficulty_levels = [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD, DifficultyLevel.EXPERT]
        current_index = difficulty_levels.index(base_difficulty)
        adjustment = complexity_adjustment.get(project_complexity.lower(), 0)
        
        new_index = max(0, min(len(difficulty_levels) - 1, current_index + adjustment))
        return difficulty_levels[new_index]
    
    async def _generate_scenario_content(
        self,
        task_category: TaskCategory,
        difficulty: DifficultyLevel,
        project_spec: Dict[str, Any],
        context_files: Dict[str, str],
        scenario_id: str
    ) -> Dict[str, Any]:
        """Generate scenario content using LLM"""
        
        # Import Rich console for debugging output
        from rich.console import Console
        console = Console()
        
        # Create context summary
        files_summary = []
        for file_path, content in context_files.items():
            lines = len(content.splitlines())
            chars = len(content)
            files_summary.append(f"- {file_path}: {lines} lines, {chars} chars")
        
        context_summary = "\n".join(files_summary)
        
        prompt = f"""
        Create a realistic {task_category.value} evaluation scenario for software development agents.
        
        PROJECT CONTEXT:
        - Name: {project_spec.get('name', 'Unknown')}
        - Language: {project_spec.get('language', 'Unknown')}
        - Domain: {project_spec.get('domain', 'Unknown')}
        - Features: {', '.join(project_spec.get('features', [])[:5])}
        - Complexity: {project_spec.get('complexity', 'medium')}
        
        AVAILABLE FILES:
        {context_summary}
        
        TASK REQUIREMENTS:
        - Category: {task_category.value}
        - Difficulty: {difficulty.value}
        - Must be realistic and challenging for AI agents
        - Should require understanding of multiple files
        - Include specific, measurable objectives
        
        Generate a JSON response with these fields:
        {{
            "title": "Clear, descriptive title for the task",
            "description": "Detailed description of the scenario and context",
            "task_prompt": "Specific task instructions for the agent",
            "expected_approach": "How an expert developer would approach this task",
            "ground_truth": "Expected solution or key insights",
            "evaluation_criteria": ["List of criteria to evaluate agent performance"]
        }}
        
        Make the scenario realistic and challenging. Focus on {self._get_category_focus(task_category)}.
        """
        
        system_prompt = f"""You are an expert software engineering instructor creating evaluation scenarios for AI development agents. Create realistic, challenging tasks that test {task_category.value} capabilities."""
        
        try:
            console.print(f"           🤖 Calling LLM for {task_category.value}...")
            response = await self.llm_generator.generate_with_model(
                self.llm_generator.generators["scenarios"],
                prompt,
                system_prompt
            )
            
            console.print(f"           📝 LLM response length: {len(response)} chars")
            logger.info(f"Raw LLM response for {scenario_id}: {response[:200]}...")
            
            # Try to parse JSON response
            try:
                parsed_response = json.loads(response)
                console.print(f"           ✅ JSON parsing successful")
                return parsed_response
            except json.JSONDecodeError as json_err:
                console.print(f"           ❌ JSON parsing failed: {str(json_err)}")
                logger.error(f"JSON parsing failed for {scenario_id}: {json_err}")
                logger.error(f"Raw response: {response}")
                
                # Try to extract JSON from response if it's wrapped in markdown
                import re
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
                if json_match:
                    try:
                        console.print(f"           🔧 Trying to extract JSON from markdown...")
                        parsed_response = json.loads(json_match.group(1))
                        console.print(f"           ✅ Markdown JSON extraction successful")
                        return parsed_response
                    except json.JSONDecodeError:
                        console.print(f"           ❌ Markdown extraction also failed")
                
                # Fallback if JSON parsing fails
                console.print(f"           🔄 Using fallback scenario")
                return self._create_fallback_scenario(task_category, difficulty, project_spec)
                
        except Exception as e:
            console.print(f"           ❌ LLM generation failed: {str(e)}")
            logger.error(f"Failed to generate scenario content for {scenario_id}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return self._create_fallback_scenario(task_category, difficulty, project_spec)
    
    def _get_category_focus(self, task_category: TaskCategory) -> str:
        """Get the focus description for each task category"""
        focus_map = {
            TaskCategory.ARCHITECTURAL_UNDERSTANDING: "system design patterns, component relationships, and architectural decisions",
            TaskCategory.CROSS_FILE_REFACTORING: "code restructuring across multiple files while maintaining functionality",
            TaskCategory.FEATURE_IMPLEMENTATION: "adding new functionality that integrates well with existing code",
            TaskCategory.BUG_INVESTIGATION: "systematic debugging, root cause analysis, and problem solving",
            TaskCategory.MULTI_SESSION_DEVELOPMENT: "incremental development over multiple sessions with context retention",
            TaskCategory.CODE_COMPREHENSION: "deep understanding of complex code structures and logic",
            TaskCategory.INTEGRATION_TESTING: "testing interactions between components and system validation",
            TaskCategory.SECURITY_ANALYSIS: "identifying security vulnerabilities and implementing security best practices"
        }
        return focus_map.get(task_category, "software development best practices")
    
    def _create_fallback_scenario(self, task_category: TaskCategory, difficulty: DifficultyLevel, project_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Create a basic fallback scenario if LLM generation fails"""
        return {
            "title": f"{task_category.value.replace('_', ' ').title()} Task",
            "description": f"A {difficulty.value} level {task_category.value} task for the {project_spec.get('name', 'project')} codebase.",
            "task_prompt": f"Please perform {task_category.value} on the provided codebase.",
            "expected_approach": "Analyze the code structure and apply appropriate development practices.",
            "ground_truth": "A well-structured solution following best practices.",
            "evaluation_criteria": [
                "Code quality and structure",
                "Adherence to requirements",
                "Use of appropriate patterns",
                "Clear documentation"
            ]
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata"""
        from datetime import datetime
        return datetime.now().isoformat() 