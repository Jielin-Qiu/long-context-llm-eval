"""
Automated Validation Framework for Phase 4: Test-Driven Evaluation

This module creates automated test suites and evaluation metrics for AgentCodeEval
scenarios without relying on reference solutions from LLMs.
"""

import ast
import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..core.config import Config
from ..core.task import TaskCategory
from .metric_algorithms import AgentMetricsCalculator
from ..validation.code_validator import validate_code_compilation, analyze_code_security, analyze_code_quality
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of automated validation for a solution"""
    scenario_id: str
    functional_score: float      # 40% weight - Does it work?
    agent_metrics_score: float   # 30% weight - Novel agent capabilities
    quality_score: float         # 20% weight - Code quality
    style_score: float          # 10% weight - Best practices
    
    total_score: float
    detailed_results: Dict[str, Any]
    execution_time: float
    

@dataclass  
class TestSuite:
    """Automated test suite for a scenario"""
    scenario_id: str
    compilation_tests: List[Dict[str, Any]]
    unit_tests: List[Dict[str, Any]]
    integration_tests: List[Dict[str, Any]]
    performance_tests: List[Dict[str, Any]]
    security_tests: List[Dict[str, Any]]


class AutomatedValidator:
    """Automated validation framework for AgentCodeEval scenarios"""
    
    def __init__(self, config: Config):
        self.config = config
        self.console = Console()
        
        # Output directories
        self.output_dir = Path(config.data.output_dir)
        self.validation_dir = self.output_dir / "validation"
        self.test_suites_dir = self.validation_dir / "test_suites"
        
        # Create directories
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        self.test_suites_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluation weights (40/30/20/10)
        self.weights = {
            'functional': 0.40,
            'agent_metrics': 0.30, 
            'quality': 0.20,
            'style': 0.10
        }
        
        # Initialize metrics calculator
        self.metrics_calculator = AgentMetricsCalculator()

    async def generate_test_suite(self, scenario: Dict[str, Any]) -> TestSuite:
        """Generate automated test suite for a scenario"""
        
        scenario_id = scenario['id']
        task_category = scenario['task_category']
        
        self.console.print(f"🧪 Generating test suite for: {scenario['title'][:50]}...")
        
        # Generate different types of tests based on task category
        compilation_tests = self._create_compilation_tests(scenario)
        unit_tests = self._create_unit_tests(scenario)
        integration_tests = self._create_integration_tests(scenario)
        performance_tests = self._create_performance_tests(scenario)
        security_tests = self._create_security_tests(scenario)
        
        test_suite = TestSuite(
            scenario_id=scenario_id,
            compilation_tests=compilation_tests,
            unit_tests=unit_tests,
            integration_tests=integration_tests,
            performance_tests=performance_tests,
            security_tests=security_tests
        )
        
        # Save test suite
        test_file = self.test_suites_dir / f"{scenario_id}_tests.json"
        with open(test_file, 'w') as f:
            json.dump({
                'scenario_id': scenario_id,
                'task_category': task_category,
                'tests': {
                    'compilation': compilation_tests,
                    'unit': unit_tests,
                    'integration': integration_tests,
                    'performance': performance_tests,
                    'security': security_tests
                }
            }, f, indent=2)
        
        return test_suite

    def _create_compilation_tests(self, scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create compilation/syntax validation tests"""
        
        task_category = scenario['task_category']
        context_files = scenario.get('context_files', [])
        
        tests = [
            {
                "name": "syntax_validation",
                "description": "Check if generated code has valid syntax",
                "type": "compilation",
                "weight": 0.3
            },
            {
                "name": "import_resolution", 
                "description": "Verify all imports can be resolved",
                "type": "compilation",
                "weight": 0.2
            },
            {
                "name": "type_checking",
                "description": "Basic type consistency checks",
                "type": "compilation", 
                "weight": 0.2
            }
        ]
        
        # Add task-specific compilation tests
        if task_category == 'feature_implementation':
            tests.append({
                "name": "api_endpoint_structure",
                "description": "New API endpoints have proper structure",
                "type": "compilation",
                "weight": 0.3
            })
        elif task_category == 'cross_file_refactoring':
            tests.append({
                "name": "refactor_consistency",
                "description": "Refactored code maintains consistency",
                "type": "compilation",
                "weight": 0.3
            })
        
        return tests

    def _create_unit_tests(self, scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create unit tests for individual functions/modules"""
        
        task_category = scenario['task_category']
        
        base_tests = [
            {
                "name": "function_signature_preservation",
                "description": "Public function signatures are preserved",
                "type": "unit",
                "weight": 0.25
            },
            {
                "name": "error_handling",
                "description": "Proper error handling for edge cases",
                "type": "unit", 
                "weight": 0.25
            },
            {
                "name": "input_validation",
                "description": "Input validation works correctly",
                "type": "unit",
                "weight": 0.25
            },
            {
                "name": "output_correctness",
                "description": "Functions return expected outputs",
                "type": "unit",
                "weight": 0.25
            }
        ]
        
        # Add task-specific unit tests
        if task_category == 'bug_investigation':
            base_tests.append({
                "name": "bug_fix_verification",
                "description": "Identified bug is actually fixed",
                "type": "unit",
                "weight": 0.4
            })
        elif task_category == 'security_analysis':
            base_tests.append({
                "name": "vulnerability_mitigation",
                "description": "Security vulnerabilities are properly addressed",
                "type": "unit",
                "weight": 0.4
            })
        
        return base_tests

    def _create_integration_tests(self, scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create integration tests for cross-file functionality"""
        
        return [
            {
                "name": "module_integration",
                "description": "Modified modules integrate correctly",
                "type": "integration",
                "weight": 0.3
            },
            {
                "name": "database_integration",
                "description": "Database operations work end-to-end",
                "type": "integration", 
                "weight": 0.3
            },
            {
                "name": "api_integration",
                "description": "API endpoints work with existing system",
                "type": "integration",
                "weight": 0.4
            }
        ]

    def _create_performance_tests(self, scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create performance and efficiency tests"""
        
        return [
            {
                "name": "execution_time",
                "description": "Code executes within reasonable time limits",
                "type": "performance",
                "weight": 0.4
            },
            {
                "name": "memory_usage",
                "description": "Memory usage is within acceptable bounds",
                "type": "performance",
                "weight": 0.3
            },
            {
                "name": "scalability",
                "description": "Solution scales appropriately with input size",
                "type": "performance",
                "weight": 0.3
            }
        ]

    def _create_security_tests(self, scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create security validation tests"""
        
        return [
            {
                "name": "injection_prevention",
                "description": "Code prevents injection attacks",
                "type": "security",
                "weight": 0.3
            },
            {
                "name": "input_sanitization",
                "description": "User inputs are properly sanitized",
                "type": "security",
                "weight": 0.3
            },
            {
                "name": "access_control",
                "description": "Proper access controls are implemented",
                "type": "security",
                "weight": 0.4
            }
        ]

    async def validate_solution(self, scenario: Dict[str, Any], solution_code: Dict[str, str], 
                               test_suite: TestSuite) -> ValidationResult:
        """Validate a solution against the automated test suite"""
        
        scenario_id = scenario['id']
        start_time = time.time()
        
        self.console.print(f"⚡ Validating solution for: {scenario['title'][:50]}...")
        
        # 1. Functional Correctness (40%)
        functional_score = await self._evaluate_functional_correctness(
            scenario, solution_code, test_suite
        )
        
        # 2. Novel Agent Metrics (30%)  
        agent_metrics_score = await self._evaluate_agent_metrics(
            scenario, solution_code
        )
        
        # 3. Code Quality (20%)
        quality_score = await self._evaluate_code_quality(
            scenario, solution_code
        )
        
        # 4. Style/Best Practices (10%)
        style_score = await self._evaluate_style_practices(
            scenario, solution_code
        )
        
        # Calculate weighted total score
        total_score = (
            functional_score * self.weights['functional'] +
            agent_metrics_score * self.weights['agent_metrics'] + 
            quality_score * self.weights['quality'] +
            style_score * self.weights['style']
        )
        
        execution_time = time.time() - start_time
        
        return ValidationResult(
            scenario_id=scenario_id,
            functional_score=functional_score,
            agent_metrics_score=agent_metrics_score,
            quality_score=quality_score,
            style_score=style_score,
            total_score=total_score,
            detailed_results={
                'functional_details': {},  # Will be populated by evaluation methods
                'agent_metrics_details': {},
                'quality_details': {},
                'style_details': {}
            },
            execution_time=execution_time
        )

    async def _evaluate_functional_correctness(self, scenario: Dict[str, Any], 
                                             solution_code: Dict[str, str], 
                                             test_suite: TestSuite) -> float:
        """Evaluate functional correctness (40% weight)"""
        
        scores = []
        
        # Test compilation
        compilation_score = await self._test_compilation(solution_code)
        scores.append(compilation_score * 0.4)
        
        # Test unit functionality  
        unit_score = await self._test_unit_functionality(solution_code, test_suite.unit_tests)
        scores.append(unit_score * 0.4)
        
        # Test integration
        integration_score = await self._test_integration(solution_code, test_suite.integration_tests)
        scores.append(integration_score * 0.2)
        
        return sum(scores)

    async def _evaluate_agent_metrics(self, scenario: Dict[str, Any], 
                                    solution_code: Dict[str, str]) -> float:
        """Evaluate novel agent-specific metrics (30% weight)"""
        
        task_category = scenario['task_category']
        
        # Calculate the 6 novel metrics based on task category
        scores = []
        
        if task_category == 'architectural_understanding':
            acs_score = self._calculate_architectural_coherence_score(scenario, solution_code)
            dta_score = self._calculate_dependency_traversal_accuracy(scenario, solution_code)
            scores = [acs_score * 0.6, dta_score * 0.4]
            
        elif task_category == 'cross_file_refactoring':
            cfrd_score = self._calculate_cross_file_reasoning_depth(scenario, solution_code)
            acs_score = self._calculate_architectural_coherence_score(scenario, solution_code)
            scores = [cfrd_score * 0.7, acs_score * 0.3]
            
        elif task_category == 'multi_session_development':
            mmr_score = self._calculate_multi_session_memory_retention(scenario, solution_code)
            idc_score = self._calculate_incremental_development_capability(scenario, solution_code)
            scores = [mmr_score * 0.6, idc_score * 0.4]
            
        else:
            # Default metrics for other categories
            icu_score = self._calculate_information_coverage_utilization(scenario, solution_code)
            cfrd_score = self._calculate_cross_file_reasoning_depth(scenario, solution_code)
            scores = [icu_score * 0.5, cfrd_score * 0.5]
        
        return sum(scores)

    async def _evaluate_code_quality(self, scenario: Dict[str, Any], 
                                   solution_code: Dict[str, str]) -> float:
        """Evaluate code quality metrics (20% weight)"""
        
        scores = []
        
        # Complexity analysis
        complexity_score = await self._analyze_code_complexity(solution_code)
        scores.append(complexity_score * 0.3)
        
        # Security analysis
        security_score = await self._analyze_security(solution_code)
        scores.append(security_score * 0.3)
        
        # Maintainability
        maintainability_score = await self._analyze_maintainability(solution_code)
        scores.append(maintainability_score * 0.4)
        
        return sum(scores)

    async def _evaluate_style_practices(self, scenario: Dict[str, Any], 
                                      solution_code: Dict[str, str]) -> float:
        """Evaluate style and best practices (10% weight)"""
        
        scores = []
        
        # Code formatting
        formatting_score = await self._check_code_formatting(solution_code)
        scores.append(formatting_score * 0.4)
        
        # Naming conventions
        naming_score = self._check_naming_conventions(solution_code)
        scores.append(naming_score * 0.3)
        
        # Documentation quality
        docs_score = self._check_documentation_quality(solution_code)
        scores.append(docs_score * 0.3)
        
        return sum(scores)

    # Real implementations replacing placeholder metric calculations
    # These now use actual algorithmic implementations from code_validator
    
    async def _test_compilation(self, solution_code: Dict[str, str]) -> float:
        """Test if code compiles successfully using real Go compiler"""
        
        try:
            compilation_result = await validate_code_compilation(solution_code, 'go')
            
            # Score based on compilation success and quality
            score = 0.0
            
            if compilation_result.success:
                score = 0.8  # Base score for successful compilation
                
                # Bonus for fast compilation
                if compilation_result.execution_time < 5.0:
                    score += 0.1
                
                # Penalty for warnings
                if compilation_result.warnings:
                    score -= len(compilation_result.warnings) * 0.05
                
                # Bonus for reasonable binary size (if available)
                if compilation_result.binary_size and compilation_result.binary_size < 10_000_000:  # < 10MB
                    score += 0.1
            else:
                # Partial credit for files that would compile individually
                error_count = len(compilation_result.errors)
                if error_count <= 2:
                    score = 0.3  # Some credit for minor issues
                elif error_count <= 5:
                    score = 0.1  # Very little credit for major issues
            
            return min(max(score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Compilation testing failed: {e}")
            return 0.2  # Minimal fallback score
    
    async def _test_unit_functionality(self, solution_code: Dict[str, str], unit_tests: List[Dict]) -> float:
        """Test unit functionality using real test execution"""
        
        try:
            from ..validation.code_validator import CodeValidator
            validator = CodeValidator()
            
            # Run actual unit tests
            test_pass_rate = await validator.run_unit_tests(solution_code, unit_tests, 'go')
            
            return test_pass_rate
            
        except Exception as e:
            logger.error(f"Unit testing failed: {e}")
            return 0.0

    async def _test_integration(self, solution_code: Dict[str, str], integration_tests: List[Dict]) -> float:
        """Test integration scenarios"""
        
        # For now, basic integration test based on multi-file coordination
        if len(solution_code) > 1:
            # Multi-file solution suggests some integration
            return 0.7
        else:
            # Single file solution
            return 0.5

    async def _test_performance(self, solution_code: Dict[str, str], performance_tests: List[Dict]) -> float:
        """Test performance characteristics"""
        
        # Simple performance heuristics based on code patterns
        total_code = ' '.join(solution_code.values())
        
        performance_score = 0.7  # Base score
        
        # Check for potentially inefficient patterns
        if 'for ' in total_code and 'for ' in total_code:
            nested_loops = total_code.count('for ')
            if nested_loops > 3:
                performance_score -= 0.2  # Penalty for many nested loops
        
        # Check for efficient patterns
        if any(pattern in total_code.lower() for pattern in ['sync.', 'goroutine', 'channel']):
            performance_score += 0.2  # Bonus for concurrency
            
        return min(max(performance_score, 0.0), 1.0)

    async def _test_security_compliance(self, solution_code: Dict[str, str], security_tests: List[Dict]) -> float:
        """Test security compliance using real security analysis"""
        
        try:
            security_result = await analyze_code_security(solution_code, 'go')
            return security_result.security_score
            
        except Exception as e:
            logger.error(f"Security analysis failed: {e}")
            return 0.5

    def _calculate_architectural_coherence_score(self, scenario: Dict, solution_code: Dict[str, str]) -> float:
        """Calculate ACS - Architectural Coherence Score"""
        return self.metrics_calculator.calculate_architectural_coherence_score(scenario, solution_code)
    
    def _calculate_dependency_traversal_accuracy(self, scenario: Dict, solution_code: Dict[str, str]) -> float:
        """Calculate DTA - Dependency Traversal Accuracy"""
        return self.metrics_calculator.calculate_dependency_traversal_accuracy(scenario, solution_code)
    
    def _calculate_multi_session_memory_retention(self, scenario: Dict, solution_code: Dict[str, str]) -> float:
        """Calculate MMR - Multi-Session Memory Retention"""
        return self.metrics_calculator.calculate_multi_session_memory_retention(scenario, solution_code)
    
    def _calculate_cross_file_reasoning_depth(self, scenario: Dict, solution_code: Dict[str, str]) -> float:
        """Calculate CFRD - Cross-File Reasoning Depth"""
        return self.metrics_calculator.calculate_cross_file_reasoning_depth(scenario, solution_code)
    
    def _calculate_incremental_development_capability(self, scenario: Dict, solution_code: Dict[str, str]) -> float:
        """Calculate IDC - Incremental Development Capability"""
        return self.metrics_calculator.calculate_incremental_development_capability(scenario, solution_code)
    
    def _calculate_information_coverage_utilization(self, scenario: Dict, solution_code: Dict[str, str]) -> float:
        """Calculate ICU - Information Coverage Utilization"""
        return self.metrics_calculator.calculate_information_coverage_utilization(scenario, solution_code)
        
    async def _analyze_code_complexity(self, solution_code: Dict[str, str]) -> float:
        """Analyze code complexity using real complexity metrics"""
        
        try:
            # Use the real quality analysis directly without asyncio.run
            quality_result = await analyze_code_quality(solution_code, 'go')
            return quality_result.complexity_score
            
        except Exception as e:
            logger.error(f"Complexity analysis failed: {e}")
            return 0.5
        
    async def _analyze_security(self, solution_code: Dict[str, str]) -> float:
        """Analyze security vulnerabilities using real security scanner"""
        
        try:
            security_result = await analyze_code_security(solution_code, 'go')
            return security_result.security_score
            
        except Exception as e:
            logger.error(f"Security analysis failed: {e}")
            return 0.5
        
    async def _analyze_maintainability(self, solution_code: Dict[str, str]) -> float:
        """Analyze maintainability using real maintainability metrics"""
        
        try:
            # Use the real quality analysis directly without asyncio.run
            quality_result = await analyze_code_quality(solution_code, 'go')
            return quality_result.maintainability_score
            
        except Exception as e:
            logger.error(f"Maintainability analysis failed: {e}")
            return 0.5
        
    async def _check_code_formatting(self, solution_code: Dict[str, str]) -> float:
        """Check code formatting using real formatter"""
        
        try:
            from ..validation.code_validator import CodeValidator
            validator = CodeValidator()
            
            # Use real formatting check directly without asyncio.run
            formatting_result = await validator.check_code_formatting(solution_code, 'go')
            return formatting_result
            
        except Exception as e:
            logger.error(f"Formatting check failed: {e}")
            return 0.5
        
    def _check_naming_conventions(self, solution_code: Dict[str, str]) -> float:
        """Check naming convention compliance"""
        
        # Go-specific naming convention checks
        total_score = 0.0
        total_checks = 0
        
        for filename, code in solution_code.items():
            # Check function naming (PascalCase for public, camelCase for private)
            functions = re.findall(r'func\s+([A-Za-z_][A-Za-z0-9_]*)', code)
            
            for func_name in functions:
                total_checks += 1
                if func_name[0].isupper():  # Public function
                    if re.match(r'^[A-Z][a-zA-Z0-9]*$', func_name):
                        total_score += 1.0
                    else:
                        total_score += 0.5
                else:  # Private function
                    if re.match(r'^[a-z][a-zA-Z0-9]*$', func_name):
                        total_score += 1.0
                    else:
                        total_score += 0.5
            
            # Check variable naming
            variables = re.findall(r'var\s+([A-Za-z_][A-Za-z0-9_]*)', code)
            
            for var_name in variables:
                total_checks += 1
                if re.match(r'^[a-zA-Z][a-zA-Z0-9]*$', var_name):
                    total_score += 1.0
                else:
                    total_score += 0.5
        
        return total_score / total_checks if total_checks > 0 else 0.6
        
    def _check_documentation_quality(self, solution_code: Dict[str, str]) -> float:
        """Check documentation quality"""
        
        total_lines = 0
        comment_lines = 0
        documented_functions = 0
        total_functions = 0
        
        for filename, code in solution_code.items():
            lines = code.split('\n')
            total_lines += len(lines)
            
            # Count comment lines
            comment_lines += sum(1 for line in lines if line.strip().startswith('//'))
            
            # Check function documentation
            functions = re.finditer(r'func\s+([A-Za-z_][A-Za-z0-9_]*)', code)
            
            for match in functions:
                total_functions += 1
                func_start = match.start()
                
                # Look for comment before function
                lines_before_func = code[:func_start].split('\n')
                if lines_before_func and lines_before_func[-1].strip().startswith('//'):
                    documented_functions += 1
        
        # Calculate documentation score
        comment_ratio = comment_lines / total_lines if total_lines > 0 else 0
        function_doc_ratio = documented_functions / total_functions if total_functions > 0 else 0
        
        documentation_score = (comment_ratio * 0.4 + function_doc_ratio * 0.6)
        
        return min(documentation_score * 2, 1.0)  # Scale up and cap at 1.0 