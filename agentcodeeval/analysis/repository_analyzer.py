"""
Repository Analysis Pipeline for AgentCodeEval

This module implements the repository filtering and analysis pipeline
that processes The Stack v2 dataset to identify high-quality repositories
suitable for benchmark generation.
"""

import logging
import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import subprocess

import pandas as pd
from datasets import load_dataset, Dataset
from tqdm import tqdm

from ..core.config import Config
from ..core.repository import Repository
from .ast_analyzer import ASTAnalyzer
from .dependency_analyzer import DependencyAnalyzer
from .complexity_analyzer import ComplexityAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class RepositoryMetrics:
    """Repository quality and complexity metrics"""
    repo_name: str
    language: str
    file_count: int
    total_tokens: int
    complexity_score: float
    quality_score: float
    has_documentation: bool
    has_tests: bool
    architectural_patterns: List[str]
    dependency_depth: int
    star_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RepositoryFilter:
    """Filters repositories based on quality criteria"""
    
    def __init__(self, config: Config):
        self.config = config
        self.supported_languages = set(config.data.supported_languages)
        
    def filter_by_language(self, repo_data: Dict) -> bool:
        """Filter by programming language"""
        language = repo_data.get('programming_language', '').lower()
        return language in self.supported_languages
    
    def filter_by_size(self, files: List[Dict]) -> bool:
        """Filter by repository size"""
        file_count = len(files)
        return (self.config.data.min_files <= file_count <= self.config.data.max_files)
    
    def filter_by_quality(self, repo_data: Dict) -> bool:
        """Filter by repository quality indicators"""
        # Check for basic quality indicators
        has_readme = any('readme' in f.get('path', '').lower() for f in repo_data.get('files', []))
        has_license = any('license' in f.get('path', '').lower() for f in repo_data.get('files', []))
        
        # Basic quality score
        quality_indicators = [has_readme, has_license]
        quality_score = sum(quality_indicators) / len(quality_indicators)
        
        return quality_score >= 0.5  # At least 50% quality indicators
    
    def passes_all_filters(self, repo_data: Dict) -> bool:
        """Check if repository passes all filters"""
        files = repo_data.get('files', [])
        
        return (
            self.filter_by_language(repo_data) and
            self.filter_by_size(files) and
            self.filter_by_quality(repo_data)
        )


class RepositoryAnalyzer:
    """Main repository analyzer for The Stack v2"""
    
    def __init__(self, config: Config):
        self.config = config
        self.filter = RepositoryFilter(config)
        self.ast_analyzer = ASTAnalyzer()
        self.dependency_analyzer = DependencyAnalyzer()
        self.complexity_analyzer = ComplexityAnalyzer()
        
        # Create cache directories
        self.cache_dir = Path(config.data.cache_dir)
        self.analysis_cache = self.cache_dir / "analysis"
        self.repo_cache = self.cache_dir / "repositories"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_cache.mkdir(parents=True, exist_ok=True)
        self.repo_cache.mkdir(parents=True, exist_ok=True)
        
    def load_the_stack_v2(self, language: str, limit: Optional[int] = None) -> Dataset:
        """Load The Stack v2 dataset for a specific language"""
        logger.info(f"Loading The Stack v2 dataset for {language}")
        
        try:
            # Load dataset with streaming for memory efficiency
            dataset = load_dataset(
                self.config.data.stack_v2_path,
                data_dir=f"data/{language}",
                split="train",
                streaming=True
            )
            
            if limit:
                # Take only the first 'limit' samples for testing
                dataset = dataset.take(limit)
                
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset for {language}: {e}")
            raise
    
    def extract_repository_info(self, sample: Dict) -> Optional[Dict]:
        """Extract repository information from dataset sample"""
        try:
            # The Stack v2 format varies, adapt as needed
            repo_data = {
                'repo_name': sample.get('repo_name', 'unknown'),
                'programming_language': sample.get('programming_language', ''),
                'files': [],
                'content': sample.get('content', ''),
                'path': sample.get('path', ''),
                'size': sample.get('size', 0)
            }
            
            # Group files by repository
            if repo_data['path'] and repo_data['content']:
                repo_data['files'] = [{
                    'path': repo_data['path'],
                    'content': repo_data['content'],
                    'size': repo_data['size']
                }]
            
            return repo_data
            
        except Exception as e:
            logger.warning(f"Failed to extract repository info: {e}")
            return None
    
    def calculate_token_count(self, content: str) -> int:
        """Estimate token count for content"""
        # Simple estimation: ~4 characters per token
        return len(content) // 4
    
    def analyze_repository_structure(self, files: List[Dict]) -> Dict[str, Any]:
        """Analyze repository structure and patterns"""
        structure_info = {
            'has_tests': False,
            'has_docs': False,
            'has_config': False,
            'directory_structure': {},
            'file_types': {}
        }
        
        for file_info in files:
            path = file_info.get('path', '')
            
            # Check for common patterns
            if any(test_dir in path.lower() for test_dir in ['test', 'tests', 'spec']):
                structure_info['has_tests'] = True
                
            if any(doc_dir in path.lower() for doc_dir in ['doc', 'docs', 'documentation']):
                structure_info['has_docs'] = True
                
            if any(config_file in path.lower() for config_file in [
                'config', 'settings', '.env', 'package.json', 'requirements.txt'
            ]):
                structure_info['has_config'] = True
            
            # Track file types
            extension = Path(path).suffix.lower()
            structure_info['file_types'][extension] = structure_info['file_types'].get(extension, 0) + 1
        
        return structure_info
    
    def calculate_complexity_metrics(self, files: List[Dict]) -> Dict[str, float]:
        """Calculate complexity metrics for repository"""
        total_complexity = 0.0
        analyzed_files = 0
        
        for file_info in files:
            try:
                content = file_info.get('content', '')
                if content and len(content) > 100:  # Skip very small files
                    # Simple complexity estimation
                    complexity = self.complexity_analyzer.analyze_complexity_from_content(content)
                    total_complexity += complexity.get('cyclomatic_complexity', 1.0)
                    analyzed_files += 1
                    
            except Exception as e:
                logger.debug(f"Failed to analyze complexity for file: {e}")
                continue
        
        avg_complexity = total_complexity / max(analyzed_files, 1)
        
        return {
            'average_complexity': avg_complexity,
            'total_complexity': total_complexity,
            'analyzed_files': analyzed_files,
            'complexity_score': min(avg_complexity / 10.0, 1.0)  # Normalize to 0-1
        }
    
    def calculate_quality_score(self, repo_data: Dict, structure_info: Dict, complexity_metrics: Dict) -> float:
        """Calculate overall repository quality score"""
        score_components = []
        
        # Structure quality (0-1)
        structure_score = (
            structure_info['has_tests'] * 0.4 +
            structure_info['has_docs'] * 0.3 +
            structure_info['has_config'] * 0.3
        )
        score_components.append(structure_score)
        
        # Complexity quality (0-1) - moderate complexity is good
        complexity_score = complexity_metrics.get('complexity_score', 0.5)
        # Invert if too complex (>0.8) or too simple (<0.2)
        if complexity_score > 0.8:
            complexity_score = 1.0 - complexity_score
        elif complexity_score < 0.2:
            complexity_score = complexity_score * 2  # Boost simple code
        score_components.append(complexity_score)
        
        # File diversity (0-1)
        file_types = structure_info.get('file_types', {})
        diversity_score = min(len(file_types) / 5.0, 1.0)  # More file types = more diverse
        score_components.append(diversity_score)
        
        # Size quality (0-1) - prefer medium-sized repos
        file_count = len(repo_data.get('files', []))
        size_score = 1.0 - abs(file_count - 50) / 100.0  # Optimal around 50 files
        size_score = max(0.0, size_score)
        score_components.append(size_score)
        
        # Final weighted score
        return sum(score_components) / len(score_components)
    
    def analyze_single_repository(self, repo_data: Dict) -> Optional[RepositoryMetrics]:
        """Analyze a single repository and return metrics"""
        try:
            files = repo_data.get('files', [])
            if not files:
                return None
            
            # Basic metrics
            total_tokens = sum(self.calculate_token_count(f.get('content', '')) for f in files)
            
            # Structure analysis
            structure_info = self.analyze_repository_structure(files)
            
            # Complexity analysis
            complexity_metrics = self.calculate_complexity_metrics(files)
            
            # Quality score
            quality_score = self.calculate_quality_score(repo_data, structure_info, complexity_metrics)
            
            # Filter by quality threshold
            if quality_score < self.config.data.min_complexity_score:
                return None
            
            # Create metrics object
            metrics = RepositoryMetrics(
                repo_name=repo_data.get('repo_name', 'unknown'),
                language=repo_data.get('programming_language', ''),
                file_count=len(files),
                total_tokens=total_tokens,
                complexity_score=complexity_metrics.get('complexity_score', 0.5),
                quality_score=quality_score,
                has_documentation=structure_info['has_docs'],
                has_tests=structure_info['has_tests'],
                architectural_patterns=[],  # TODO: Implement pattern detection
                dependency_depth=0,  # TODO: Implement dependency analysis
                star_count=0  # TODO: Get from GitHub API if available
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to analyze repository {repo_data.get('repo_name', 'unknown')}: {e}")
            return None
    
    def process_language_dataset(self, language: str, target_repos: int = 1000) -> List[RepositoryMetrics]:
        """Process repositories for a specific language"""
        logger.info(f"Processing {language} repositories (target: {target_repos})")
        
        # Check cache first
        cache_file = self.analysis_cache / f"{language}_repositories.json"
        if cache_file.exists():
            logger.info(f"Loading cached results for {language}")
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                return [RepositoryMetrics(**item) for item in cached_data]
        
        try:
            # Load dataset
            dataset = self.load_the_stack_v2(language, limit=target_repos * 3)  # Load more for filtering
            
            processed_repos = []
            repo_groups = {}  # Group files by repository
            
            # Process samples and group by repository
            logger.info("Grouping files by repository...")
            for sample in tqdm(dataset, desc=f"Processing {language} files"):
                repo_info = self.extract_repository_info(sample)
                if not repo_info:
                    continue
                
                repo_name = repo_info['repo_name']
                if repo_name not in repo_groups:
                    repo_groups[repo_name] = {
                        'repo_name': repo_name,
                        'programming_language': repo_info['programming_language'],
                        'files': []
                    }
                
                repo_groups[repo_name]['files'].extend(repo_info['files'])
                
                # Stop if we have enough repositories
                if len(repo_groups) >= target_repos * 2:
                    break
            
            logger.info(f"Found {len(repo_groups)} repositories for {language}")
            
            # Filter and analyze repositories
            logger.info("Filtering and analyzing repositories...")
            for repo_name, repo_data in tqdm(repo_groups.items(), desc="Analyzing repos"):
                # Apply filters
                if not self.filter.passes_all_filters(repo_data):
                    continue
                
                # Analyze repository
                metrics = self.analyze_single_repository(repo_data)
                if metrics:
                    processed_repos.append(metrics)
                
                # Stop when we have enough good repositories
                if len(processed_repos) >= target_repos:
                    break
            
            logger.info(f"Successfully analyzed {len(processed_repos)} repositories for {language}")
            
            # Cache results
            with open(cache_file, 'w') as f:
                json.dump([metrics.to_dict() for metrics in processed_repos], f, indent=2)
            
            return processed_repos
            
        except Exception as e:
            logger.error(f"Failed to process {language} dataset: {e}")
            return []
    
    def run_full_analysis(self, target_repos_per_language: int = 1000) -> Dict[str, List[RepositoryMetrics]]:
        """Run full repository analysis across all supported languages"""
        logger.info("Starting full repository analysis")
        
        all_results = {}
        
        for language in self.config.data.supported_languages:
            logger.info(f"Processing language: {language}")
            
            try:
                repos = self.process_language_dataset(language, target_repos_per_language)
                all_results[language] = repos
                
                logger.info(f"Completed {language}: {len(repos)} repositories")
                
            except Exception as e:
                logger.error(f"Failed to process {language}: {e}")
                all_results[language] = []
        
        # Save summary
        self.save_analysis_summary(all_results)
        
        return all_results
    
    def save_analysis_summary(self, results: Dict[str, List[RepositoryMetrics]]):
        """Save analysis summary and statistics"""
        summary = {
            'total_repositories': sum(len(repos) for repos in results.values()),
            'languages': {},
            'quality_distribution': {},
            'complexity_distribution': {}
        }
        
        all_repos = []
        for language, repos in results.items():
            summary['languages'][language] = {
                'count': len(repos),
                'avg_quality': sum(r.quality_score for r in repos) / max(len(repos), 1),
                'avg_complexity': sum(r.complexity_score for r in repos) / max(len(repos), 1),
                'avg_tokens': sum(r.total_tokens for r in repos) / max(len(repos), 1)
            }
            all_repos.extend(repos)
        
        # Quality distribution
        quality_scores = [r.quality_score for r in all_repos]
        summary['quality_distribution'] = {
            'min': min(quality_scores) if quality_scores else 0,
            'max': max(quality_scores) if quality_scores else 0,
            'mean': sum(quality_scores) / max(len(quality_scores), 1),
            'high_quality_count': sum(1 for s in quality_scores if s > 0.7)
        }
        
        # Save summary
        summary_file = self.analysis_cache / "analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Analysis complete. Summary saved to {summary_file}")
        logger.info(f"Total repositories: {summary['total_repositories']}")
        logger.info(f"High quality repositories (>0.7): {summary['quality_distribution']['high_quality_count']}")


def run_phase_1(config: Config):
    """Run Phase 1: Repository Analysis and Selection"""
    analyzer = RepositoryAnalyzer(config)
    
    # Target: 50,000 repositories across all languages
    total_target = 50000
    languages = config.data.supported_languages
    repos_per_language = total_target // len(languages)
    
    logger.info(f"Starting Phase 1 with target: {total_target} repositories")
    logger.info(f"Target per language: {repos_per_language}")
    
    results = analyzer.run_full_analysis(repos_per_language)
    
    total_analyzed = sum(len(repos) for repos in results.values())
    logger.info(f"Phase 1 complete: {total_analyzed} repositories analyzed")
    
    return results


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    config = Config()
    
    # Run Phase 1
    results = run_phase_1(config) 