"""
Evaluation metrics for AgentCodeEval
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class AgentMetrics:
    """Container for agent evaluation metrics"""
    
    # Individual metric scores (0-1 scale)
    architectural_coherence: float = 0.0
    dependency_traversal: float = 0.0
    multi_session_memory: float = 0.0
    cross_file_reasoning: float = 0.0
    incremental_development: float = 0.0
    information_coverage: float = 0.0
    
    # Composite score (0-5 scale)
    composite_score: Optional[float] = None
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def calculate_composite_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate CADS (Composite Agent Development Score)"""
        if weights is None:
            weights = {
                "architectural_coherence": 0.2,
                "dependency_traversal": 0.2,
                "multi_session_memory": 0.2,
                "cross_file_reasoning": 0.15,
                "incremental_development": 0.15,
                "information_coverage": 0.1
            }
        
        score = (
            self.architectural_coherence * weights["architectural_coherence"] +
            self.dependency_traversal * weights["dependency_traversal"] +
            self.multi_session_memory * weights["multi_session_memory"] +
            self.cross_file_reasoning * weights["cross_file_reasoning"] +
            self.incremental_development * weights["incremental_development"] +
            self.information_coverage * weights["information_coverage"]
        )
        
        # Scale to 0-5 range
        self.composite_score = score * 5.0
        return self.composite_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "architectural_coherence": self.architectural_coherence,
            "dependency_traversal": self.dependency_traversal,
            "multi_session_memory": self.multi_session_memory,
            "cross_file_reasoning": self.cross_file_reasoning,
            "incremental_development": self.incremental_development,
            "information_coverage": self.information_coverage,
            "composite_score": self.composite_score,
            "metadata": self.metadata
        } 