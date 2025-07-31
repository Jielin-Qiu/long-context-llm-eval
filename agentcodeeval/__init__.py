"""
AgentCodeEval: A Novel Benchmark for Evaluating Long-Context Language Models 
in Software Development Agent Tasks

This package provides tools for generating, evaluating, and analyzing
software development agent performance on large-scale codebases.
"""

__version__ = "0.1.0"
__author__ = "AgentCodeEval Team"

from .core import *
from .analysis import *
from .generation import *
from .evaluation import *

__all__ = [
    "__version__",
    "__author__",
] 