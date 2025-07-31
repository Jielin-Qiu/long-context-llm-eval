"""Main evaluator for AgentCodeEval"""

class AgentEvaluator:
    def __init__(self, config):
        self.config = config
    
    def evaluate(self, task, response):
        return {"score": 0.5}

def run_evaluation(config, models, categories, difficulty):
    """Run evaluation pipeline"""
    return {"results": "placeholder"} 