#!/usr/bin/env python3
"""
Debug script for Phase 3 scenario generation
Tests a single scenario generation to identify issues
"""

import asyncio
import json
from pathlib import Path
from agentcodeeval.core.config import Config
from agentcodeeval.generation.scenario_generator import ScenarioGenerator
from agentcodeeval.core.task import TaskCategory

async def debug_scenario_generation():
    """Debug a single scenario generation"""
    print("🔍 Debug: Phase 3 Scenario Generation")
    
    # Load config
    config = Config(config_path="test_config.yaml")
    
    # Create generator
    generator = ScenarioGenerator(config)
    
    # Load our completed project
    project_dir = Path("data/generated/web_application_project")
    metadata_file = project_dir / "project_metadata.json"
    
    if not metadata_file.exists():
        print("❌ No project metadata found. Run Phase 2 first!")
        return
    
    with open(metadata_file, 'r') as f:
        project_data = json.load(f)
    
    print(f"✅ Loaded project: {project_data['specification']['name']}")
    
    # Test generating one scenario
    task_category = TaskCategory.ARCHITECTURAL_UNDERSTANDING
    print(f"🎯 Testing {task_category.value} scenario generation...")
    
    try:
        scenarios = await generator.generate_task_scenarios(
            project_dir=project_dir,
            project_data=project_data,
            task_category=task_category,
            num_instances=1  # Just test one
        )
        
        if scenarios:
            print(f"✅ Generated {len(scenarios)} scenarios")
            print(f"📊 First scenario keys: {list(scenarios[0].keys())}")
            if 'title' in scenarios[0]:
                print(f"📝 Title: {scenarios[0]['title']}")
            if 'task_prompt' in scenarios[0]:
                print(f"🎯 Task prompt: {scenarios[0]['task_prompt'][:100]}...")
        else:
            print("❌ No scenarios generated")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_scenario_generation()) 