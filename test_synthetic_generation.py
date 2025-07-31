#!/usr/bin/env python3
"""
Test Synthetic Project Generation Pipeline for AgentCodeEval
"""

import asyncio
import os
import logging
from agentcodeeval.core.config import Config
from agentcodeeval.generation.synthetic_generator import (
    SyntheticProjectGenerator, 
    ProjectDomain, 
    ProjectComplexity
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_synthetic_generation():
    """Test the synthetic project generation pipeline"""
    print("🧪 Testing AgentCodeEval Synthetic Generation Pipeline")
    print("=" * 60)
    
    # Set up API keys for testing
    api_keys = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'), 
        'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY')
    }
    
    # Check for available API keys
    available_apis = [api for api, key in api_keys.items() if key]
    print(f"Available API keys: {available_apis}")
    
    if not available_apis:
        print("❌ No API keys found. Please set at least one:")
        print("   export OPENAI_API_KEY='your-key-here'")
        print("   export ANTHROPIC_API_KEY='your-key-here'")
        print("   export GOOGLE_API_KEY='your-key-here'")
        return False
    
    try:
        # Initialize configuration and generator
        config = Config()
        generator = SyntheticProjectGenerator(config)
        
        print(f"\n✅ Generator initialized successfully")
        print(f"Output directory: {config.data.generated_dir}")
        
        # Test 1: Generate a simple project specification
        print("\n📝 Test 1: Generating project specification...")
        
        spec = await generator.generate_project_specification(
            domain=ProjectDomain.WEB_APPLICATION,
            complexity=ProjectComplexity.EASY,
            language="python"
        )
        
        print(f"   ✅ Generated specification:")
        print(f"      Name: {spec.name}")
        print(f"      Description: {spec.description[:100]}...")
        print(f"      Target files: {spec.target_file_count}")
        print(f"      Target tokens: {spec.target_token_count:,}")
        print(f"      Features: {', '.join(spec.features[:3])}...")
        
        # Test 2: Generate project architecture
        print("\n🏗️  Test 2: Generating project architecture...")
        
        file_structure, overview = await generator.generate_project_architecture(spec)
        
        print(f"   ✅ Generated architecture:")
        print(f"      Overview: {overview[:100]}...")
        print(f"      File structure keys: {list(file_structure.keys())}")
        
        # Test 3: Generate a few sample files
        print("\n📄 Test 3: Generating sample files...")
        
        # Extract some file paths for testing
        file_paths = generator._extract_file_paths(file_structure)[:3]  # First 3 files
        
        generated_files = []
        for file_path in file_paths:
            try:
                file_obj = await generator.generate_file_content(file_path, spec, file_structure)
                generated_files.append(file_obj)
                print(f"   ✅ Generated: {file_path} ({len(file_obj.content)} chars, score: {file_obj.complexity_score})")
            except Exception as e:
                print(f"   ❌ Failed to generate {file_path}: {e}")
        
        if generated_files:
            print(f"\n   📊 Sample file content preview:")
            sample_file = generated_files[0]
            print(f"      File: {sample_file.path}")
            print(f"      Type: {sample_file.file_type}")
            print(f"      Content preview:")
            content_lines = sample_file.content.split('\n')[:5]
            for line in content_lines:
                print(f"         {line}")
            print(f"         ... ({len(sample_file.content)} total characters)")
        
        # Test 4: Generate complete small project
        print("\n🚀 Test 4: Generating complete project...")
        
        # Generate a small complete project
        complete_project = await generator.generate_complete_project(
            domain=ProjectDomain.API_SERVICE,
            complexity=ProjectComplexity.EASY,
            language="python"
        )
        
        print(f"   ✅ Generated complete project:")
        print(f"      Name: {complete_project.specification.name}")
        print(f"      Files: {len(complete_project.files)}")
        print(f"      Total content: {sum(len(f.content) for f in complete_project.files):,} characters")
        print(f"      Setup instructions: {len(complete_project.setup_instructions)} chars")
        print(f"      Test scenarios: {len(complete_project.test_scenarios)}")
        
        # Test 5: Save project to disk
        print("\n💾 Test 5: Saving project to disk...")
        
        project_path = await generator.save_project(complete_project)
        print(f"   ✅ Project saved to: {project_path}")
        
        # Verify saved files
        from pathlib import Path
        project_dir = Path(project_path)
        saved_files = list(project_dir.rglob("*"))
        print(f"   📁 Saved {len(saved_files)} files/directories")
        
        print(f"\n🎉 All tests passed! Synthetic generation pipeline is working.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def demo_multi_project_generation():
    """Demo generating multiple projects of different types"""
    print("\n🌟 Demo: Multi-project Generation")
    print("=" * 40)
    
    config = Config()
    generator = SyntheticProjectGenerator(config)
    
    # Define test scenarios
    scenarios = [
        (ProjectDomain.WEB_APPLICATION, ProjectComplexity.EASY, "javascript"),
        (ProjectDomain.DATA_PIPELINE, ProjectComplexity.MEDIUM, "python"),
        (ProjectDomain.MACHINE_LEARNING, ProjectComplexity.MEDIUM, "python"),
    ]
    
    for i, (domain, complexity, language) in enumerate(scenarios, 1):
        try:
            print(f"\n📋 Scenario {i}: {domain.value} ({complexity.value}, {language})")
            
            # Generate specification only (faster for demo)
            spec = await generator.generate_project_specification(domain, complexity, language)
            
            print(f"   ✅ {spec.name}")
            print(f"      Features: {', '.join(spec.features[:3])}...")
            print(f"      Patterns: {', '.join(spec.architecture_patterns)}")
            print(f"      Target: {spec.target_file_count} files, {spec.target_token_count:,} tokens")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print(f"\n✨ Demo complete!")

async def main():
    """Run all tests"""
    success = await test_synthetic_generation()
    
    if success:
        await demo_multi_project_generation()
        
        print(f"\n🎯 Next Steps:")
        print(f"   1. Set up your API keys in environment variables")
        print(f"   2. Run full benchmark generation with: agentcodeeval generate")
        print(f"   3. The system will generate 12,000 evaluation instances")
        print(f"   4. Each project will be ~20-100 files with realistic complexity")
    else:
        print(f"\n🔧 Please fix the issues above before proceeding.")

if __name__ == "__main__":
    asyncio.run(main()) 