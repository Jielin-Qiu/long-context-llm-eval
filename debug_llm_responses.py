#!/usr/bin/env python3
"""
Debug LLM Response Parsing - Investigate fallback solution warnings

This script helps debug why some LLM responses trigger fallback generation
by capturing and analyzing the actual responses.
"""

import asyncio
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from agentcodeeval.core.config import Config
from agentcodeeval.generation.synthetic_generator import MultiLLMGenerator
from agentcodeeval.utils.llm_parsing import LLMResponseParser, parse_llm_response

console = Console()

async def debug_llm_responses():
    """Debug LLM response parsing issues"""
    
    # Load a test scenario
    scenarios_dir = Path("data/output/scenarios")
    scenario_files = list(scenarios_dir.glob("*.json"))
    
    if not scenario_files:
        console.print("❌ No scenario files found", style="bold red")
        return
    
    # Load first scenario
    with open(scenario_files[0], 'r') as f:
        scenario_data = json.load(f)
        scenarios = scenario_data.get('scenarios', [])
    
    if not scenarios:
        console.print("❌ No scenarios found in file", style="bold red")
        return
    
    test_scenario = scenarios[0]
    
    console.print(Panel.fit("🔍 Debugging LLM Response Parsing", style="bold blue"))
    console.print(f"📄 Test scenario: {test_scenario.get('title', 'Unknown')}")
    
    # Create solution prompt (same as in evaluator)
    solution_prompt = f"""You are an expert Go software engineer working on: {test_scenario.get('title', 'Development Task')}

**TASK DESCRIPTION**: {test_scenario.get('description', '')}

**REQUIREMENTS**: 
{test_scenario.get('task_prompt', '')}

**CONTEXT FILES**: {', '.join(test_scenario.get('context_files', []))}

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

    # Test with each model
    config = Config("test_config.yaml")
    llm_generator = MultiLLMGenerator(config)
    parser = LLMResponseParser()
    
    models = ['openai', 'anthropic', 'google']
    
    for model_key in models:
        console.print(f"\n🤖 Testing {model_key.upper()} response parsing...")
        
        try:
            # Generate response
            response = await llm_generator.generate_with_model(model_key, solution_prompt)
            
            console.print(f"📝 Raw response length: {len(response)} characters")
            
            # Show first 500 characters of response
            preview = response[:500] + "..." if len(response) > 500 else response
            console.print(Panel(preview, title=f"{model_key.upper()} Response Preview"))
            
            # Test parsing with detailed logging
            parsed_result = parse_llm_response(response, expected_language='go')
            
            if len(parsed_result) > 0:
                console.print(f"✅ Successfully parsed {len(parsed_result)} files")
                for filename, content in parsed_result.items():
                    console.print(f"   📄 {filename}: {len(content)} chars")
                    # Show actual content for debugging
                    if len(content) < 100:  # Show short content fully
                        console.print(f"      Content: {repr(content)}")
                    else:
                        console.print(f"      Preview: {repr(content[:100])}...")
            else:
                console.print("❌ Parsing failed - would trigger fallback")
                
                # Let's manually check what went wrong
                console.print("\n🔍 Manual Analysis:")
                
                # Check for JSON patterns
                import re
                json_patterns = [
                    r'```json\s*(\{.*?\})\s*```',
                    r'```\s*(\{.*?\})\s*```',
                    r'(\{[^{}]*"files"[^{}]*\{.*?\}[^{}]*\})',
                ]
                
                found_json = False
                for i, pattern in enumerate(json_patterns):
                    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
                    if matches:
                        console.print(f"   ✅ Found JSON with pattern {i+1}: {len(matches)} matches")
                        for j, match in enumerate(matches[:2]):  # Show first 2 matches
                            console.print(f"   📋 Match {j+1} preview:")
                            match_preview = match[:200] + "..." if len(match) > 200 else match
                            console.print(Syntax(match_preview, "json", theme="monokai"))
                        found_json = True
                        break
                
                if not found_json:
                    console.print("   ❌ No JSON patterns found")
                    
                    # Check for code blocks
                    code_patterns = [
                        r'```go\s*(.*?)\s*```',
                        r'```\s*(package\s+main.*?)\s*```',
                    ]
                    
                    for i, pattern in enumerate(code_patterns):
                        matches = re.findall(pattern, response, re.DOTALL)
                        if matches:
                            console.print(f"   ✅ Found Go code with pattern {i+1}: {len(matches)} matches")
                            break
                    else:
                        console.print("   ❌ No recognizable code patterns found")
                
                # Save problematic response for analysis
                debug_file = f"debug_response_{model_key}.txt"
                with open(debug_file, 'w') as f:
                    f.write(f"Model: {model_key}\n")
                    f.write(f"Scenario: {test_scenario.get('title')}\n")
                    f.write(f"Response length: {len(response)}\n")
                    f.write("=" * 50 + "\n")
                    f.write(response)
                
                console.print(f"💾 Saved problematic response to: {debug_file}")
            
        except Exception as e:
            console.print(f"❌ Error with {model_key}: {e}")
    
    console.print("\n🎯 Summary:")
    console.print("• If parsing failed, check the debug_response_*.txt files")
    console.print("• Look for patterns in how models format their responses")
    console.print("• Consider improving parsing patterns or prompt instructions")

async def main():
    try:
        await debug_llm_responses()
    except Exception as e:
        console.print(f"❌ Debug failed: {e}", style="bold red")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 