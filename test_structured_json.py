#!/usr/bin/env python3
"""
Test Structured JSON - Debug why properly formatted JSON isn't being recognized
"""

import re
import json
from agentcodeeval.utils.llm_parsing import LLMResponseParser

def test_structured_json():
    """Test the structured JSON patterns"""
    
    test_response = '''```json
{
    "files": {
        "main.go": "package main\\n\\nfunc main() { fmt.Println(\\"Hello\\") }"
    }
}
```'''

    print("🧪 Testing structured JSON patterns...")
    
    parser = LLMResponseParser()
    
    # Check the actual patterns used
    print(f"📋 JSON patterns in parser:")
    for i, pattern in enumerate(parser.json_patterns):
        print(f"   Pattern {i+1}: {pattern}")
    
    # Test each pattern manually
    print(f"\n🔍 Testing patterns against our response:")
    for i, pattern in enumerate(parser.json_patterns):
        matches = re.findall(pattern, test_response, re.DOTALL | re.IGNORECASE)
        print(f"   Pattern {i+1}: {len(matches)} matches")
        
        if matches:
            for j, match in enumerate(matches):
                print(f"      Match {j+1} ({len(match)} chars): {repr(match[:100])}")
                
                # Try to parse the match
                try:
                    cleaned = parser._clean_json_string(match)
                    data = json.loads(cleaned)
                    files = parser._extract_files_from_data(data)
                    print(f"      ✅ Successfully parsed: {len(files) if files else 0} files")
                    if files:
                        for filename, content in files.items():
                            print(f"         📄 {filename}: {len(content)} chars")
                except Exception as e:
                    print(f"      ❌ Parse failed: {e}")
    
    # Test the full extraction
    print(f"\n🔬 Full _extract_structured_json result:")
    result = parser._extract_structured_json(test_response)
    if result:
        print(f"✅ Success: {len(result)} files")
        for filename, content in result.items():
            print(f"   📄 {filename}: {len(content)} chars - {repr(content[:50])}")
    else:
        print("❌ Failed to extract")

if __name__ == "__main__":
    test_structured_json() 