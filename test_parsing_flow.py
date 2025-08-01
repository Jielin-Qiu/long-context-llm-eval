#!/usr/bin/env python3
"""
Test Parsing Flow - Debug the parse_llm_response step by step
"""

import json
import re
from agentcodeeval.utils.llm_parsing import LLMResponseParser

def test_parsing_flow():
    """Test the complete parsing flow step by step"""
    
    # Test response - wrapped in markdown like real LLM responses
    test_response = '''```json
{
    "approach": "I'll implement a comprehensive file upload validation system.",
    "files": {
        "main.go": "package main\\n\\nimport (\\n\\t\\"fmt\\"\\n\\t\\"log\\"\\n)\\n\\nfunc main() {\\n\\tfmt.Println(\\"Starting server...\\")\\n\\tlog.Fatal(\\"Done\\")\\n}",
        "handler.go": "package main\\n\\nimport \\"net/http\\"\\n\\nfunc handleUpload(w http.ResponseWriter, r *http.Request) {\\n\\tw.WriteHeader(200)\\n\\tw.Write([]byte(\\"OK\\"))\\n}"
    },
    "explanation": "This is a test implementation."
}
```'''

    print("🧪 Testing complete parsing flow...")
    print(f"📝 Response length: {len(test_response)} characters")
    
    parser = LLMResponseParser()
    
    # Step 1: Test structured JSON extraction (Strategy 1)
    print("\n🔍 Strategy 1: Structured JSON extraction")
    structured_result = parser._extract_structured_json(test_response)
    if structured_result:
        print(f"✅ Found {len(structured_result)} files")
        for filename, content in structured_result.items():
            print(f"   📄 {filename}: {len(content)} chars")
            print(f"      Preview: {repr(content[:50])}")
    else:
        print("❌ No structured JSON found")
    
    # Step 2: Test JSON-like extraction (Strategy 2)  
    print("\n🔍 Strategy 2: JSON-like extraction")
    json_like_result = parser._extract_code_from_json_like(test_response)
    if json_like_result:
        print(f"✅ Found {len(json_like_result)} files")
        for filename, content in json_like_result.items():
            print(f"   📄 {filename}: {len(content)} chars")
            print(f"      Preview: {repr(content[:50])}")
    else:
        print("❌ No JSON-like structure found")
    
    # Step 3: Test markdown code block extraction (Strategy 3)
    print("\n🔍 Strategy 3: Markdown code blocks")
    markdown_result = parser._extract_from_markdown_code_blocks(test_response, 'go')
    if markdown_result:
        print(f"✅ Found {len(markdown_result)} files")
        for filename, content in markdown_result.items():
            print(f"   📄 {filename}: {len(content)} chars")
            print(f"      Preview: {repr(content[:50])}")
    else:
        print("❌ No markdown code blocks found")
    
    # Step 4: Test the main parse function
    print("\n🔍 Main parse function result")
    from agentcodeeval.utils.llm_parsing import parse_llm_response
    final_result = parse_llm_response(test_response, expected_language='go')
    print(f"✅ Final result: {len(final_result)} files")
    for filename, content in final_result.items():
        print(f"   📄 {filename}: {len(content)} chars")
        print(f"      Preview: {repr(content[:50])}")
        if len(content) < 100:
            print(f"      FULL CONTENT: {repr(content)}")

if __name__ == "__main__":
    test_parsing_flow() 