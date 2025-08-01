#!/usr/bin/env python3
"""
Test JSON Unescaping - Debug the string unescaping issue
"""

import json
import re
from agentcodeeval.utils.llm_parsing import LLMResponseParser

def test_unescaping():
    """Test the JSON unescaping logic"""
    
    # Test input - the actual JSON string from our test
    json_string = '''{
    "approach": "I'll implement a comprehensive file upload validation system that demonstrates the bug and provides a fix.",
    "files": {
        "main.go": "package main\\n\\nimport (\\n\\t\\"fmt\\"\\n\\t\\"log\\"\\n\\t\\"net/http\\"\\n\\t\\"os\\"\\n)\\n\\nfunc main() {\\n\\tfmt.Println(\\"Starting file upload server...\\")\\n\\thttp.HandleFunc(\\"/upload\\", handleUpload)\\n\\tlog.Fatal(http.ListenAndServe(\\":8080\\", nil))\\n}",
        "handler.go": "package main\\n\\nimport (\\n\\t\\"io\\"\\n\\t\\"net/http\\"\\n\\t\\"path/filepath\\"\\n)\\n\\nfunc handleUpload(w http.ResponseWriter, r *http.Request) {\\n\\tif r.Method != \\"POST\\" {\\n\\t\\thttp.Error(w, \\"Method not allowed\\", http.StatusMethodNotAllowed)\\n\\t\\treturn\\n\\t}\\n\\treturn\\n}"
    },
    "explanation": "This implementation demonstrates a simple file upload system with validation."
}'''
    
    print("🧪 Testing JSON parsing step by step...")
    
    # Step 1: Parse JSON
    try:
        data = json.loads(json_string)
        print("✅ Step 1: JSON parsing successful")
        print(f"   Files found: {list(data['files'].keys())}")
        
        # Step 2: Extract raw file content
        main_go_raw = data['files']['main.go']
        print(f"\n📄 Raw main.go content ({len(main_go_raw)} chars):")
        print(f"   First 100 chars: {repr(main_go_raw[:100])}")
        
        # Step 3: Test our unescaping
        parser = LLMResponseParser()
        unescaped = parser._unescape_code(main_go_raw)
        print(f"\n🔧 After unescaping ({len(unescaped)} chars):")
        print(f"   First 100 chars: {repr(unescaped[:100])}")
        
        if len(unescaped) != len(main_go_raw.replace('\\n', '\n').replace('\\"', '"').replace('\\t', '\t')):
            print("❌ Unescaping changed the length unexpectedly!")
        else:
            print("✅ Unescaping looks correct")
            
        # Step 4: Test manual unescaping
        manual_unescaped = main_go_raw.replace('\\n', '\n').replace('\\"', '"').replace('\\t', '\t')
        print(f"\n🔧 Manual unescaping ({len(manual_unescaped)} chars):")
        print(f"   First 100 chars: {repr(manual_unescaped[:100])}")
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")

if __name__ == "__main__":
    test_unescaping() 