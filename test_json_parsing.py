#!/usr/bin/env python3
"""
Test JSON Parsing - Debug the LLM response parsing issue

This script tests our JSON parsing with known good input to isolate the bug.
"""

from agentcodeeval.utils.llm_parsing import parse_llm_response
from rich.console import Console
from rich.syntax import Syntax

console = Console()

def test_json_parsing():
    """Test parsing with known good JSON"""
    
    # Create a test response like what OpenAI generates
    test_response = '''```json
{
    "approach": "I'll implement a comprehensive file upload validation system that demonstrates the bug and provides a fix.",
    "files": {
        "main.go": "package main\\n\\nimport (\\n\\t\\"fmt\\"\\n\\t\\"log\\"\\n\\t\\"net/http\\"\\n\\t\\"os\\"\\n)\\n\\nfunc main() {\\n\\tfmt.Println(\\"Starting file upload server...\\")\\n\\thttp.HandleFunc(\\"/upload\\", handleUpload)\\n\\tlog.Fatal(http.ListenAndServe(\\":8080\\", nil))\\n}",
        "handler.go": "package main\\n\\nimport (\\n\\t\\"io\\"\\n\\t\\"net/http\\"\\n\\t\\"path/filepath\\"\\n)\\n\\nfunc handleUpload(w http.ResponseWriter, r *http.Request) {\\n\\tif r.Method != \\"POST\\" {\\n\\t\\thttp.Error(w, \\"Method not allowed\\", http.StatusMethodNotAllowed)\\n\\t\\treturn\\n\\t}\\n\\t\\n\\t// Parse multipart form\\n\\terr := r.ParseMultipartForm(10 << 20) // 10 MB\\n\\tif err != nil {\\n\\t\\thttp.Error(w, \\"Unable to parse form\\", http.StatusBadRequest)\\n\\t\\treturn\\n\\t}\\n\\t\\n\\tfile, header, err := r.FormFile(\\"file\\")\\n\\tif err != nil {\\n\\t\\thttp.Error(w, \\"Unable to retrieve file\\", http.StatusBadRequest)\\n\\t\\treturn\\n\\t}\\n\\tdefer file.Close()\\n\\t\\n\\t// Validate file\\n\\tif !isValidFile(header.Filename) {\\n\\t\\thttp.Error(w, \\"Invalid file type\\", http.StatusBadRequest)\\n\\t\\treturn\\n\\t}\\n\\t\\n\\tw.WriteHeader(http.StatusOK)\\n\\tw.Write([]byte(\\"File uploaded successfully\\"))\\n}\\n\\nfunc isValidFile(filename string) bool {\\n\\text := filepath.Ext(filename)\\n\\tallowedExts := []string{\\".jpg\\", \\".jpeg\\", \\".png\\", \\".pdf\\", \\".txt\\"}\\n\\t\\n\\tfor _, allowed := range allowedExts {\\n\\t\\tif ext == allowed {\\n\\t\\t\\treturn true\\n\\t\\t}\\n\\t}\\n\\treturn false\\n}"
    },
    "explanation": "This implementation demonstrates a simple file upload system with validation. The main.go sets up an HTTP server, and handler.go contains the upload logic with file type validation."
}
```'''

    console.print("🧪 Testing JSON parsing with known good input...")
    console.print(f"📝 Test response length: {len(test_response)} characters")
    
    # Test our parser
    result = parse_llm_response(test_response, expected_language='go')
    
    console.print(f"\n📊 Parsing results:")
    console.print(f"   Files found: {len(result)}")
    
    for filename, content in result.items():
        console.print(f"\n📄 {filename}:")
        console.print(f"   Length: {len(content)} characters")
        console.print(f"   First 100 chars: {repr(content[:100])}")
        
        if len(content) > 100:
            console.print("   ✅ Content looks good (>100 chars)")
        else:
            console.print("   ❌ Content too short - this is the bug!")
            
            # Show the raw content
            console.print(f"   Full content: {repr(content)}")

if __name__ == "__main__":
    test_json_parsing() 