"""
Advanced LLM Response Parsing for AgentCodeEval

This module provides robust parsing capabilities for LLM responses,
handling JSON extraction, code block parsing, and intelligent fallbacks.
"""

import json
import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMResponseParser:
    """Advanced parser for LLM responses with multiple fallback strategies"""
    
    def __init__(self):
        # Patterns for different response formats
        self.json_patterns = [
            # Pattern 1: ```json blocks - enhanced for very long content
            r'```json\s*(\{.*?\})\s*```',
            
            # Pattern 2: Generic ``` blocks  
            r'```\s*(\{.*?\})\s*```',
            
            # Pattern 3: Specific files pattern - non-greedy
            r'(\{[^{}]*?"files"[^{}]*?\{.*?\}[^{}]*?\})',
            
            # Pattern 4: Any JSON-like structure - more conservative
            r'(\{(?:[^{}]|\{[^{}]*\})*\})',
            
            # Pattern 5: NEW - Handle very long JSON with line breaks (non-greedy)
            r'```json\s*(\{(?:[^`]|`(?!``))*\})\s*```',
            
            # Pattern 6: NEW - Multiline JSON without markdown
            r'(\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})'
        ]
        
        self.code_block_patterns = {
            'go': r'```go\s*(.*?)\s*```',
            'python': r'```python\s*(.*?)\s*```', 
            'javascript': r'```(?:js|javascript)\s*(.*?)\s*```',
            'typescript': r'```(?:ts|typescript)\s*(.*?)\s*```',
            'rust': r'```rust\s*(.*?)\s*```',
            'java': r'```java\s*(.*?)\s*```',
            'cpp': r'```(?:cpp|c\+\+)\s*(.*?)\s*```',
            'generic': r'```\w*\s*(.*?)\s*```'
        }
        
        # Language-specific indicators
        self.language_indicators = {
            'go': ['package main', 'func main', 'import "', 'type ', 'func '],
            'python': ['def ', 'import ', 'from ', 'class ', '__init__'],
            'javascript': ['function ', 'const ', 'let ', 'var ', '=>'],
            'typescript': ['interface ', 'type ', 'function ', 'const ', '=>'],
            'rust': ['fn ', 'use ', 'mod ', 'struct ', 'impl '],
            'java': ['public class', 'private ', 'public ', 'static '],
            'cpp': ['#include', 'int main', 'class ', 'namespace ']
        }

    def _is_viable_result(self, files: Dict[str, str]) -> bool:
        """Check if the extracted files are viable (not suspiciously short)"""
        if not files:
            return False
        
        # Check if any file has reasonable content length
        min_viable_length = 50  # Minimum characters for viable code
        
        for filename, content in files.items():
            # For code files, expect at least some meaningful content
            if len(content) >= min_viable_length:
                return True
        
        # If all files are very short, this might be a parsing artifact
        return False

    def parse(self, response: str, expected_language: str = 'python') -> Dict[str, str]:
        """Parse LLM response with multiple fallback strategies"""
        
        # Strategy 1: Structured JSON extraction
        structured_result = self._extract_structured_json(response)
        if structured_result and self._is_viable_result(structured_result):
            logger.info(f"✅ Successfully extracted {len(structured_result)} files from structured JSON")
            return structured_result
            
        # Strategy 2: JSON-like structures with viability check
        json_like_result = self._extract_code_from_json_like(response)
        if json_like_result and self._is_viable_result(json_like_result):
            logger.info(f"✅ Successfully extracted {len(json_like_result)} files from JSON-like structure")
            return json_like_result
            
        # Strategy 3: Markdown code blocks
        code_blocks_result = self._extract_code_blocks(response, expected_language)
        if code_blocks_result and self._is_viable_result(code_blocks_result):
            logger.info(f"✅ Successfully extracted {len(code_blocks_result)} code blocks")
            return code_blocks_result
            
        # Strategy 4: Intelligent text parsing
        text_parsing_result = self._intelligent_text_parsing(response, expected_language)
        if text_parsing_result and self._is_viable_result(text_parsing_result):
            logger.info("✅ Successfully parsed code from text analysis")
            return text_parsing_result
        
        # Strategy 5: Manual extraction for extremely long JSON
        manual_result = self._extract_files_manually(response)
        if manual_result and self._is_viable_result(manual_result):
            logger.info("✅ Successfully extracted files manually")
            return manual_result
            
        # Strategy 6: If we got results but they were short, try the best short result
        if json_like_result:
            logger.warning("⚠️  Using short JSON-like result as last resort")
            return json_like_result
        elif structured_result:
            logger.warning("⚠️  Using short structured result as last resort")
            return structured_result
            
        # Strategy 7: Final fallback
        logger.warning("⚠️  Using fallback solution generation")
        return self._create_fallback_solution(response, expected_language)

    def _extract_structured_json(self, response: str) -> Optional[Dict[str, str]]:
        """Extract properly structured JSON responses"""
        
        for pattern in self.json_patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                try:
                    # Clean up the JSON string
                    cleaned_json = self._clean_json_string(match)
                    data = json.loads(cleaned_json)
                    
                    # Look for files in various possible keys
                    files = self._extract_files_from_data(data)
                    if files:
                        return files
                        
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON parsing failed: {e}")
                    continue
                    
        return None

    def _extract_code_from_json_like(self, response: str) -> Optional[Dict[str, str]]:
        """Extract code from JSON-like structures that might have parsing issues"""
        
        # Look for patterns like "filename.go": "code content"
        file_pattern = r'"([^"]+\.(?:go|py|js|ts|rs|java|cpp|h))"\s*:\s*"([^"]*(?:\\.[^"]*)*)"'
        matches = re.findall(file_pattern, response, re.DOTALL)
        
        if matches:
            files = {}
            for filename, code in matches:
                # Unescape the code content
                unescaped_code = self._unescape_code(code)
                files[filename] = unescaped_code
            return files
            
        # Look for key-value patterns without quotes
        kv_pattern = r'(\w+\.(?:go|py|js|ts|rs|java|cpp|h))\s*[:=]\s*`([^`]+)`'
        kv_matches = re.findall(kv_pattern, response, re.DOTALL)
        
        if kv_matches:
            return {filename: code for filename, code in kv_matches}
            
        return None

    def _extract_code_blocks(self, response: str, expected_language: str) -> Optional[Dict[str, str]]:
        """Extract code from markdown code blocks"""
        
        files = {}
        
        # Try language-specific pattern first
        if expected_language in self.code_block_patterns:
            pattern = self.code_block_patterns[expected_language]
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            
            for i, code in enumerate(matches):
                filename = f"solution_{i+1}.{self._get_file_extension(expected_language)}"
                files[filename] = code.strip()
        
        # Try generic code blocks
        if not files:
            pattern = self.code_block_patterns['generic']
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            
            for i, code in enumerate(matches):
                # Try to detect language from content
                detected_lang = self._detect_language(code)
                extension = self._get_file_extension(detected_lang or expected_language)
                filename = f"solution_{i+1}.{extension}"
                files[filename] = code.strip()
        
        return files if files else None

    def _intelligent_text_parsing(self, response: str, expected_language: str) -> Optional[Dict[str, str]]:
        """Parse code using intelligent text analysis"""
        
        # Look for file headers or separators
        file_separators = [
            r'(?:^|\n)(?:File|Filename):\s*([^\n]+)',
            r'(?:^|\n)#+\s*([^\n]+\.(?:go|py|js|ts|rs|java|cpp|h))',
            r'(?:^|\n)//\s*([^\n]+\.(?:go|py|js|ts|rs|java|cpp|h))',
            r'(?:^|\n)#\s*([^\n]+\.(?:go|py|js|ts|rs|java|cpp|h))'
        ]
        
        files = {}
        
        for separator_pattern in file_separators:
            matches = list(re.finditer(separator_pattern, response, re.IGNORECASE | re.MULTILINE))
            
            if matches:
                for i, match in enumerate(matches):
                    filename = match.group(1).strip()
                    start = match.end()
                    
                    # Find the end of this file's content
                    if i + 1 < len(matches):
                        end = matches[i + 1].start()
                    else:
                        end = len(response)
                    
                    code_content = response[start:end].strip()
                    
                    # Clean up the code content
                    code_content = self._clean_code_content(code_content)
                    
                    if code_content and len(code_content) > 10:  # Minimum viable code
                        files[filename] = code_content
        
        return files if files else None

    def _create_fallback_solution(self, response: str, expected_language: str) -> Dict[str, str]:
        """Create a fallback solution when parsing fails"""
        
        # Try to detect if there's any code-like content
        code_indicators = self.language_indicators.get(expected_language, [])
        
        if any(indicator in response for indicator in code_indicators):
            # Extract the most relevant portion
            lines = response.split('\n')
            code_lines = []
            
            for line in lines:
                if any(indicator in line for indicator in code_indicators):
                    code_lines.append(line)
                elif code_lines and (line.strip().startswith((' ', '\t')) or 
                                   any(char in line for char in '{}();')):
                    code_lines.append(line)
                elif code_lines:
                    break  # Stop when we seem to have left the code section
            
            if code_lines:
                code_content = '\n'.join(code_lines)
                filename = f"solution.{self._get_file_extension(expected_language)}"
                return {filename: code_content}
        
        # Ultimate fallback: create template with response as comments
        extension = self._get_file_extension(expected_language)
        template = self._create_language_template(expected_language, response)
        
        return {f"solution.{extension}": template}

    def _clean_json_string(self, json_str: str) -> str:
        """Clean and fix common JSON formatting issues"""
        
        # Remove leading/trailing whitespace
        json_str = json_str.strip()
        
        # Fix trailing commas before closing brackets/braces
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Enhanced cleaning for very long strings
        
        # Fix any double-escaped sequences that might confuse JSON parser
        json_str = json_str.replace('\\\\n', '\\n')  # Fix double-escaped newlines
        json_str = json_str.replace('\\\\t', '\\t')  # Fix double-escaped tabs
        json_str = json_str.replace('\\\\"', '\\"')  # Fix double-escaped quotes
        json_str = json_str.replace('\\\\r', '\\r')  # Fix double-escaped carriage returns
        
        # NEW: Handle problematic characters in very long strings
        # Fix unescaped quotes that aren't part of JSON structure
        # This is a more aggressive fix for malformed long strings
        try:
            # First, try to parse as-is
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError as e:
            # If parsing fails, try to fix common issues
            
            # Try to identify if it's an unterminated string issue
            if "Unterminated string" in str(e):
                # More aggressive cleaning for very long content
                # Split by lines and try to fix string escaping issues
                lines = json_str.split('\n')
                fixed_lines = []
                
                for line in lines:
                    # Ensure all quotes in JSON string values are properly escaped
                    # This is a simplified fix - more sophisticated logic could be added
                    fixed_line = line
                    
                    # If we're inside a JSON string value (after :"), fix unescaped quotes
                    if '": "' in line and not line.strip().endswith('",') and not line.strip().endswith('"'):
                        # This might be a broken string - try to close it
                        if not fixed_line.rstrip().endswith('"'):
                            fixed_line = fixed_line.rstrip() + '"'
                    
                    fixed_lines.append(fixed_line)
                
                json_str = '\n'.join(fixed_lines)
        
        return json_str

    def _extract_files_from_data(self, data: Any) -> Optional[Dict[str, str]]:
        """Extract files from parsed JSON data"""
        
        if isinstance(data, dict):
            # Direct files key
            if 'files' in data and isinstance(data['files'], dict):
                return data['files']
            
            # Solution key containing files
            if 'solution' in data and isinstance(data['solution'], dict):
                if 'files' in data['solution']:
                    return data['solution']['files']
                return data['solution']
            
            # Code key
            if 'code' in data and isinstance(data['code'], dict):
                return data['code']
            
            # Look for any dict that looks like filename -> code mapping
            for key, value in data.items():
                if isinstance(value, dict):
                    # Check if it looks like a filename -> code mapping
                    if any(k.endswith(('.go', '.py', '.js', '.ts', '.rs', '.java', '.cpp', '.h')) 
                           for k in value.keys()):
                        return value
        
        return None

    def _unescape_code(self, code: str) -> str:
        """Unescape code content from JSON strings"""
        
        code = code.replace('\\"', '"')
        code = code.replace('\\n', '\n')
        code = code.replace('\\t', '\t')
        code = code.replace('\\r', '\r')
        code = code.replace('\\\\', '\\')
        
        return code

    def _detect_language(self, code: str) -> Optional[str]:
        """Detect programming language from code content"""
        
        code_lower = code.lower()
        
        for language, indicators in self.language_indicators.items():
            matches = sum(1 for indicator in indicators if indicator in code_lower)
            if matches >= 2:  # Require at least 2 indicators
                return language
        
        return None

    def _get_file_extension(self, language: str) -> str:
        """Get file extension for a programming language"""
        
        extensions = {
            'go': 'go',
            'python': 'py',
            'javascript': 'js', 
            'typescript': 'ts',
            'rust': 'rs',
            'java': 'java',
            'cpp': 'cpp'
        }
        
        return extensions.get(language, 'go')  # Default to Go

    def _clean_code_content(self, content: str) -> str:
        """Clean extracted code content"""
        
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip empty lines at the beginning
            if not cleaned_lines and not line.strip():
                continue
            cleaned_lines.append(line)
        
        # Remove trailing empty lines
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
        
        return '\n'.join(cleaned_lines)

    def _create_language_template(self, language: str, original_response: str) -> str:
        """Create a basic template for the given language"""
        
        templates = {
            'go': f"""package main

import "fmt"

// Generated solution based on LLM response
func main() {{
    fmt.Println("Solution implementation")
    // TODO: Implement functionality based on requirements
}}

/*
Original LLM Response:
{original_response[:1000]}...
*/
""",
            'python': f"""#!/usr/bin/env python3
\"\"\"
Generated solution based on LLM response
\"\"\"

def main():
    print("Solution implementation")
    # TODO: Implement functionality based on requirements

if __name__ == "__main__":
    main()

# Original LLM Response:
# {original_response[:500]}...
""",
            'javascript': f"""// Generated solution based on LLM response

function main() {{
    console.log("Solution implementation");
    // TODO: Implement functionality based on requirements
}}

main();

/*
Original LLM Response:
{original_response[:500]}...
*/
"""
        }
        
        return templates.get(language, templates['go'])

    def _extract_files_manually(self, response: str) -> Optional[Dict[str, str]]:
        """Manually extract files from JSON-like responses when regex fails
        
        This handles cases where the JSON contains extremely long strings
        that break regex patterns.
        """
        
        files = {}
        
        # Look for the files section
        files_start = response.find('"files"')
        if files_start == -1:
            return None
        
        # Find the opening brace of the files object
        brace_pos = response.find('{', files_start)
        if brace_pos == -1:
            return None
        
        # Parse character by character to extract files
        i = brace_pos + 1
        while i < len(response):
            # Skip whitespace
            while i < len(response) and response[i].isspace():
                i += 1
            
            if i >= len(response):
                break
                
            # Check for end of files object
            if response[i] == '}':
                break
            
            # Look for filename in quotes
            if response[i] == '"':
                # Extract filename
                i += 1  # Skip opening quote
                filename_start = i
                while i < len(response) and response[i] != '"':
                    if response[i] == '\\':
                        i += 2  # Skip escaped character
                    else:
                        i += 1
                
                if i >= len(response):
                    break
                    
                filename = response[filename_start:i]
                i += 1  # Skip closing quote
                
                # Skip whitespace and colon
                while i < len(response) and (response[i].isspace() or response[i] == ':'):
                    i += 1
                
                # Look for content in quotes
                if i < len(response) and response[i] == '"':
                    i += 1  # Skip opening quote
                    content_start = i
                    content_chars = []
                    
                    # Extract content, handling escape sequences
                    while i < len(response):
                        if response[i] == '\\' and i + 1 < len(response):
                            # Handle escape sequence
                            next_char = response[i + 1]
                            if next_char == 'n':
                                content_chars.append('\n')
                            elif next_char == 't':
                                content_chars.append('\t')
                            elif next_char == 'r':
                                content_chars.append('\r')
                            elif next_char == '"':
                                content_chars.append('"')
                            elif next_char == '\\':
                                content_chars.append('\\')
                            else:
                                content_chars.append(next_char)
                            i += 2
                        elif response[i] == '"':
                            # End of string
                            break
                        else:
                            content_chars.append(response[i])
                            i += 1
                    
                    if filename.endswith(('.go', '.py', '.js', '.ts', '.rs', '.java', '.cpp', '.h')):
                        content = ''.join(content_chars)
                        files[filename] = content
                    
                    i += 1  # Skip closing quote
            
            # Skip to next comma or end
            while i < len(response) and response[i] not in ',}':
                i += 1
            
            if i < len(response) and response[i] == ',':
                i += 1
        
        return files if files else None


# Convenience function for easy import
def parse_llm_response(response: str, expected_language: str = 'go') -> Dict[str, str]:
    """Parse LLM response using the advanced parser"""
    parser = LLMResponseParser()
    return parser.parse(response, expected_language) 