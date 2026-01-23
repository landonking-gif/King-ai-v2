#!/usr/bin/env python3
"""
Test script to verify Copilot CLI integration is working for Ralph
"""

import subprocess
import sys
from pathlib import Path

def test_copilot_cli():
    """Test if copilot CLI is accessible"""
    print("Testing Copilot CLI integration...")
    print("=" * 60)
    
    try:
        # Test 1: Check if copilot command exists
        print("\n1. Checking if 'copilot' command is available...")
        result = subprocess.run(
            ['copilot', '--help'],
            capture_output=True,
            text=True,
            timeout=5,
            shell=True
        )
        
        if result.returncode == 0:
            print("   ✅ Copilot CLI is available!")
        else:
            print("   ❌ Copilot CLI command failed")
            print(f"   Error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("   ❌ Copilot CLI not found in PATH")
        print("\n   Make sure you're running this from VS Code terminal")
        print("   where the 'copilot' command is available.")
        return False
    except subprocess.TimeoutExpired:
        print("   ⚠️  Copilot CLI timed out (might still work)")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 2: Test simple prompt
    print("\n2. Testing simple prompt with Copilot CLI...")
    test_prompt = "Write a Python function that adds two numbers and returns the result."
    
    try:
        process = subprocess.Popen(
            ['copilot'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True
        )
        
        stdout, stderr = process.communicate(input=test_prompt, timeout=30)
        
        if process.returncode == 0 and stdout:
            print("   ✅ Successfully received response from Copilot CLI!")
            print(f"\n   Response preview (first 200 chars):")
            print("   " + "-" * 56)
            preview = stdout[:200].replace('\n', '\n   ')
            print(f"   {preview}")
            if len(stdout) > 200:
                print("   ... (truncated)")
            print("   " + "-" * 56)
        else:
            print("   ⚠️  Copilot CLI responded but may have issues")
            if stderr:
                print(f"   Error output: {stderr[:200]}")
                
    except subprocess.TimeoutExpired:
        print("   ❌ Copilot CLI timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"   ❌ Error during prompt test: {e}")
        return False
    
    # Test 3: Check PRD file exists
    print("\n3. Checking for prd.json file...")
    prd_path = Path("prd.json")
    if prd_path.exists():
        print(f"   ✅ Found prd.json at {prd_path.absolute()}")
        
        # Load and validate PRD
        import json
        try:
            with open(prd_path, 'r') as f:
                prd_data = json.load(f)
            
            if 'branchName' in prd_data:
                print(f"   ✅ PRD branch: {prd_data['branchName']}")
            
            if 'userStories' in prd_data:
                stories = prd_data['userStories']
                incomplete = [s for s in stories if not s.get('passes', False)]
                print(f"   ✅ Total stories: {len(stories)}")
                print(f"   ✅ Incomplete stories: {len(incomplete)}")
            else:
                print("   ⚠️  No userStories found in PRD")
                
        except json.JSONDecodeError as e:
            print(f"   ❌ Invalid JSON in prd.json: {e}")
            return False
    else:
        print("   ❌ prd.json not found in project root")
        print("   Ralph requires a prd.json file to work")
        return False
    
    # Test 4: Check prompt template
    print("\n4. Checking prompt template...")
    prompt_path = Path("scripts/ralph/prompt.md")
    if prompt_path.exists():
        print(f"   ✅ Found prompt template at {prompt_path}")
        
        # Check for required placeholders
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_placeholders = [
            '{{STORY_ID}}',
            '{{STORY_TITLE}}',
            '{{STORY_DESCRIPTION}}',
            '{{STORY_ACCEPTANCE}}',
            '{{PRD_CONTEXT}}',
            '{{PROGRESS_CONTEXT}}'
        ]
        
        missing = [p for p in required_placeholders if p not in content]
        if missing:
            print(f"   ⚠️  Missing placeholders: {', '.join(missing)}")
        else:
            print("   ✅ All required placeholders present")
    else:
        print(f"   ❌ Prompt template not found at {prompt_path}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Ralph is ready to use.")
    print("\nRun Ralph with:")
    print("  python scripts/ralph/ralph.py")
    print("\nOr for limited iterations:")
    print("  python scripts/ralph/ralph.py 10")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_copilot_cli()
    sys.exit(0 if success else 1)
