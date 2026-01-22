#!/usr/bin/env python3
"""
Check GitHub Copilot CLI authentication and provide setup guidance
"""

import os
import subprocess
import sys

def check_copilot_cli():
    """Check if GitHub Copilot CLI is installed"""
    try:
        result = subprocess.run(['copilot', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ GitHub Copilot CLI is installed")
            print(f"   Version: {result.stdout.strip()}")
            return True
        else:
            print("❌ GitHub Copilot CLI not responding correctly")
            return False
    except FileNotFoundError:
        print("❌ GitHub Copilot CLI not found in PATH")
        print("\n📥 Installation instructions:")
        print("   Visit: https://github.com/github/copilot-cli")
        print("   Or run: npm install -g @githubnext/github-copilot-cli")
        return False
    except subprocess.TimeoutExpired:
        print("⚠️  GitHub Copilot CLI timed out")
        return False

def check_github_token():
    """Check for GitHub authentication token"""
    token_vars = ['GITHUB_TOKEN', 'GH_TOKEN', 'COPILOT_GITHUB_TOKEN']
    found_tokens = []
    
    for var in token_vars:
        token = os.getenv(var)
        if token:
            masked = token[:4] + '*' * (len(token) - 8) + token[-4:] if len(token) > 8 else '***'
            found_tokens.append((var, masked))
    
    if found_tokens:
        print("\n✅ GitHub tokens found:")
        for var, masked in found_tokens:
            print(f"   {var}: {masked}")
        return True
    else:
        print("\n❌ No GitHub tokens found in environment")
        print("\n🔑 Token setup instructions:")
        print("   1. Create a token: https://github.com/settings/tokens")
        print("   2. Required scopes: 'read:user', 'user:email'")
        print("   3. Set environment variable:")
        print("      PowerShell: $env:GITHUB_TOKEN = 'your_token_here'")
        print("      Bash: export GITHUB_TOKEN='your_token_here'")
        print("\n   Or authenticate with GitHub CLI:")
        print("      gh auth login")
        return False

def check_gh_cli():
    """Check GitHub CLI authentication"""
    try:
        result = subprocess.run(['gh', 'auth', 'status'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("\n✅ GitHub CLI is authenticated")
            print("   You can use: gh auth token")
            return True
        else:
            print("\n⚠️  GitHub CLI not authenticated")
            print("   Run: gh auth login")
            return False
    except FileNotFoundError:
        print("\n⚠️  GitHub CLI (gh) not installed")
        print("   Install from: https://cli.github.com/")
        return False
    except subprocess.TimeoutExpired:
        print("\n⚠️  GitHub CLI timed out")
        return False

def test_copilot_cli():
    """Test if Copilot CLI can actually generate code"""
    token = os.getenv('GITHUB_TOKEN') or os.getenv('GH_TOKEN') or os.getenv('COPILOT_GITHUB_TOKEN')
    
    if not token:
        # Try to get token from gh auth
        try:
            result = subprocess.run(['gh', 'auth', 'token'], 
                                  capture_output=True, text=True, timeout=5, check=True)
            token = result.stdout.strip()
            os.environ['GITHUB_TOKEN'] = token
            print("\n✅ Retrieved token from GitHub CLI")
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print("\n❌ Cannot retrieve authentication token")
            return False
    
    print("\n🧪 Testing Copilot CLI generation...")
    try:
        env = os.environ.copy()
        env['GITHUB_TOKEN'] = token
        
        result = subprocess.run(
            ['copilot', '-p', 'Write a Python function that returns "hello"', '--silent'],
            capture_output=True,
            text=True,
            timeout=30,
            env=env
        )
        
        if result.returncode == 0 and result.stdout.strip():
            print("✅ Copilot CLI test successful!")
            print(f"   Generated {len(result.stdout)} characters of code")
            return True
        else:
            print(f"❌ Copilot CLI test failed")
            if result.stderr:
                print(f"   Error: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️  Copilot CLI test timed out (may still work)")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Run all authentication checks"""
    print("=" * 60)
    print("GitHub Copilot CLI Authentication Check")
    print("=" * 60)
    
    # Check Copilot CLI installation
    cli_ok = check_copilot_cli()
    
    # Check authentication tokens
    token_ok = check_github_token()
    
    # Check GitHub CLI
    gh_ok = check_gh_cli()
    
    # Test actual generation if CLI is installed
    if cli_ok:
        test_ok = test_copilot_cli()
    else:
        test_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    if cli_ok and (token_ok or gh_ok) and test_ok:
        print("✅ Everything is ready! You can run Ralph loop.")
        print("\nRun: python scripts/ralph/ralph.py")
        return 0
    else:
        print("⚠️  Setup incomplete. Please fix the issues above.")
        print("\nQuick setup:")
        print("1. Install Copilot CLI: npm install -g @githubnext/github-copilot-cli")
        print("2. Authenticate: gh auth login")
        print("3. Or set token: $env:GITHUB_TOKEN = 'your_token'")
        return 1

if __name__ == "__main__":
    sys.exit(main())
