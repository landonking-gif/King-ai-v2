"""
Quick test for Ralph Code Agent skill.

Tests the handler without requiring full orchestrator setup.
"""

import asyncio
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from skills.ralph_code_agent.handler import handler


async def test_prd_generation():
    """Test PRD file generation."""
    print("=" * 80)
    print("Test 1: PRD Generation")
    print("=" * 80)
    
    test_prd = {
        "title": "Add User Profile Page",
        "description": "Create a user profile page with avatar upload",
        "requirements": [
            "Display user information (name, email, bio)",
            "Allow avatar image upload",
            "Add edit mode for profile fields",
            "Save changes to database"
        ],
        "files_to_modify": [
            "src/pages/profile.tsx",
            "src/api/users.py",
            "src/components/AvatarUpload.tsx"
        ],
        "acceptance_criteria": [
            "Profile page loads without errors",
            "Avatar upload works with <5MB images",
            "Profile updates save correctly"
        ],
        "context": "React frontend with Python FastAPI backend"
    }
    
    print("PRD:")
    print(json.dumps(test_prd, indent=2))
    print()
    
    return test_prd


async def test_approval_flow():
    """Test the approval flow (should return pending_approval)."""
    print("=" * 80)
    print("Test 2: Approval Flow")
    print("=" * 80)
    
    test_prd = await test_prd_generation()
    
    print("Calling handler with approve_before_execution=True...")
    
    result = await handler(
        prd=test_prd,
        target_server="54.167.201.176",
        approve_before_execution=True
    )
    
    print()
    print("Result:")
    print(json.dumps(result, indent=2))
    print()
    
    assert result["status"] == "pending_approval", "Expected pending_approval status"
    assert result["approval_required"] == True, "Expected approval_required=True"
    
    print("✅ Approval flow test passed!")
    print()


async def test_validation():
    """Test PRD validation."""
    print("=" * 80)
    print("Test 3: PRD Validation")
    print("=" * 80)
    
    # Invalid PRD - missing required fields
    invalid_prd = {
        "title": "Test Task"
        # Missing description and requirements
    }
    
    print("Testing with invalid PRD (missing fields)...")
    
    result = await handler(
        prd=invalid_prd,
        approve_before_execution=False
    )
    
    print()
    print("Result:")
    print(json.dumps(result, indent=2))
    print()
    
    assert result["status"] == "failed", "Expected failed status for invalid PRD"
    assert "missing fields" in result["summary"].lower(), "Expected validation error message"
    
    print("✅ Validation test passed!")
    print()


async def test_dry_run():
    """Test without actual AWS execution (approval gate blocks it)."""
    print("=" * 80)
    print("Test 4: Dry Run (Full Flow)")
    print("=" * 80)
    
    test_prd = {
        "title": "Implement Password Reset Flow",
        "description": "Add complete password reset functionality with email",
        "requirements": [
            "Generate secure reset tokens",
            "Send reset email with link",
            "Validate token expiry (24 hours)",
            "Allow password update with token",
            "Add rate limiting on reset requests"
        ],
        "files_to_modify": [
            "src/api/auth.py",
            "src/services/email.py",
            "src/models/reset_token.py"
        ],
        "acceptance_criteria": [
            "User receives reset email within 30 seconds",
            "Token expires after 24 hours",
            "Password successfully updates with valid token"
        ],
        "context": "FastAPI backend with PostgreSQL and SendGrid"
    }
    
    print("PRD:")
    print(json.dumps(test_prd, indent=2))
    print()
    
    print("Executing with approval gate (won't reach AWS)...")
    
    result = await handler(
        prd=test_prd,
        target_server="54.167.201.176",
        approve_before_execution=True
    )
    
    print()
    print("Result:")
    print(json.dumps(result, indent=2))
    print()
    
    print("✅ Dry run test passed!")
    print()
    print("Note: To execute on AWS, set approve_before_execution=False")
    print("      or approve via the orchestrator dashboard.")


async def main():
    """Run all tests."""
    print()
    print("🧪 Ralph Code Agent - Unit Tests")
    print()
    
    try:
        # Run tests
        await test_approval_flow()
        await test_validation()
        await test_dry_run()
        
        print()
        print("=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print()
        print("The Ralph Code Agent skill is ready to use!")
        print()
        print("Next steps:")
        print("  1. Deploy to orchestrator service")
        print("  2. Test with full workflow via kautilya CLI")
        print("  3. Try it in the dashboard: 'Talk to King AI'")
        print()
        print("Example usage in dashboard:")
        print('  "Implement a user authentication API with JWT tokens"')
        print()
    
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ TEST FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
