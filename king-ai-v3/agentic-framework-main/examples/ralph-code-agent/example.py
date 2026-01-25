"""
Example: Using Ralph Code Agent with King AI v3

This example demonstrates how to delegate code implementation
to the Ralph autonomous coding agent running on AWS.
"""

import asyncio
import json
from pathlib import Path

# Assuming King AI v3 SDK is available
from king_ai import KingAIClient


async def main():
    """Main example function."""
    
    # Initialize King AI client
    client = KingAIClient(
        base_url="https://king-ai-studio.me",
        api_key="your-api-key"  # Or use environment variable
    )
    
    print("=" * 80)
    print("Ralph Code Agent Example - User Authentication Feature")
    print("=" * 80)
    print()
    
    # Define the coding task
    task = {
        "task_description": "Implement JWT-based user authentication API",
        "requirements": [
            "Create login and logout endpoints",
            "Implement JWT token generation and validation",
            "Add password hashing with bcrypt",
            "Include rate limiting on authentication attempts",
            "Add proper error handling and logging",
            "Write unit tests for authentication logic"
        ],
        "files_context": [
            "src/api/auth.py",
            "src/models/user.py",
            "src/utils/security.py",
            "tests/test_auth.py"
        ],
        "target_server": "3.236.144.91"  # AWS server
    }
    
    print("📝 Task Details:")
    print(f"  Title: {task['task_description']}")
    print(f"  Requirements: {len(task['requirements'])} items")
    print(f"  Target Server: {task['target_server']}")
    print()
    
    # Submit workflow to King AI
    print("🚀 Submitting to King AI v3 Orchestrator...")
    
    workflow = await client.execute_workflow(
        manifest_id="ralph-code-implementation",
        inputs=task
    )
    
    workflow_id = workflow["id"]
    print(f"  ✅ Workflow created: {workflow_id}")
    print()
    
    # Monitor workflow execution
    print("⏳ Monitoring execution...")
    
    async for update in client.stream_workflow_updates(workflow_id):
        step_id = update.get("step_id")
        status = update.get("status")
        message = update.get("message", "")
        
        if step_id == "generate_prd":
            print(f"  📋 Generating PRD... {status}")
            if status == "completed":
                prd = update.get("output", {}).get("prd", {})
                print(f"     - PRD Title: {prd.get('title', 'N/A')}")
                print(f"     - Requirements: {len(prd.get('requirements', []))}")
        
        elif step_id == "ralph_execution":
            if status == "pending_approval":
                print(f"  ⏸️  APPROVAL REQUIRED")
                print(f"     {message}")
                print()
                print("  Review the PRD and approve execution:")
                print(f"  Dashboard: https://king-ai-studio.me/workflows/{workflow_id}")
                print()
                
                # Wait for approval (in real scenario, human would approve via dashboard)
                approval_choice = input("  Approve execution? (y/n): ")
                
                if approval_choice.lower() == 'y':
                    await client.approve_workflow_step(workflow_id, step_id)
                    print("  ✅ Approved - Ralph will now execute")
                else:
                    await client.reject_workflow_step(workflow_id, step_id)
                    print("  ❌ Rejected - Workflow cancelled")
                    return
            
            elif status == "executing":
                print(f"  🔧 Ralph is coding on AWS... {message}")
            
            elif status == "completed":
                result = update.get("output", {}).get("code_result", {})
                print(f"  ✅ Ralph execution completed!")
                print(f"     - Status: {result.get('status')}")
                print(f"     - Files Changed: {len(result.get('files_changed', []))}")
                print(f"     - Execution Time: {result.get('execution_time', 0):.1f}s")
        
        elif step_id == "verify_results":
            if status == "completed":
                print(f"  ✅ Verification completed")
    
    # Get final results
    print()
    print("📊 Retrieving final results...")
    
    final_result = await client.get_workflow_result(workflow_id)
    
    implementation_result = final_result.get("implementation_result", {})
    
    print()
    print("=" * 80)
    print("IMPLEMENTATION RESULTS")
    print("=" * 80)
    print()
    print(f"Status: {implementation_result.get('status', 'unknown')}")
    print()
    print("Summary:")
    print(implementation_result.get('summary', 'No summary available'))
    print()
    print("Files Changed:")
    for file_path in implementation_result.get('files_changed', []):
        print(f"  - {file_path}")
    print()
    
    # View provenance
    print("📜 Provenance Information:")
    provenance = await client.get_artifact_provenance(
        workflow_id=workflow_id,
        artifact_id=final_result.get("artifact_id")
    )
    
    print(f"  - Actor: {provenance.get('actor_id')}")
    print(f"  - Timestamp: {provenance.get('timestamp')}")
    print(f"  - Tools Used: {', '.join(provenance.get('tool_ids', []))}")
    print()
    
    print("✅ Workflow completed successfully!")
    print()
    print("Next steps:")
    print("  1. Review code changes on AWS server")
    print("  2. Run tests to verify implementation")
    print("  3. Create pull request for review")
    print("  4. Deploy to staging/production")


if __name__ == "__main__":
    asyncio.run(main())
