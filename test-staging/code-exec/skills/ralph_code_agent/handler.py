"""
Ralph Code Agent Skill - Handler

Integrates the Ralph autonomous coding agent with King AI v3.
Delegates code implementation tasks to Ralph loop running on AWS server.

Workflow:
1. Receive detailed PRD (Product Requirements Document)
2. Generate Ralph-compatible task specification
3. Invoke Ralph loop on target server (AWS)
4. Monitor execution and collect results
5. Return structured output with provenance
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def create_ralph_prd_file(prd: Dict[str, Any], output_path: Path) -> None:
    """
    Create a PRD file compatible with Ralph's expected format.
    
    Args:
        prd: Product Requirements Document dict
        output_path: Path to write the PRD file
    """
    ralph_prd = {
        "title": prd.get("title", "Code Implementation Task"),
        "description": prd.get("description", ""),
        "requirements": prd.get("requirements", []),
        "files": prd.get("files_to_modify", []),
        "acceptance_criteria": prd.get("acceptance_criteria", []),
        "context": prd.get("context", ""),
        "metadata": {
            "source": "king-ai-v3-orchestrator",
            "timestamp": datetime.utcnow().isoformat(),
            "workflow_id": os.getenv("WORKFLOW_ID", "unknown")
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ralph_prd, f, indent=2)
    
    logger.info(f"Created Ralph PRD file at: {output_path}")


async def execute_ralph_on_aws(
    prd_path: Path,
    target_server: str,
    ssh_key_path: Optional[str] = None,
    memory_service_url: str = "http://localhost:8002"
) -> Dict[str, Any]:
    """
    Execute Ralph loop on AWS server via SSH with memory service integration.
    
    Args:
        prd_path: Path to PRD file
        target_server: AWS server IP or hostname
        ssh_key_path: Path to SSH key (defaults to king-ai-studio.pem)
        memory_service_url: URL of memory service for Ralph to use
    
    Returns:
        Dict with execution results
    """
    if ssh_key_path is None:
        # Default to King AI Studio key
        ssh_key_path = os.path.expanduser(
            "~/Downloads/landon/king-ai-v2/king-ai-v3/agentic-framework-main/king-ai-studio.pem"
        )
    
    if not os.path.exists(ssh_key_path):
        raise FileNotFoundError(f"SSH key not found: {ssh_key_path}")
    
    # Upload PRD to server
    remote_prd_path = f"/tmp/ralph_prd_{int(time.time())}.json"
    
    logger.info(f"Uploading PRD to {target_server}:{remote_prd_path}")
    
    upload_cmd = [
        "scp",
        "-i", ssh_key_path,
        "-o", "StrictHostKeyChecking=no",
        str(prd_path),
        f"ubuntu@{target_server}:{remote_prd_path}"
    ]
    
    try:
        result = subprocess.run(
            upload_cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            logger.error(f"Failed to upload PRD: {result.stderr}")
            return {
                "status": "failed",
                "summary": f"Failed to upload PRD to server: {result.stderr}",
                "files_changed": [],
                "ralph_output": result.stderr,
                "execution_time": 0,
                "approval_required": False
            }
    except subprocess.TimeoutExpired:
        logger.error("SCP upload timed out")
        return {
            "status": "timeout",
            "summary": "Upload to server timed out",
            "files_changed": [],
            "ralph_output": "",
            "execution_time": 0,
            "approval_required": False
        }
    
    # Execute Ralph on the server with memory service integration
    ralph_script_path = "/home/ubuntu/king-ai-v2/scripts/ralph/ralph.py"
    
    # Pass memory service URL to Ralph
    ssh_cmd = [
        "ssh",
        "-i", ssh_key_path,
        "-o", "StrictHostKeyChecking=no",
        f"ubuntu@{target_server}",
        f"cd /home/ubuntu/king-ai-v2 && python3 {ralph_script_path} --prd {remote_prd_path} --memory-service {memory_service_url} --json-output"
    ]
    
    logger.info(f"Executing Ralph on {target_server} with memory service {memory_service_url}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for Ralph execution
        )
        
        execution_time = time.time() - start_time
        
        # Parse Ralph output (assuming JSON output)
        try:
            ralph_result = json.loads(result.stdout)
            
            return {
                "status": "success" if result.returncode == 0 else "failed",
                "summary": ralph_result.get("summary", "Ralph execution completed"),
                "files_changed": ralph_result.get("files_changed", []),
                "ralph_output": result.stdout,
                "execution_time": execution_time,
                "approval_required": False
            }
        except json.JSONDecodeError:
            # Fallback if output is not JSON
            return {
                "status": "success" if result.returncode == 0 else "failed",
                "summary": f"Ralph {'completed' if result.returncode == 0 else 'failed'} (non-JSON output)",
                "files_changed": [],
                "ralph_output": result.stdout + "\n" + result.stderr,
                "execution_time": execution_time,
                "approval_required": False
            }
    
    except subprocess.TimeoutExpired:
        logger.error("Ralph execution timed out")
        return {
            "status": "timeout",
            "summary": "Ralph execution exceeded 10 minute timeout",
            "files_changed": [],
            "ralph_output": "Execution timed out",
            "execution_time": 600,
            "approval_required": False
        }
    except Exception as e:
        logger.exception("Error executing Ralph")
        return {
            "status": "failed",
            "summary": f"Error executing Ralph: {str(e)}",
            "files_changed": [],
            "ralph_output": str(e),
            "execution_time": time.time() - start_time,
            "approval_required": False
        }
    finally:
        # Clean up remote PRD file
        cleanup_cmd = [
            "ssh",
            "-i", ssh_key_path,
            "-o", "StrictHostKeyChecking=no",
            f"ubuntu@{target_server}",
            f"rm -f {remote_prd_path}"
        ]
        subprocess.run(cleanup_cmd, capture_output=True, timeout=10)


async def handler(
    prd: Dict[str, Any],
    target_server: str = "54.167.201.176",
    approve_before_execution: bool = True
) -> Dict[str, Any]:
    """
    Main handler for Ralph Code Agent skill.
    
    Args:
        prd: Product Requirements Document with task details
        target_server: Target server IP/hostname (default: AWS)
        approve_before_execution: If True, requires human approval
    
    Returns:
        Dict with execution results and provenance
    """
    logger.info(f"Ralph Code Agent invoked for task: {prd.get('title', 'Unnamed')}")
    
    # Validate PRD structure
    required_fields = ["title", "description", "requirements"]
    missing = [f for f in required_fields if f not in prd]
    
    if missing:
        return {
            "status": "failed",
            "summary": f"Invalid PRD: missing fields {missing}",
            "files_changed": [],
            "ralph_output": "",
            "execution_time": 0,
            "approval_required": False
        }
    
    # Check if approval is required
    if approve_before_execution:
        logger.warning("Human approval required - returning pending status")
        return {
            "status": "pending_approval",
            "summary": f"Task '{prd['title']}' requires human approval before Ralph execution",
            "files_changed": [],
            "ralph_output": json.dumps(prd, indent=2),
            "execution_time": 0,
            "approval_required": True
        }
    
    # Create temporary PRD file
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.json',
        delete=False,
        prefix='ralph_prd_'
    ) as tmp_file:
        prd_path = Path(tmp_file.name)
    
    try:
        # Generate Ralph-compatible PRD
        create_ralph_prd_file(prd, prd_path)
        
        # Execute Ralph on AWS
        result = await execute_ralph_on_aws(prd_path, target_server)
        
        logger.info(f"Ralph execution completed: {result['status']}")
        
        return result
    
    finally:
        # Clean up local PRD file
        if prd_path.exists():
            prd_path.unlink()


# Synchronous wrapper for compatibility
def execute(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronous wrapper for handler (required by skill executor).
    """
    return asyncio.run(handler(**params))
