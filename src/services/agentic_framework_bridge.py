"""
Agentic Framework Bridge - Connects King AI v2 orchestrator to King AI v3 agentic framework.

This bridge enables:
1. Dashboard AI interface to access agentic framework agents
2. Ralph code agent integration for autonomous coding
3. Workflow execution from YAML manifests
4. MCP gateway tool access
5. Memory service integration
"""

import asyncio
import httpx
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from src.utils.structured_logging import get_logger
from config.settings import settings

logger = get_logger("agentic_framework_bridge")


class AgenticFrameworkProvider(Enum):
    """Available agentic framework service providers."""
    LOCAL = "local"  # Local orchestrator at localhost
    AWS = "aws"      # AWS-hosted orchestrator
    

@dataclass
class AgenticAgentConfig:
    """Configuration for an agentic framework agent."""
    name: str
    agent_type: str
    capabilities: List[str]
    manifest_path: Optional[str] = None
    endpoint: Optional[str] = None
    

class AgenticFrameworkBridge:
    """
    Bridge between King AI v2 MasterAI and King AI v3 Agentic Framework.
    
    Provides unified access to:
    - Agentic framework orchestrator
    - Ralph code agent
    - Custom workflow agents
    - MCP gateway tools
    - Memory service
    """
    
    def __init__(
        self,
        orchestrator_url: str = None,
        mcp_gateway_url: str = None,
        memory_service_url: str = None,
    ):
        """
        Initialize the agentic framework bridge.
        
        Args:
            orchestrator_url: URL of the agentic framework orchestrator
            mcp_gateway_url: URL of the MCP gateway
            memory_service_url: URL of the memory service
        """
        # Default URLs from settings or environment
        self.orchestrator_url = orchestrator_url or getattr(
            settings, 'agentic_orchestrator_url', 'http://localhost:8001'
        )
        self.mcp_gateway_url = mcp_gateway_url or getattr(
            settings, 'mcp_gateway_url', 'http://localhost:3000'
        )
        self.memory_service_url = memory_service_url or getattr(
            settings, 'memory_service_url', 'http://localhost:8002'
        )
        
        # HTTP client with timeout
        self.http_client = httpx.AsyncClient(timeout=60.0)
        
        # Registered agentic agents
        self._agentic_agents: Dict[str, AgenticAgentConfig] = {}
        
        # Manifests cache
        self._manifests_cache: Dict[str, dict] = {}
        
        # Framework base path
        self._framework_path = Path(__file__).parent.parent.parent / "king-ai-v3" / "agentic-framework-main"
        
        # Register default agents
        self._register_default_agentic_agents()
        
        logger.info(
            "Agentic Framework Bridge initialized",
            orchestrator_url=self.orchestrator_url,
            mcp_gateway_url=self.mcp_gateway_url,
        )
    
    def _register_default_agentic_agents(self):
        """Register default agents from the agentic framework."""
        # Ralph code agent
        self._agentic_agents["ralph"] = AgenticAgentConfig(
            name="ralph",
            agent_type="code_agent",
            capabilities=[
                "autonomous_coding",
                "code_implementation",
                "test_writing",
                "bug_fixing",
                "code_review",
                "refactoring"
            ],
            manifest_path="ralph-code-agent",
            endpoint=f"{self.orchestrator_url}/api/workflows/ralph"
        )
        
        # Research agent from examples
        self._agentic_agents["agentic_research"] = AgenticAgentConfig(
            name="agentic_research",
            agent_type="research_agent",
            capabilities=[
                "web_research",
                "data_analysis",
                "report_generation",
                "fact_verification"
            ],
            manifest_path="01-simple-agent",
            endpoint=f"{self.orchestrator_url}/api/workflows/research"
        )
        
        # Custom workflow agent
        self._agentic_agents["workflow_executor"] = AgenticAgentConfig(
            name="workflow_executor",
            agent_type="workflow_agent",
            capabilities=[
                "yaml_workflow_execution",
                "multi_step_orchestration",
                "artifact_management"
            ],
            endpoint=f"{self.orchestrator_url}/api/workflows"
        )
        
        logger.info(f"Registered {len(self._agentic_agents)} agentic framework agents")
    
    async def close(self):
        """Close HTTP client and cleanup resources."""
        await self.http_client.aclose()
    
    def list_agentic_agents(self) -> List[Dict[str, Any]]:
        """
        List all available agentic framework agents.
        
        Returns:
            List of agent configurations
        """
        return [
            {
                "name": agent.name,
                "type": agent.agent_type,
                "capabilities": agent.capabilities,
                "source": "agentic_framework"
            }
            for agent in self._agentic_agents.values()
        ]
    
    async def execute_ralph_task(
        self,
        task_description: str,
        requirements: List[str] = None,
        files_context: List[str] = None,
        target_server: str = None,
    ) -> Dict[str, Any]:
        """
        Execute a coding task using the Ralph autonomous code agent.
        
        Args:
            task_description: Description of the coding task
            requirements: List of specific requirements
            files_context: List of files to consider
            target_server: Target server for code execution
            
        Returns:
            Execution result with status and outputs
        """
        logger.info(
            "Executing Ralph code task",
            task=task_description[:100],
            requirements_count=len(requirements or [])
        )
        
        # Build Ralph task payload
        payload = {
            "task_description": task_description,
            "requirements": requirements or [],
            "files_context": files_context or [],
            "target_server": target_server or getattr(settings, 'ralph_target_server', '100.24.50.240'),
            "config": {
                "max_iterations": 10,
                "auto_commit": False,
                "require_approval": True
            }
        }
        
        try:
            # Try to connect to the agentic framework orchestrator
            response = await self.http_client.post(
                f"{self.orchestrator_url}/api/workflows/start",
                json={
                    "manifest_id": "ralph-code-implementation",
                    "inputs": payload
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "workflow_id": result.get("workflow_id"),
                    "status": result.get("status"),
                    "message": f"Ralph task submitted successfully",
                    "task": task_description
                }
            else:
                logger.warning(f"Ralph orchestrator returned {response.status_code}")
                return await self._execute_ralph_fallback(payload)
                
        except httpx.ConnectError:
            logger.warning("Agentic framework orchestrator not available, using fallback")
            return await self._execute_ralph_fallback(payload)
        except Exception as e:
            logger.error(f"Error executing Ralph task: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to execute Ralph task"
            }
    
    async def _execute_ralph_fallback(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback execution when orchestrator is not available.
        Uses local Ralph script execution.
        """
        # Check for Ralph script
        ralph_script = self._framework_path / "scripts" / "ralph" / "ralph.sh"
        
        if not ralph_script.exists():
            return {
                "success": False,
                "error": "Ralph script not found and orchestrator not available",
                "message": "Please ensure the agentic framework orchestrator is running",
                "suggestion": "Run: cd king-ai-v3/agentic-framework-main && python orchestrator/run_service.py"
            }
        
        # Create a PRD for Ralph
        prd = {
            "title": payload["task_description"],
            "requirements": payload["requirements"],
            "files": payload["files_context"],
        }
        
        return {
            "success": True,
            "status": "queued",
            "message": "Ralph task queued for local execution",
            "prd": prd,
            "note": "Orchestrator not available - task will run when orchestrator starts"
        }
    
    async def execute_workflow(
        self,
        manifest_id: str,
        inputs: Dict[str, Any],
        approval_callback: callable = None
    ) -> Dict[str, Any]:
        """
        Execute a YAML workflow from the agentic framework.
        
        Args:
            manifest_id: ID of the workflow manifest
            inputs: Input data for the workflow
            approval_callback: Optional callback for approval gates
            
        Returns:
            Workflow execution result
        """
        logger.info(f"Executing workflow: {manifest_id}")
        
        try:
            response = await self.http_client.post(
                f"{self.orchestrator_url}/api/workflows/start",
                json={
                    "manifest_id": manifest_id,
                    "inputs": inputs
                }
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error": f"Workflow execution failed: {response.status_code}",
                    "detail": response.text
                }
                
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def invoke_mcp_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Invoke a tool through the MCP gateway.
        
        Args:
            tool_name: Name of the MCP tool
            arguments: Arguments for the tool
            
        Returns:
            Tool execution result
        """
        try:
            response = await self.http_client.post(
                f"{self.mcp_gateway_url}/api/tools/invoke",
                json={
                    "tool": tool_name,
                    "arguments": arguments
                }
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error": f"MCP tool invocation failed: {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"MCP tool error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def store_memory(
        self,
        content: str,
        memory_type: str = "short_term",
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Store content in the memory service.
        
        Args:
            content: Content to store
            memory_type: Type of memory (short_term, long_term, episodic)
            metadata: Additional metadata
            
        Returns:
            Storage result
        """
        try:
            response = await self.http_client.post(
                f"{self.memory_service_url}/api/memory/store",
                json={
                    "content": content,
                    "type": memory_type,
                    "metadata": metadata or {}
                }
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error": f"Memory storage failed: {response.status_code}"
                }
                
        except Exception as e:
            logger.warning(f"Memory service not available: {e}")
            return {
                "success": False,
                "error": str(e),
                "note": "Memory service not available"
            }
    
    async def retrieve_memory(
        self,
        query: str,
        memory_type: str = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Retrieve memories matching a query.
        
        Args:
            query: Search query
            memory_type: Optional type filter
            limit: Maximum results
            
        Returns:
            Retrieved memories
        """
        try:
            response = await self.http_client.post(
                f"{self.memory_service_url}/api/memory/retrieve",
                json={
                    "query": query,
                    "type": memory_type,
                    "limit": limit
                }
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error": f"Memory retrieval failed: {response.status_code}"
                }
                
        except Exception as e:
            logger.warning(f"Memory service not available: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def check_orchestrator_health(self) -> Dict[str, Any]:
        """Check if the agentic framework orchestrator is healthy."""
        try:
            response = await self.http_client.get(
                f"{self.orchestrator_url}/health",
                timeout=5.0
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "healthy": True,
                    "status": data.get("status", "unknown"),
                    "uptime": data.get("uptime_seconds", 0)
                }
            else:
                return {
                    "healthy": False,
                    "status": "unhealthy",
                    "error": f"Status code: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "healthy": False,
                "status": "unreachable",
                "error": str(e)
            }
    
    def get_available_manifests(self) -> List[Dict[str, Any]]:
        """
        Get list of available workflow manifests.
        
        Returns:
            List of manifest metadata
        """
        manifests_dir = self._framework_path / "orchestrator" / "manifests"
        manifests = []
        
        if manifests_dir.exists():
            for manifest_file in manifests_dir.glob("*.yaml"):
                try:
                    with open(manifest_file, "r") as f:
                        data = yaml.safe_load(f)
                        manifests.append({
                            "id": manifest_file.stem,
                            "name": data.get("name", manifest_file.stem),
                            "description": data.get("description", ""),
                            "version": data.get("version", "1.0.0"),
                            "steps": len(data.get("steps", []))
                        })
                except Exception as e:
                    logger.warning(f"Failed to load manifest {manifest_file}: {e}")
        
        return manifests


# Singleton instance
_bridge_instance: Optional[AgenticFrameworkBridge] = None


def get_agentic_bridge() -> AgenticFrameworkBridge:
    """Get or create the agentic framework bridge singleton."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = AgenticFrameworkBridge()
    return _bridge_instance
