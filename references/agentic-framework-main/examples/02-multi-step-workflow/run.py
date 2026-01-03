"""
Multi-Step Workflow Example

Demonstrates how multiple agents collaborate in a sequential pipeline.
This example simulates the workflow execution to show the pattern without
requiring the full orchestrator infrastructure.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any
import yaml


class WorkflowSimulator:
    """Simulates workflow execution for demonstration purposes."""

    def __init__(self, manifest_path: Path):
        """Initialize with manifest file."""
        with open(manifest_path, 'r') as f:
            self.manifest = yaml.safe_load(f)

    async def execute(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow with user input."""
        print(f"\n{'='*60}")
        print(f"Executing Workflow: {self.manifest['name']}")
        print(f"{'='*60}\n")

        artifacts = {}
        step_count = len(self.manifest['steps'])

        for idx, step in enumerate(self.manifest['steps'], 1):
            print(f"[Step {idx}/{step_count}] {step['id'].upper()}: {step['description']}")
            print(f"  Role: {step['role']}")
            print(f"  Capabilities: {', '.join(step['capabilities'])}")
            print(f"  Timeout: {step['timeout']}s")

            # Simulate step execution
            await asyncio.sleep(0.5)  # Simulate processing time

            # Create mock artifact based on step
            if step['id'] == 'research':
                artifact = self._simulate_research(user_input)
            elif step['id'] == 'verify':
                artifact = self._simulate_verification(artifacts['research'])
            elif step['id'] == 'synthesize':
                artifact = self._simulate_synthesis(
                    artifacts['research'],
                    artifacts['verify']
                )
            else:
                artifact = {"status": "completed"}

            artifacts[step['id']] = artifact

            print(f"  ✓ Completed - Generated artifact: {step['outputs'][0]}")
            print(f"    Preview: {self._preview_artifact(artifact)}\n")

        return artifacts

    def _simulate_research(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate research step output."""
        query = user_input.get('query', 'unknown query')
        return {
            "id": "research-001",
            "query": query,
            "summary": f"Research findings on '{query}' show significant developments...",
            "sources": [
                {"url": "https://example.com/source1", "title": "AI Research Paper"},
                {"url": "https://example.com/source2", "title": "Industry Report"}
            ],
            "confidence": 0.85,
            "key_findings": [
                "Multi-agent systems improve task decomposition",
                "Collaboration protocols are critical for success",
                "Enterprise adoption is accelerating"
            ]
        }

    def _simulate_verification(self, research_artifact: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate verification step output."""
        return {
            "id": "verify-001",
            "claims_verified": 3,
            "claims_disputed": 0,
            "overall_confidence": 0.92,
            "verifications": [
                {
                    "claim": "Multi-agent systems improve task decomposition",
                    "verdict": "SUPPORTED",
                    "evidence_sources": 2
                },
                {
                    "claim": "Enterprise adoption is accelerating",
                    "verdict": "SUPPORTED",
                    "evidence_sources": 3
                }
            ]
        }

    def _simulate_synthesis(self, research: Dict[str, Any], verification: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate synthesis step output."""
        return {
            "id": "synthesis-001",
            "title": f"Research Report: {research.get('query', 'Unknown')}",
            "executive_summary": "This report synthesizes research findings and verification results...",
            "sections": [
                {
                    "title": "Key Findings",
                    "content": research.get('summary', ''),
                    "confidence": research.get('confidence', 0)
                },
                {
                    "title": "Fact Verification",
                    "content": f"All major claims verified with {verification.get('overall_confidence', 0):.1%} confidence",
                    "verified_claims": verification.get('claims_verified', 0)
                }
            ],
            "conclusion": "The research indicates strong evidence for the investigated claims.",
            "quality_score": 0.88
        }

    def _preview_artifact(self, artifact: Dict[str, Any]) -> str:
        """Create a preview of the artifact."""
        if 'summary' in artifact:
            return artifact['summary'][:60] + "..."
        elif 'overall_confidence' in artifact:
            return f"Confidence: {artifact['overall_confidence']:.1%}"
        elif 'executive_summary' in artifact:
            return artifact['executive_summary'][:60] + "..."
        else:
            return json.dumps(artifact)[:60] + "..."


async def main():
    """Run the multi-step workflow example."""
    print("=" * 60)
    print("Multi-Step Workflow Example")
    print("=" * 60)

    # Load workflow manifest
    manifest_path = Path(__file__).parent / "workflow.yaml"

    if not manifest_path.exists():
        print(f"\n❌ Error: Workflow manifest not found at {manifest_path}")
        return

    # Create simulator
    simulator = WorkflowSimulator(manifest_path)

    # User input
    user_input = {
        "query": "Multi-agent AI systems in enterprise applications"
    }

    print(f"\n📝 User Query: {user_input['query']}")
    print("\n🔄 Workflow Execution:\n")

    # Execute workflow
    try:
        results = await simulator.execute(user_input)

        # Display final results
        print("="  * 60)
        print("Workflow Complete!")
        print("=" * 60)

        final_report = results.get('synthesize', {})
        print(f"\n📊 Final Report: {final_report.get('title', 'N/A')}")
        print(f"\n{final_report.get('executive_summary', 'No summary available')}")
        print(f"\nQuality Score: {final_report.get('quality_score', 0):.1%}")

        print("\n💡 What you learned:")
        print("   ✓ How to define multi-step workflows in YAML")
        print("   ✓ How agents pass artifacts between steps")
        print("   ✓ How to configure timeouts and policies")
        print("   ✓ How the pipeline pattern works")

        print("\n📁 Files:")
        print("   - workflow.yaml: Workflow definition")
        print("   - run.py: Simulation script")

        print("\nNext steps:")
        print("  - See examples/03-custom-skill/ for creating skills")
        print("  - See examples/04-mcp-integration/ for external tool access")
        print("  - Read docs/ for orchestrator API details")

    except Exception as e:
        print(f"\n❌ Error executing workflow: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
