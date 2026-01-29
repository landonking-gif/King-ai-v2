"""
Ralph Memory Service Integration

Provides memory and reflection capabilities for Ralph loop:
- Diary: Logs each attempt with context and learnings
- Reflect: Analyzes patterns after story completion
- Learning: Stores knowledge for future iterations
"""

import httpx
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path


class RalphMemoryClient:
    """Client for Ralph to interact with King AI v3 Memory Service"""
    
    def __init__(self, memory_service_url: str = None):
        # Get from env or use default
        self.memory_service_url = memory_service_url or os.getenv("RALPH_MEMORY_SERVICE_URL", "http://localhost:8002")
        self.session_id = f"ralph_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.actor_id = "ralph-autonomous-loop"
        
    async def diary(
        self,
        story_id: str,
        story_title: str,
        attempt_number: int,
        success: bool,
        changes_made: int,
        prompt_used: str,
        code_generated: Optional[str],
        error: Optional[str] = None,
        quality_checks: Optional[List[tuple]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a diary entry after each story attempt.
        
        This captures:
        - What was tried
        - What happened
        - What was learned
        - Context for future iterations
        
        Returns:
            memory_id: Reference to stored diary entry
        """
        diary_entry = {
            "entry_type": "diary",
            "timestamp": datetime.now().isoformat(),
            "story": {
                "id": story_id,
                "title": story_title,
                "attempt_number": attempt_number,
                "success": success
            },
            "execution": {
                "changes_made": changes_made,
                "quality_checks_passed": all(result for _, result in (quality_checks or [])) if quality_checks else None,
                "error": error
            },
            "artifacts": {
                "prompt_length": len(prompt_used) if prompt_used else 0,
                "code_generated_length": len(code_generated) if code_generated else 0,
                "prompt_preview": prompt_used[:500] if prompt_used else None,
                "code_preview": code_generated[:500] if code_generated else None
            },
            "learning": self._extract_learning(success, error, changes_made, quality_checks),
            "context": context or {}
        }
        
        # Commit to memory service
        artifact = {
            "id": f"{story_id}_attempt_{attempt_number}_{int(datetime.now().timestamp())}",
            "artifact_type": "diary_entry",
            "content": diary_entry,
            "created_by": self.actor_id,
            "session_id": self.session_id,
            "tags": [
                "ralph_diary",
                story_id,
                f"attempt_{attempt_number}",
                "success" if success else "failure"
            ],
            "safety_class": "safe",
            "confidence": 1.0,
            "metadata": {
                "story_id": story_id,
                "attempt": attempt_number,
                "outcome": "success" if success else "failure"
            }
        }
        
        commit_request = {
            "artifact": artifact,
            "actor_id": self.actor_id,
            "actor_type": "autonomous_loop",
            "tool_ids": ["ralph_loop", "copilot_cli"],
            "generate_embedding": True,
            "store_in_cold": False
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.memory_service_url}/memory/commit",
                    json=commit_request
                )
                response.raise_for_status()
                result = response.json()
                
                print(f"📔 Diary entry committed: {result['memory_id']}")
                return result["memory_id"]
                
        except Exception as e:
            print(f"⚠️  Failed to commit diary entry: {e}")
            # Don't fail the loop if memory service is down
            return None
    
    async def reflect(
        self,
        story_id: str,
        story_title: str,
        total_attempts: int,
        final_success: bool,
        all_attempts: List[Dict[str, Any]],
        files_changed: List[str],
        commit_sha: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Reflect on completed story to extract patterns and insights.
        
        This analyzes:
        - What worked vs. what didn't
        - Common patterns across attempts
        - Improvements for future stories
        - Knowledge to retain
        
        Returns:
            reflection: Structured reflection with insights and learnings
        """
        # Analyze attempts to find patterns
        failures = [a for a in all_attempts if not a.get("success")]
        successes = [a for a in all_attempts if a.get("success")]
        
        # Extract common failure patterns
        failure_patterns = self._analyze_failure_patterns(failures)
        
        # Identify what finally worked
        success_factors = self._analyze_success_factors(successes, failures)
        
        # Generate insights
        insights = self._generate_insights(
            story_title,
            total_attempts,
            final_success,
            failure_patterns,
            success_factors,
            files_changed
        )
        
        reflection = {
            "reflection_type": "story_completion",
            "timestamp": datetime.now().isoformat(),
            "story": {
                "id": story_id,
                "title": story_title,
                "total_attempts": total_attempts,
                "final_success": final_success,
                "files_changed": files_changed,
                "commit_sha": commit_sha
            },
            "analysis": {
                "failure_patterns": failure_patterns,
                "success_factors": success_factors,
                "insights": insights
            },
            "learnings": {
                "what_worked": self._extract_what_worked(successes),
                "what_didnt_work": self._extract_what_didnt_work(failures),
                "recommendations": self._generate_recommendations(
                    failure_patterns,
                    success_factors
                )
            },
            "attempts_summary": [
                {
                    "attempt": i + 1,
                    "success": a.get("success"),
                    "changes": a.get("changes_made", 0),
                    "error": a.get("error")
                }
                for i, a in enumerate(all_attempts)
            ]
        }
        
        # Store reflection in memory service
        artifact = {
            "id": f"{story_id}_reflection_{int(datetime.now().timestamp())}",
            "artifact_type": "reflection",
            "content": reflection,
            "created_by": self.actor_id,
            "session_id": self.session_id,
            "tags": [
                "ralph_reflection",
                story_id,
                "completed" if final_success else "failed",
                f"attempts_{total_attempts}"
            ],
            "safety_class": "safe",
            "confidence": 1.0,
            "metadata": {
                "story_id": story_id,
                "total_attempts": total_attempts,
                "outcome": "success" if final_success else "failure"
            }
        }
        
        commit_request = {
            "artifact": artifact,
            "actor_id": self.actor_id,
            "actor_type": "autonomous_loop",
            "tool_ids": ["ralph_loop", "reflection_engine"],
            "generate_embedding": True,
            "store_in_cold": False
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.memory_service_url}/memory/commit",
                    json=commit_request
                )
                response.raise_for_status()
                result = response.json()
                
                print(f"\n{'='*60}")
                print("🤔 REFLECTION COMPLETE")
                print('='*60)
                print(f"Story: {story_title}")
                print(f"Attempts: {total_attempts}")
                print(f"Outcome: {'✅ Success' if final_success else '❌ Failed'}")
                print(f"\nKey Insights:")
                for insight in insights[:3]:  # Show top 3
                    print(f"  • {insight}")
                print(f"\nReflection stored: {result['memory_id']}")
                print('='*60 + '\n')
                
                return reflection
                
        except Exception as e:
            print(f"⚠️  Failed to commit reflection: {e}")
            return reflection
    
    async def query_past_learnings(
        self,
        story_title: str,
        story_description: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query memory for similar past experiences.
        
        This helps Ralph learn from previous iterations:
        - Similar stories that succeeded/failed
        - Relevant patterns and insights
        - Best practices discovered
        """
        query_text = f"{story_title} {story_description}"
        
        query_request = {
            "query_text": query_text,
            "top_k": top_k,
            "filter_artifact_type": None,  # Get both diary and reflections
            "min_similarity": 0.6
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.memory_service_url}/memory/query",
                    json=query_request
                )
                response.raise_for_status()
                result = response.json()
                
                learnings = []
                for item in result.get("items", []):
                    learnings.append({
                        "artifact_type": item["artifact_type"],
                        "content": item["content"],
                        "similarity": item["similarity"],
                        "metadata": item.get("metadata", {})
                    })
                
                if learnings:
                    print(f"\n💡 Found {len(learnings)} relevant past experience(s)")
                    print("Top matches:")
                    for i, learning in enumerate(learnings[:3], 1):
                        print(f"  {i}. {learning['artifact_type']} (similarity: {learning['similarity']:.2f})")
                
                return learnings
                
        except Exception as e:
            print(f"⚠️  Failed to query past learnings: {e}")
            return []
    
    # Helper methods
    
    def _extract_learning(
        self,
        success: bool,
        error: Optional[str],
        changes_made: int,
        quality_checks: Optional[List[tuple]]
    ) -> Dict[str, Any]:
        """Extract learnings from this attempt"""
        learning = {
            "outcome": "success" if success else "failure",
            "code_changes_applied": changes_made > 0,
            "quality_checks_passed": all(result for _, result in (quality_checks or [])) if quality_checks else None
        }
        
        if not success:
            if changes_made == 0:
                learning["issue"] = "no_code_generated"
                learning["lesson"] = "Copilot may need more specific prompt or context"
            elif error:
                learning["issue"] = "error_occurred"
                learning["error_type"] = error.split(':')[0] if ':' in error else "unknown"
                learning["lesson"] = "Need to handle this error case better"
            elif quality_checks and not all(result for _, result in quality_checks):
                learning["issue"] = "quality_checks_failed"
                learning["lesson"] = "Generated code has syntax or type errors"
        else:
            learning["success_factor"] = "code_generated_and_validated"
            
        return learning
    
    def _analyze_failure_patterns(self, failures: List[Dict]) -> List[str]:
        """Analyze common patterns in failures"""
        patterns = []
        
        if not failures:
            return patterns
        
        # Count failure types
        no_code_count = sum(1 for f in failures if f.get("changes_made", 0) == 0)
        error_count = sum(1 for f in failures if f.get("error"))
        quality_count = sum(1 for f in failures if not f.get("error") and f.get("changes_made", 0) > 0)
        
        if no_code_count > 0:
            patterns.append(f"Copilot failed to generate code in {no_code_count} attempt(s)")
        if error_count > 0:
            patterns.append(f"Errors occurred in {error_count} attempt(s)")
        if quality_count > 0:
            patterns.append(f"Quality checks failed in {quality_count} attempt(s)")
            
        return patterns
    
    def _analyze_success_factors(
        self,
        successes: List[Dict],
        failures: List[Dict]
    ) -> List[str]:
        """Identify what finally made it work"""
        factors = []
        
        if successes:
            success = successes[-1]  # Last success
            
            if len(failures) > 0:
                factors.append(f"Succeeded after {len(failures)} failed attempt(s)")
            
            if success.get("changes_made", 0) > 0:
                factors.append(f"Applied {success['changes_made']} code change(s)")
            
            if success.get("quality_checks_passed"):
                factors.append("All quality checks passed")
                
        return factors
    
    def _generate_insights(
        self,
        story_title: str,
        total_attempts: int,
        final_success: bool,
        failure_patterns: List[str],
        success_factors: List[str],
        files_changed: List[str]
    ) -> List[str]:
        """Generate high-level insights from the story completion"""
        insights = []
        
        if final_success:
            if total_attempts == 1:
                insights.append("Story completed successfully on first attempt")
            else:
                insights.append(f"Persistence paid off: succeeded after {total_attempts} attempts")
            
            if len(files_changed) > 0:
                insights.append(f"Modified {len(files_changed)} file(s): {', '.join(files_changed[:3])}")
        else:
            insights.append(f"Story failed after {total_attempts} attempts - may need human intervention")
        
        # Add pattern insights
        if failure_patterns:
            insights.append(f"Common issues: {', '.join(failure_patterns[:2])}")
        
        return insights
    
    def _extract_what_worked(self, successes: List[Dict]) -> List[str]:
        """Extract successful strategies"""
        if not successes:
            return ["No successful attempts to analyze"]
        
        strategies = []
        
        for success in successes:
            if success.get("changes_made", 0) > 0:
                strategies.append("Copilot generated and applied code successfully")
            if success.get("quality_checks_passed"):
                strategies.append("Code passed all quality checks on first try")
                
        return list(set(strategies))  # Deduplicate
    
    def _extract_what_didnt_work(self, failures: List[Dict]) -> List[str]:
        """Extract failed strategies"""
        if not failures:
            return []
        
        issues = []
        
        for failure in failures:
            if failure.get("changes_made", 0) == 0:
                issues.append("Copilot didn't generate usable code")
            if failure.get("error"):
                error_type = failure["error"].split(':')[0] if ':' in failure["error"] else "Unknown"
                issues.append(f"Error: {error_type}")
                
        return list(set(issues))  # Deduplicate
    
    def _generate_recommendations(
        self,
        failure_patterns: List[str],
        success_factors: List[str]
    ) -> List[str]:
        """Generate recommendations for future iterations"""
        recommendations = []
        
        # Based on failure patterns
        if any("no code" in p.lower() for p in failure_patterns):
            recommendations.append("Make prompts more specific with file paths and code examples")
        
        if any("quality checks failed" in p.lower() for p in failure_patterns):
            recommendations.append("Add syntax validation examples to prompt")
        
        if any("error" in p.lower() for p in failure_patterns):
            recommendations.append("Include error handling patterns in prompt")
        
        # Based on success factors
        if success_factors:
            recommendations.append("Continue using strategies that worked")
        
        if not recommendations:
            recommendations.append("Analyze this case further to improve future iterations")
            
        return recommendations
