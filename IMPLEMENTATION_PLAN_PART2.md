# King AI v2 Implementation Plan - Part 2 of 5

## Master AI & Evolution Engine Fixes

### Overview

The Master AI layer is ~75% complete with a solid architecture. However, several critical stub methods need implementation, and there are interface mismatches between the legacy and new evolution systems that cause runtime errors.

---

## Task 2.1: Fix Evolution Engine Stub Methods

**Priority:** 🔴 CRITICAL  
**File:** `src/master_ai/evolution.py`  
**Issue:** Multiple stub methods that only log but don't execute

### Instructions

#### 2.1.1: Implement `_apply_change` Method

**Location:** Lines 552-560  
**Current (Stub):**
```python
async def _apply_change(self, change: CodeChange):
    """Apply a single code change."""
    # This will be implemented in Part 5.5
    # For now, just log
    logger.info(
        "Applying change",
        file=change.file_path,
        type=change.change_type
    )
```

**Replace with:**
```python
async def _apply_change(self, change: CodeChange):
    """Apply a single code change."""
    import aiofiles
    from pathlib import Path
    
    file_path = Path(change.file_path)
    
    logger.info(
        "Applying change",
        file=str(file_path),
        type=change.change_type.value
    )
    
    try:
        if change.change_type == ChangeType.CREATE:
            # Create new file
            file_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(change.new_content or "")
                
        elif change.change_type == ChangeType.MODIFY:
            # Modify existing file
            if not file_path.exists():
                raise FileNotFoundError(f"Cannot modify non-existent file: {file_path}")
            
            if change.new_content:
                async with aiofiles.open(file_path, 'w') as f:
                    await f.write(change.new_content)
            elif change.diff:
                # Apply diff using code patcher
                from src.services.code_patcher import CodePatcher
                patcher = CodePatcher()
                async with aiofiles.open(file_path, 'r') as f:
                    original = await f.read()
                patched = patcher.apply_diff(original, change.diff)
                async with aiofiles.open(file_path, 'w') as f:
                    await f.write(patched)
                    
        elif change.change_type == ChangeType.DELETE:
            # Delete file (with safety check)
            if file_path.exists():
                # Backup before delete
                backup_path = file_path.with_suffix(file_path.suffix + '.bak')
                async with aiofiles.open(file_path, 'r') as f:
                    content = await f.read()
                async with aiofiles.open(backup_path, 'w') as f:
                    await f.write(content)
                file_path.unlink()
                
        elif change.change_type == ChangeType.RENAME:
            # Rename/move file
            if change.new_content:  # new_content contains new path for renames
                new_path = Path(change.new_content)
                new_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.rename(new_path)
                
        logger.info("Change applied successfully", file=str(file_path))
        
    except Exception as e:
        logger.error("Failed to apply change", file=str(file_path), error=str(e))
        raise
```

#### 2.1.2: Implement `_apply_config_changes` Method

**Location:** Lines 562-565  
**Current (Stub):**
```python
async def _apply_config_changes(self, config_changes: Dict[str, Any]):
    """Apply configuration changes."""
    # This will be implemented in Part 5.5
    logger.info("Applying config changes", changes=config_changes)
```

**Replace with:**
```python
async def _apply_config_changes(self, config_changes: Dict[str, Any]):
    """Apply configuration changes."""
    import yaml
    import aiofiles
    from pathlib import Path
    
    logger.info("Applying config changes", changes=list(config_changes.keys()))
    
    config_dir = Path("config")
    
    for config_key, config_value in config_changes.items():
        try:
            if config_key == "risk_profile":
                # Update risk_profiles.yaml
                risk_file = config_dir / "risk_profiles.yaml"
                async with aiofiles.open(risk_file, 'r') as f:
                    content = await f.read()
                risk_config = yaml.safe_load(content)
                
                # Merge changes
                if isinstance(config_value, dict):
                    for profile_name, profile_changes in config_value.items():
                        if profile_name in risk_config.get("profiles", {}):
                            risk_config["profiles"][profile_name].update(profile_changes)
                
                async with aiofiles.open(risk_file, 'w') as f:
                    await f.write(yaml.dump(risk_config, default_flow_style=False))
                    
            elif config_key == "settings":
                # Update settings via environment or .env file
                env_file = Path(".env")
                env_lines = []
                
                if env_file.exists():
                    async with aiofiles.open(env_file, 'r') as f:
                        env_lines = (await f.read()).splitlines()
                
                # Update or add settings
                for setting_key, setting_value in config_value.items():
                    env_key = setting_key.upper()
                    found = False
                    for i, line in enumerate(env_lines):
                        if line.startswith(f"{env_key}="):
                            env_lines[i] = f"{env_key}={setting_value}"
                            found = True
                            break
                    if not found:
                        env_lines.append(f"{env_key}={setting_value}")
                
                async with aiofiles.open(env_file, 'w') as f:
                    await f.write('\n'.join(env_lines))
                    
            elif config_key == "playbook":
                # Update playbook YAML
                playbook_name = config_value.get("name", "custom")
                playbook_file = config_dir / "playbooks" / f"{playbook_name}.yaml"
                
                async with aiofiles.open(playbook_file, 'w') as f:
                    await f.write(yaml.dump(config_value, default_flow_style=False))
                    
            logger.info(f"Applied config change: {config_key}")
            
        except Exception as e:
            logger.error(f"Failed to apply config change: {config_key}", error=str(e))
            raise
```

#### 2.1.3: Implement `_create_rollback_data` Method

**Location:** Lines 567-570  
**Current (Stub):**
```python
async def _create_rollback_data(self, proposal: EvolutionProposal) -> Dict[str, Any]:
    """Create rollback data for the proposal."""
    # This will be implemented in Part 5.75
    return {"placeholder": True}
```

**Replace with:**
```python
async def _create_rollback_data(self, proposal: EvolutionProposal) -> Dict[str, Any]:
    """Create rollback data for the proposal."""
    import aiofiles
    from pathlib import Path
    from datetime import datetime
    
    rollback_data = {
        "proposal_id": proposal.id,
        "created_at": datetime.utcnow().isoformat(),
        "original_files": {},
        "deleted_files": {},
        "created_files": [],
        "config_backup": {},
        "git_commit": None
    }
    
    # Backup original file contents
    for change in proposal.changes:
        file_path = Path(change.file_path)
        
        if change.change_type in (ChangeType.MODIFY, ChangeType.DELETE):
            if file_path.exists():
                async with aiofiles.open(file_path, 'r') as f:
                    rollback_data["original_files"][str(file_path)] = await f.read()
                    
        elif change.change_type == ChangeType.CREATE:
            rollback_data["created_files"].append(str(file_path))
    
    # Backup config if there are config changes
    if proposal.configuration_changes:
        config_files = ["config/settings.py", "config/risk_profiles.yaml", ".env"]
        for config_file in config_files:
            config_path = Path(config_file)
            if config_path.exists():
                async with aiofiles.open(config_path, 'r') as f:
                    rollback_data["config_backup"][config_file] = await f.read()
    
    # Get current git commit for reference
    try:
        from src.services.git_manager import GitManager
        git = GitManager()
        status = await git.get_status()
        rollback_data["git_commit"] = status.get("head_commit")
    except Exception:
        pass  # Git not available
    
    logger.info("Created rollback data", proposal_id=proposal.id, 
                files_backed_up=len(rollback_data["original_files"]))
    
    return rollback_data
```

#### 2.1.4: Implement `_rollback_proposal` Method

**Location:** Lines 572-575  
**Current (Stub):**
```python
async def _rollback_proposal(self, proposal: EvolutionProposal):
    """Rollback a failed proposal."""
    # This will be implemented in Part 5.75
    logger.warning("Rolling back proposal", proposal_id=proposal.id)
```

**Replace with:**
```python
async def _rollback_proposal(self, proposal: EvolutionProposal):
    """Rollback a failed proposal."""
    import aiofiles
    from pathlib import Path
    
    logger.warning("Rolling back proposal", proposal_id=proposal.id)
    
    # Get rollback data from proposal metadata
    rollback_data = proposal.metadata.get("rollback_data", {})
    
    if not rollback_data:
        logger.error("No rollback data available", proposal_id=proposal.id)
        return
    
    errors = []
    
    # Restore original files
    for file_path, original_content in rollback_data.get("original_files", {}).items():
        try:
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(original_content)
            logger.info(f"Restored file: {file_path}")
        except Exception as e:
            errors.append(f"Failed to restore {file_path}: {e}")
    
    # Delete created files
    for file_path in rollback_data.get("created_files", []):
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
            logger.info(f"Deleted created file: {file_path}")
        except Exception as e:
            errors.append(f"Failed to delete {file_path}: {e}")
    
    # Restore config backups
    for config_file, backup_content in rollback_data.get("config_backup", {}).items():
        try:
            async with aiofiles.open(config_file, 'w') as f:
                await f.write(backup_content)
            logger.info(f"Restored config: {config_file}")
        except Exception as e:
            errors.append(f"Failed to restore config {config_file}: {e}")
    
    # Update proposal status
    proposal.status = ProposalStatus.ROLLED_BACK
    await self._persist_proposal(proposal)
    
    if errors:
        logger.error("Rollback completed with errors", errors=errors)
    else:
        logger.info("Rollback completed successfully", proposal_id=proposal.id)
```

#### 2.1.5: Implement `_validate_execution` Method

**Location:** Lines 577-580  
**Current (Stub):**
```python
async def _validate_execution(self, proposal: EvolutionProposal) -> bool:
    """Validate that execution was successful."""
    # This will be implemented in Part 5.5
    return True
```

**Replace with:**
```python
async def _validate_execution(self, proposal: EvolutionProposal) -> bool:
    """Validate that execution was successful."""
    from pathlib import Path
    import ast
    import aiofiles
    
    logger.info("Validating execution", proposal_id=proposal.id)
    
    validation_errors = []
    
    # Check all modified/created files exist and are valid
    for change in proposal.changes:
        file_path = Path(change.file_path)
        
        if change.change_type == ChangeType.DELETE:
            if file_path.exists():
                validation_errors.append(f"File should be deleted but exists: {file_path}")
            continue
            
        if change.change_type in (ChangeType.CREATE, ChangeType.MODIFY):
            if not file_path.exists():
                validation_errors.append(f"File should exist but doesn't: {file_path}")
                continue
            
            # Validate Python syntax for .py files
            if file_path.suffix == '.py':
                try:
                    async with aiofiles.open(file_path, 'r') as f:
                        content = await f.read()
                    ast.parse(content)
                except SyntaxError as e:
                    validation_errors.append(f"Syntax error in {file_path}: {e}")
    
    # Run sandbox tests if available
    try:
        from src.services.sandbox import SandboxManager
        sandbox = SandboxManager()
        
        # Create a simple validation test
        test_result = await sandbox.run_tests(
            test_files=["tests/"],
            timeout=60
        )
        
        if not test_result.get("success", False):
            validation_errors.append(f"Tests failed: {test_result.get('error', 'Unknown')}")
            
    except Exception as e:
        logger.warning("Sandbox validation skipped", error=str(e))
    
    if validation_errors:
        logger.error("Execution validation failed", errors=validation_errors)
        proposal.metadata["validation_errors"] = validation_errors
        return False
    
    logger.info("Execution validation passed", proposal_id=proposal.id)
    return True
```

#### 2.1.6: Implement `get_similar_proposals` Method

**Location:** Lines 582-585  
**Current (Stub):**
```python
async def get_similar_proposals(self, proposal: EvolutionProposal) -> List[EvolutionProposal]:
    """Get historically similar proposals."""
    # This will be implemented with vector search in Part 3
    return []
```

**Replace with:**
```python
async def get_similar_proposals(self, proposal: EvolutionProposal) -> List[EvolutionProposal]:
    """Get historically similar proposals using vector similarity."""
    from src.master_ai.context_memory import VectorStore, EmbeddingClient
    
    try:
        embedding_client = EmbeddingClient()
        vector_store = VectorStore(embedding_client)
        
        # Create search text from proposal
        search_text = f"{proposal.proposal_type.value}: {proposal.description}"
        if proposal.changes:
            files_changed = [c.file_path for c in proposal.changes[:5]]
            search_text += f" Files: {', '.join(files_changed)}"
        
        # Search for similar proposals
        similar = await vector_store.search(
            query=search_text,
            namespace="evolution_proposals",
            top_k=5
        )
        
        # Load full proposals from database
        similar_proposals = []
        async with get_db() as db:
            for result in similar:
                proposal_id = result.get("metadata", {}).get("proposal_id")
                if proposal_id:
                    db_proposal = await db.get(DBEvolutionProposal, proposal_id)
                    if db_proposal:
                        similar_proposals.append(
                            EvolutionProposal.from_db(db_proposal)
                        )
        
        return similar_proposals[:5]
        
    except Exception as e:
        logger.warning("Could not find similar proposals", error=str(e))
        return []
```

---

## Task 2.2: Fix Method Signature Mismatch in brain.py

**Priority:** 🔴 CRITICAL  
**File:** `src/master_ai/brain.py`  
**Issue:** `propose_improvement` called with wrong arguments

### Instructions

**Location:** Around line 490

**Find this code:**
```python
proposal = await self.evolution.propose_improvement(context)
```

**Replace with:**
```python
# Build context for evolution proposal
evolution_context = {
    "business_state": context.get("business_data", {}),
    "recent_actions": context.get("recent_actions", []),
    "performance_metrics": context.get("kpis", {}),
    "current_goals": context.get("goals", []),
    "risk_profile": self.risk_profile
}

proposal = await self.evolution.propose_improvement(
    context=evolution_context,
    focus_area=context.get("focus_area", "general")
)
```

---

## Task 2.3: Fix Evolution Proposal Interface Compatibility

**Priority:** 🔴 CRITICAL  
**File:** `src/master_ai/brain.py`  
**Issue:** Code expects dict with `is_beneficial` but new interface returns `EvolutionProposal` object

### Instructions

**Location:** Around lines 493-510

**Find this code block:**
```python
if proposal and proposal.get("is_beneficial"):
    # Check confidence
    confidence = await self.evolution.confidence_scorer.score_proposal(proposal)
    if confidence >= 0.8:
        # Submit for approval
        ...
```

**Replace with:**
```python
if proposal:
    # Handle both legacy dict format and new EvolutionProposal object
    from src.master_ai.evolution_models import EvolutionProposal
    
    if isinstance(proposal, EvolutionProposal):
        # New interface - proposal is already an object
        is_beneficial = proposal.status != ProposalStatus.REJECTED
        confidence = proposal.confidence_score
        proposal_data = proposal.to_dict()
    else:
        # Legacy interface - proposal is a dict
        is_beneficial = proposal.get("is_beneficial", False)
        confidence = proposal.get("confidence", 0.0)
        proposal_data = proposal
    
    if is_beneficial and confidence >= 0.8:
        # Submit for approval
        approval_request = await self.approval_manager.create_request(
            action_type="evolution",
            description=proposal_data.get("description", "System evolution proposal"),
            risk_level=RiskLevel.HIGH,
            data=proposal_data,
            requester="master_ai"
        )
        
        logger.info(
            "Evolution proposal submitted for approval",
            proposal_id=proposal_data.get("id"),
            confidence=confidence,
            approval_id=approval_request.id
        )
    elif not is_beneficial:
        logger.info("Evolution proposal rejected - not beneficial")
    else:
        logger.info(
            "Evolution proposal rejected - low confidence",
            confidence=confidence,
            threshold=0.8
        )
```

---

## Task 2.4: Add Missing Import for ProposalStatus

**Priority:** 🔴 CRITICAL  
**File:** `src/master_ai/brain.py`  
**Issue:** `ProposalStatus` referenced but not imported

### Instructions

**Location:** Top of file, in imports section

**Find:**
```python
from src.master_ai.evolution import EvolutionEngine
```

**Add after it:**
```python
from src.master_ai.evolution_models import (
    EvolutionProposal,
    ProposalStatus,
    ProposalType,
    ChangeType,
    CodeChange
)
```

---

## Task 2.5: Add Missing `evolution_confidence_threshold` Setting

**Priority:** 🟡 MEDIUM  
**File:** `config/settings.py`  
**Issue:** Setting referenced but not defined

### Instructions

**Location:** In the Settings class

**Add the following setting:**
```python
# Evolution settings
evolution_confidence_threshold: float = 0.8
evolution_daily_limit: int = 1
evolution_sandbox_timeout: int = 300  # seconds
evolution_require_tests: bool = True
```

---

## Task 2.6: Fix Confidence Scorer Event Loop Conflict

**Priority:** 🟡 MEDIUM  
**File:** `src/master_ai/evolution.py`  
**Issue:** `asyncio.run()` called inside potentially existing event loop

### Instructions

**Location:** In `ConfidenceScorer.score_proposal` method (around line 145-155)

**Find code like:**
```python
try:
    import asyncio
    result = asyncio.run(self._async_score(proposal))
```

**Replace with:**
```python
import asyncio

try:
    # Check if we're already in an event loop
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = None

if loop is not None:
    # We're in an async context, create a task
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(asyncio.run, self._async_score(proposal))
        result = future.result(timeout=30)
else:
    # No event loop, safe to use asyncio.run
    result = asyncio.run(self._async_score(proposal))
```

**Better approach - make the method async:**
```python
async def score_proposal(self, proposal: EvolutionProposal) -> float:
    """Score a proposal's confidence level."""
    scores = {
        "description_quality": self._score_description(proposal.description),
        "change_safety": self._score_change_safety(proposal.changes),
        "test_coverage": await self._score_test_coverage(proposal),
        "historical_success": await self._score_historical(proposal),
    }
    
    # Weighted average
    weights = {
        "description_quality": 0.2,
        "change_safety": 0.3,
        "test_coverage": 0.25,
        "historical_success": 0.25,
    }
    
    total_score = sum(scores[k] * weights[k] for k in scores)
    proposal.confidence_score = total_score
    
    return total_score
```

---

## Task 2.7: Integrate Scheduler with Application Startup

**Priority:** 🟡 HIGH  
**File:** `src/api/main.py`  
**Issue:** Scheduler exists but isn't started with the application

### Instructions

**Location:** In the FastAPI app startup event

**Find the startup event handler:**
```python
@app.on_event("startup")
async def startup_event():
    ...
```

**Add scheduler integration:**
```python
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    from src.services.scheduler import Scheduler, TaskFrequency
    from src.master_ai.brain import MasterAI
    
    # Existing startup code...
    
    # Initialize and start scheduler
    scheduler = Scheduler()
    app.state.scheduler = scheduler
    
    # Register default autonomous tasks
    master_ai = MasterAI()
    app.state.master_ai = master_ai
    
    # KPI Review - Every 6 hours
    async def kpi_review_task():
        await master_ai.run_autonomous_cycle("kpi_review")
    
    scheduler.register_task(
        name="kpi_review",
        callback=kpi_review_task,
        frequency=TaskFrequency.EVERY_6_HOURS,
        description="Review KPIs and suggest optimizations"
    )
    
    # Evolution Check - Daily
    async def evolution_task():
        await master_ai.run_autonomous_cycle("evolution")
    
    scheduler.register_task(
        name="evolution_check",
        callback=evolution_task,
        frequency=TaskFrequency.DAILY,
        description="Check for system evolution opportunities"
    )
    
    # Business Health Check - Hourly
    async def health_check_task():
        await master_ai.run_autonomous_cycle("health_check")
    
    scheduler.register_task(
        name="business_health",
        callback=health_check_task,
        frequency=TaskFrequency.HOURLY,
        description="Check health of all business units"
    )
    
    # Start the scheduler
    await scheduler.start()
    logger.info("Scheduler started with autonomous tasks")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    # Stop scheduler
    if hasattr(app.state, 'scheduler'):
        await app.state.scheduler.stop()
        logger.info("Scheduler stopped")
```

---

## Task 2.8: Connect ML Retraining Pipeline to Evolution Engine

**Priority:** 🟡 MEDIUM  
**File:** `src/master_ai/evolution.py`  
**Issue:** MLRetrainingPipeline exists but isn't connected to MasterAI

### Instructions

**Location:** In `EvolutionEngine` class, add new method

**Add the following method:**
```python
async def propose_ml_retraining(
    self,
    model_name: str,
    training_data_path: str,
    metrics: Dict[str, float],
    reason: str
) -> Optional[EvolutionProposal]:
    """Propose ML model retraining based on performance metrics."""
    from src.master_ai.ml_retraining import MLRetrainingPipeline
    
    # Check if retraining would be beneficial
    pipeline = MLRetrainingPipeline()
    
    # Analyze current model performance
    analysis = await pipeline.analyze_performance(model_name, metrics)
    
    if not analysis.get("needs_retraining", False):
        logger.info(
            "ML retraining not needed",
            model=model_name,
            metrics=metrics
        )
        return None
    
    # Create proposal for retraining
    proposal = EvolutionProposal(
        proposal_type=ProposalType.ML_RETRAIN,
        description=f"Retrain {model_name}: {reason}",
        rationale=analysis.get("rationale", "Performance below threshold"),
        estimated_impact=analysis.get("estimated_improvement", "5-10% improvement"),
        changes=[],
        configuration_changes={
            "model_name": model_name,
            "training_data": training_data_path,
            "current_metrics": metrics,
            "training_config": {
                "method": "lora",
                "epochs": 3,
                "learning_rate": 2e-5,
                "lora_rank": 16,
                "lora_alpha": 32
            }
        }
    )
    
    # Score the proposal
    proposal.confidence_score = await self.confidence_scorer.score_proposal(proposal)
    
    if proposal.confidence_score >= self.confidence_threshold:
        proposal.status = ProposalStatus.PENDING_VALIDATION
        await self._persist_proposal(proposal)
        
        # Store in vector store for similarity search
        await self._index_proposal(proposal)
        
        return proposal
    
    return None


async def execute_ml_retraining(self, proposal: EvolutionProposal) -> Dict[str, Any]:
    """Execute approved ML retraining proposal."""
    from src.master_ai.ml_retraining import MLRetrainingPipeline
    
    if proposal.proposal_type != ProposalType.ML_RETRAIN:
        raise ValueError("Proposal is not an ML retraining proposal")
    
    if proposal.status != ProposalStatus.APPROVED:
        raise ValueError("Proposal must be approved before execution")
    
    config = proposal.configuration_changes
    pipeline = MLRetrainingPipeline()
    
    try:
        # Start retraining
        result = await pipeline.train(
            model_name=config["model_name"],
            training_data=config["training_data"],
            **config.get("training_config", {})
        )
        
        proposal.status = ProposalStatus.EXECUTED
        proposal.metadata["training_result"] = result
        await self._persist_proposal(proposal)
        
        return {
            "success": True,
            "adapter_path": result.get("adapter_path"),
            "metrics": result.get("final_metrics")
        }
        
    except Exception as e:
        proposal.status = ProposalStatus.FAILED
        proposal.metadata["error"] = str(e)
        await self._persist_proposal(proposal)
        
        return {
            "success": False,
            "error": str(e)
        }
```

---

## Task 2.9: Add Proposal Indexing for Vector Search

**Priority:** 🟡 MEDIUM  
**File:** `src/master_ai/evolution.py`  
**Issue:** Proposals not indexed for similarity search

### Instructions

**Location:** In `EvolutionEngine` class, add new method

**Add:**
```python
async def _index_proposal(self, proposal: EvolutionProposal):
    """Index proposal in vector store for similarity search."""
    from src.master_ai.context_memory import VectorStore, EmbeddingClient
    
    try:
        embedding_client = EmbeddingClient()
        vector_store = VectorStore(embedding_client)
        
        # Create text representation
        text = f"""
        Type: {proposal.proposal_type.value}
        Description: {proposal.description}
        Rationale: {proposal.rationale}
        Impact: {proposal.estimated_impact}
        Files: {', '.join(c.file_path for c in proposal.changes[:10])}
        Status: {proposal.status.value}
        Confidence: {proposal.confidence_score}
        """
        
        await vector_store.store(
            text=text,
            metadata={
                "proposal_id": proposal.id,
                "type": proposal.proposal_type.value,
                "status": proposal.status.value,
                "confidence": proposal.confidence_score,
                "created_at": proposal.created_at.isoformat() if proposal.created_at else None
            },
            namespace="evolution_proposals"
        )
        
        logger.debug("Indexed proposal", proposal_id=proposal.id)
        
    except Exception as e:
        logger.warning("Failed to index proposal", proposal_id=proposal.id, error=str(e))
```

---

## Task 2.10: Fix Daily Proposal Limit Inconsistency

**Priority:** 🟡 MEDIUM  
**Files:** 
- `src/master_ai/evolution.py`
- `src/master_ai/brain.py`

**Issue:** Different daily limits at different layers

### Instructions

1. **In `config/settings.py`**, ensure single source of truth:
```python
evolution_daily_limit: int = 1
```

2. **In `src/master_ai/evolution.py`**, read from settings:
```python
from config.settings import settings

class EvolutionEngine:
    def __init__(self, ...):
        ...
        self.daily_limit = settings.evolution_daily_limit
```

3. **In `src/master_ai/brain.py`**, also read from settings:
```python
from config.settings import settings

# In the evolution check method
if proposals_today >= settings.evolution_daily_limit:
    logger.info("Daily evolution limit reached")
    return None
```

---

## Part 2 Verification Checklist

After completing all tasks, verify:

- [ ] All stub methods in `evolution.py` are implemented
- [ ] `_apply_change` handles CREATE, MODIFY, DELETE, RENAME
- [ ] `_rollback_proposal` restores files from backup
- [ ] `brain.py` correctly handles both dict and EvolutionProposal
- [ ] ProposalStatus is properly imported
- [ ] Settings include `evolution_confidence_threshold`
- [ ] Scheduler starts with application
- [ ] Default autonomous tasks are registered
- [ ] ML retraining is connected to evolution engine
- [ ] Proposals are indexed for similarity search
- [ ] Run tests: `pytest tests/test_evolution.py -v`
- [ ] Run tests: `pytest tests/test_master_ai.py -v`

---

## Next Steps

Proceed to **IMPLEMENTATION_PLAN_PART3.md** for Sub-Agents & Router enhancements.
