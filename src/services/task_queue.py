"""
Background Task Queue with Redis.
Provides job queuing with retry, dead letter queue, and monitoring.
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union
from enum import Enum
import traceback

from src.utils.structured_logging import get_logger

logger = get_logger("task_queue")


class JobStatus(str, Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"
    DEAD = "dead"  # In dead letter queue


class JobPriority(int, Enum):
    """Job priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


@dataclass
class Job:
    """A queued job."""
    id: str
    name: str
    payload: Dict[str, Any]
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    max_retries: int = 3
    retry_count: int = 0
    retry_delay: float = 5.0  # Base delay in seconds
    timeout: float = 300.0  # 5 minutes default
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "payload": self.payload,
            "priority": self.priority.value,
            "status": self.status.value,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "retry_delay": self.retry_delay,
            "timeout": self.timeout,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "result": self.result,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        return cls(
            id=data["id"],
            name=data["name"],
            payload=data["payload"],
            priority=JobPriority(data.get("priority", 5)),
            status=JobStatus(data.get("status", "pending")),
            max_retries=data.get("max_retries", 3),
            retry_count=data.get("retry_count", 0),
            retry_delay=data.get("retry_delay", 5.0),
            timeout=data.get("timeout", 300.0),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            error=data.get("error"),
            result=data.get("result"),
            metadata=data.get("metadata", {}),
        )


JobHandler = Callable[[Job], Coroutine[Any, Any, Any]]


class TaskQueue:
    """
    Redis-backed task queue with priorities and retries.
    
    Features:
    - Priority-based job ordering
    - Automatic retry with exponential backoff
    - Dead letter queue for failed jobs
    - Job result storage
    - Concurrent worker support
    """
    
    # Redis key prefixes
    QUEUE_KEY = "taskqueue:jobs:{priority}"
    JOB_KEY = "taskqueue:job:{job_id}"
    RESULT_KEY = "taskqueue:result:{job_id}"
    DLQ_KEY = "taskqueue:dlq"
    PROCESSING_KEY = "taskqueue:processing"
    STATS_KEY = "taskqueue:stats"
    
    def __init__(
        self,
        redis_client = None,
        worker_count: int = 4,
        poll_interval: float = 1.0,
    ):
        self._redis = redis_client
        self._handlers: Dict[str, JobHandler] = {}
        self._worker_count = worker_count
        self._poll_interval = poll_interval
        self._workers: List[asyncio.Task] = []
        self._running = False
        self._lock = asyncio.Lock()
    
    def set_redis(self, redis_client) -> None:
        """Set the Redis client."""
        self._redis = redis_client
    
    def register_handler(self, job_name: str, handler: JobHandler) -> None:
        """
        Register a handler for a job type.
        
        Usage:
            async def process_email(job: Job) -> bool:
                # Process the job
                return True
            
            queue.register_handler("send_email", process_email)
        """
        self._handlers[job_name] = handler
        logger.info(f"Registered handler for job: {job_name}")
    
    async def enqueue(
        self,
        name: str,
        payload: Dict[str, Any],
        priority: JobPriority = JobPriority.NORMAL,
        max_retries: int = 3,
        delay: float = 0.0,
        timeout: float = 300.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Job:
        """
        Add a job to the queue.
        
        Args:
            name: Job handler name
            payload: Job data
            priority: Job priority
            max_retries: Maximum retry attempts
            delay: Delay before processing (seconds)
            timeout: Job timeout
            metadata: Additional job metadata
            
        Returns:
            The created Job
        """
        job = Job(
            id=str(uuid.uuid4()),
            name=name,
            payload=payload,
            priority=priority,
            max_retries=max_retries,
            timeout=timeout,
            metadata=metadata or {},
        )
        
        # Store job data
        await self._redis.set(
            self.JOB_KEY.format(job_id=job.id),
            json.dumps(job.to_dict()),
            ex=86400 * 7,  # Keep for 7 days
        )
        
        # Add to priority queue
        if delay > 0:
            # Delayed job - use sorted set with score as execution time
            execute_at = datetime.utcnow().timestamp() + delay
            await self._redis.zadd(
                f"taskqueue:delayed",
                {job.id: execute_at},
            )
        else:
            # Immediate job
            queue_key = self.QUEUE_KEY.format(priority=priority.value)
            await self._redis.lpush(queue_key, job.id)
        
        # Update stats
        await self._redis.hincrby(self.STATS_KEY, "enqueued", 1)
        
        logger.info(
            f"Enqueued job {job.id}: {name}",
            job_id=job.id,
            priority=priority.value,
            delay=delay,
        )
        
        return job
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        data = await self._redis.get(self.JOB_KEY.format(job_id=job_id))
        if not data:
            return None
        return Job.from_dict(json.loads(data))
    
    async def get_result(self, job_id: str) -> Optional[Any]:
        """Get the result of a completed job."""
        data = await self._redis.get(self.RESULT_KEY.format(job_id=job_id))
        if not data:
            return None
        return json.loads(data)
    
    async def _dequeue(self) -> Optional[Job]:
        """Dequeue the highest priority job."""
        # Check queues in priority order
        for priority in sorted(JobPriority, key=lambda p: p.value, reverse=True):
            queue_key = self.QUEUE_KEY.format(priority=priority.value)
            job_id = await self._redis.rpop(queue_key)
            
            if job_id:
                job = await self.get_job(job_id.decode() if isinstance(job_id, bytes) else job_id)
                if job:
                    # Mark as processing
                    await self._redis.sadd(self.PROCESSING_KEY, job.id)
                    return job
        
        return None
    
    async def _process_job(self, job: Job) -> bool:
        """Process a single job."""
        handler = self._handlers.get(job.name)
        if not handler:
            logger.error(f"No handler for job: {job.name}")
            await self._fail_job(job, f"No handler registered for: {job.name}")
            return False
        
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        await self._update_job(job)
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                handler(job),
                timeout=job.timeout,
            )
            
            # Success
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.result = result
            await self._update_job(job)
            
            # Store result
            await self._redis.set(
                self.RESULT_KEY.format(job_id=job.id),
                json.dumps(result) if result else "null",
                ex=86400,  # Keep result for 1 day
            )
            
            # Update stats
            await self._redis.hincrby(self.STATS_KEY, "completed", 1)
            
            logger.info(f"Job completed: {job.id}")
            return True
            
        except asyncio.TimeoutError:
            await self._fail_job(job, "Job timed out")
            return False
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            await self._fail_job(job, error_msg)
            return False
            
        finally:
            # Remove from processing set
            await self._redis.srem(self.PROCESSING_KEY, job.id)
    
    async def _fail_job(self, job: Job, error: str) -> None:
        """Handle a failed job."""
        job.error = error
        job.retry_count += 1
        
        if job.retry_count < job.max_retries:
            # Schedule retry with exponential backoff
            job.status = JobStatus.RETRY
            delay = job.retry_delay * (2 ** (job.retry_count - 1))
            
            await self._update_job(job)
            
            # Re-enqueue with delay
            execute_at = datetime.utcnow().timestamp() + delay
            await self._redis.zadd(
                "taskqueue:delayed",
                {job.id: execute_at},
            )
            
            await self._redis.hincrby(self.STATS_KEY, "retried", 1)
            
            logger.warning(
                f"Job {job.id} failed, scheduling retry {job.retry_count}/{job.max_retries}",
                job_id=job.id,
                delay=delay,
                error=error,
            )
        else:
            # Move to dead letter queue
            job.status = JobStatus.DEAD
            await self._update_job(job)
            await self._redis.lpush(self.DLQ_KEY, job.id)
            
            await self._redis.hincrby(self.STATS_KEY, "dead", 1)
            
            logger.error(
                f"Job {job.id} moved to dead letter queue after {job.max_retries} retries",
                job_id=job.id,
                error=error,
            )
    
    async def _update_job(self, job: Job) -> None:
        """Update job data in Redis."""
        await self._redis.set(
            self.JOB_KEY.format(job_id=job.id),
            json.dumps(job.to_dict()),
            ex=86400 * 7,
        )
    
    async def _check_delayed_jobs(self) -> None:
        """Move delayed jobs that are ready to their queues."""
        now = datetime.utcnow().timestamp()
        
        # Get jobs ready to execute
        ready_jobs = await self._redis.zrangebyscore(
            "taskqueue:delayed",
            "-inf",
            now,
            start=0,
            num=100,
        )
        
        for job_id in ready_jobs:
            job_id = job_id.decode() if isinstance(job_id, bytes) else job_id
            job = await self.get_job(job_id)
            
            if job:
                # Remove from delayed set
                await self._redis.zrem("taskqueue:delayed", job_id)
                
                # Add to appropriate priority queue
                queue_key = self.QUEUE_KEY.format(priority=job.priority.value)
                await self._redis.lpush(queue_key, job_id)
    
    async def _worker(self, worker_id: int) -> None:
        """Worker coroutine that processes jobs."""
        logger.info(f"Worker {worker_id} started")
        
        while self._running:
            try:
                # Check for delayed jobs
                await self._check_delayed_jobs()
                
                # Dequeue and process
                job = await self._dequeue()
                
                if job:
                    await self._process_job(job)
                else:
                    # No jobs, wait before polling again
                    await asyncio.sleep(self._poll_interval)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def start(self) -> None:
        """Start the task queue workers."""
        if self._running:
            return
        
        self._running = True
        
        for i in range(self._worker_count):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)
        
        logger.info(f"Task queue started with {self._worker_count} workers")
    
    async def stop(self, timeout: float = 30.0) -> None:
        """Stop the task queue workers."""
        self._running = False
        
        if self._workers:
            # Wait for workers to finish
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._workers, return_exceptions=True),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                # Cancel workers
                for worker in self._workers:
                    worker.cancel()
            
            self._workers = []
        
        logger.info("Task queue stopped")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        stats = await self._redis.hgetall(self.STATS_KEY)
        
        # Count jobs in each queue
        queue_sizes = {}
        for priority in JobPriority:
            queue_key = self.QUEUE_KEY.format(priority=priority.value)
            size = await self._redis.llen(queue_key)
            queue_sizes[priority.name.lower()] = size
        
        # Count delayed and processing
        delayed = await self._redis.zcard("taskqueue:delayed")
        processing = await self._redis.scard(self.PROCESSING_KEY)
        dlq_size = await self._redis.llen(self.DLQ_KEY)
        
        return {
            "enqueued": int(stats.get(b"enqueued", 0)),
            "completed": int(stats.get(b"completed", 0)),
            "retried": int(stats.get(b"retried", 0)),
            "dead": int(stats.get(b"dead", 0)),
            "queues": queue_sizes,
            "delayed": delayed,
            "processing": processing,
            "dead_letter_queue": dlq_size,
            "workers": self._worker_count,
            "running": self._running,
        }
    
    async def get_dead_letter_jobs(self, limit: int = 100) -> List[Job]:
        """Get jobs from the dead letter queue."""
        job_ids = await self._redis.lrange(self.DLQ_KEY, 0, limit - 1)
        jobs = []
        
        for job_id in job_ids:
            job_id = job_id.decode() if isinstance(job_id, bytes) else job_id
            job = await self.get_job(job_id)
            if job:
                jobs.append(job)
        
        return jobs
    
    async def retry_dead_letter_job(self, job_id: str) -> bool:
        """Retry a job from the dead letter queue."""
        job = await self.get_job(job_id)
        if not job or job.status != JobStatus.DEAD:
            return False
        
        # Remove from DLQ
        await self._redis.lrem(self.DLQ_KEY, 1, job_id)
        
        # Reset job state
        job.status = JobStatus.PENDING
        job.retry_count = 0
        job.error = None
        await self._update_job(job)
        
        # Re-enqueue
        queue_key = self.QUEUE_KEY.format(priority=job.priority.value)
        await self._redis.lpush(queue_key, job_id)
        
        logger.info(f"Retried dead letter job: {job_id}")
        return True


# Global task queue instance
task_queue = TaskQueue()


def get_task_queue() -> TaskQueue:
    """Get the global task queue instance."""
    return task_queue


# Decorator for registering job handlers
def job_handler(name: str, queue: Optional[TaskQueue] = None):
    """
    Decorator to register a function as a job handler.
    
    Usage:
        @job_handler("send_email")
        async def send_email(job: Job):
            # Process job
            return {"sent": True}
    """
    def decorator(func: JobHandler) -> JobHandler:
        (queue or task_queue).register_handler(name, func)
        return func
    return decorator
