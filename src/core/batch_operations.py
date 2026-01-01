"""Batch operations for efficient bulk processing."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class BatchState(Enum):
    """Batch processing states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingMode(Enum):
    """Batch processing modes."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CHUNKED = "chunked"


@dataclass
class BatchConfig:
    """Configuration for batch operations."""

    chunk_size: int = 50
    max_parallel: int = 10
    rate_limit_delay: float = 0.1
    timeout_per_item: float = 30.0
    continue_on_error: bool = True
    retry_failed: bool = True
    max_retries: int = 3
    mode: ProcessingMode = ProcessingMode.CHUNKED


@dataclass
class BatchMetrics:
    """Metrics for batch processing."""

    total_items: int = 0
    processed: int = 0
    successful: int = 0
    failed: int = 0
    retried: int = 0
    total_time_ms: float = 0.0
    avg_time_per_item_ms: float = 0.0

    @property
    def progress_pct(self) -> float:
        """Calculate progress percentage."""
        if self.total_items == 0:
            return 100.0
        return (self.processed / self.total_items) * 100

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.processed == 0:
            return 100.0
        return (self.successful / self.processed) * 100

    def record_success(self, duration_ms: float) -> None:
        """Record successful processing."""
        self.processed += 1
        self.successful += 1
        self._update_timing(duration_ms)

    def record_failure(self, duration_ms: float) -> None:
        """Record failed processing."""
        self.processed += 1
        self.failed += 1
        self._update_timing(duration_ms)

    def record_retry(self) -> None:
        """Record retry."""
        self.retried += 1

    def _update_timing(self, duration_ms: float) -> None:
        """Update timing metrics."""
        self.total_time_ms += duration_ms
        if self.processed > 0:
            self.avg_time_per_item_ms = self.total_time_ms / self.processed

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_items": self.total_items,
            "processed": self.processed,
            "successful": self.successful,
            "failed": self.failed,
            "retried": self.retried,
            "progress_pct": round(self.progress_pct, 2),
            "success_rate": round(self.success_rate, 2),
            "total_time_ms": round(self.total_time_ms, 2),
            "avg_time_per_item_ms": round(self.avg_time_per_item_ms, 2),
        }


@dataclass
class BatchItem(Generic[T]):
    """An item in a batch."""

    index: int
    data: T
    state: BatchState = BatchState.PENDING
    result: Any = None
    error: Optional[str] = None
    attempts: int = 0
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "state": self.state.value,
            "error": self.error,
            "attempts": self.attempts,
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass
class BatchResult(Generic[T, R]):
    """Result of batch processing."""

    batch_id: str
    state: BatchState
    items: List[BatchItem[T]] = field(default_factory=list)
    results: List[R] = field(default_factory=list)
    metrics: BatchMetrics = field(default_factory=BatchMetrics)
    error: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0

    def __post_init__(self):
        """Initialize lists."""
        if self.items is None:
            self.items = []
        if self.results is None:
            self.results = []

    @property
    def duration_ms(self) -> float:
        """Calculate total duration."""
        if self.end_time == 0:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    @property
    def successful_items(self) -> List[BatchItem[T]]:
        """Get successful items."""
        return [i for i in self.items if i.state == BatchState.COMPLETED]

    @property
    def failed_items(self) -> List[BatchItem[T]]:
        """Get failed items."""
        return [i for i in self.items if i.state == BatchState.FAILED]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "batch_id": self.batch_id,
            "state": self.state.value,
            "total_items": len(self.items),
            "successful_count": len(self.successful_items),
            "failed_count": len(self.failed_items),
            "duration_ms": round(self.duration_ms, 2),
            "metrics": self.metrics.to_dict(),
            "error": self.error,
        }


@dataclass
class BatchProcessor(Generic[T, R]):
    """Processor for batch operations.

    Features:
    - Multiple processing modes
    - Rate limiting
    - Error handling with retries
    - Progress tracking
    - Chunked processing
    """

    config: BatchConfig = field(default_factory=BatchConfig)
    _metrics: BatchMetrics = field(default_factory=BatchMetrics)
    _running: bool = False
    _cancelled: bool = False
    _processor: Optional[Callable[[T], R]] = None
    _progress_callback: Optional[Callable[[int, int], None]] = None

    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._metrics = BatchMetrics()
        self._running = False
        self._cancelled = False

    def set_processor(self, processor: Callable[[T], R]) -> None:
        """Set the item processor function."""
        self._processor = processor

    def set_progress_callback(
        self,
        callback: Callable[[int, int], None],
    ) -> None:
        """Set progress callback (processed, total)."""
        self._progress_callback = callback

    async def process(
        self,
        items: List[T],
        batch_id: str = None,
    ) -> BatchResult[T, R]:
        """Process a batch of items."""
        if not self._processor:
            raise ValueError("Processor function not set")

        batch_id = batch_id or f"batch_{int(time.time())}"
        result = BatchResult[T, R](
            batch_id=batch_id,
            state=BatchState.RUNNING,
            start_time=time.time(),
        )

        # Create batch items
        batch_items = [
            BatchItem[T](index=i, data=item)
            for i, item in enumerate(items)
        ]
        result.items = batch_items
        result.metrics.total_items = len(items)

        self._running = True
        self._cancelled = False

        try:
            if self.config.mode == ProcessingMode.SEQUENTIAL:
                await self._process_sequential(batch_items, result)
            elif self.config.mode == ProcessingMode.PARALLEL:
                await self._process_parallel(batch_items, result)
            else:  # CHUNKED
                await self._process_chunked(batch_items, result)

        except Exception as e:
            result.error = str(e)
            result.state = BatchState.FAILED
            logger.error(f"Batch {batch_id} failed: {e}")

        finally:
            self._running = False
            result.end_time = time.time()

            # Determine final state
            if self._cancelled:
                result.state = BatchState.CANCELLED
            elif result.metrics.failed > 0:
                if result.metrics.successful > 0:
                    result.state = BatchState.PARTIAL
                else:
                    result.state = BatchState.FAILED
            else:
                result.state = BatchState.COMPLETED

            # Collect results
            result.results = [
                item.result for item in batch_items
                if item.state == BatchState.COMPLETED
            ]

        return result

    async def _process_sequential(
        self,
        items: List[BatchItem[T]],
        result: BatchResult[T, R],
    ) -> None:
        """Process items one by one."""
        for item in items:
            if self._cancelled:
                break

            await self._process_item(item, result)

            if self._progress_callback:
                self._progress_callback(result.metrics.processed, len(items))

    async def _process_parallel(
        self,
        items: List[BatchItem[T]],
        result: BatchResult[T, R],
    ) -> None:
        """Process items in parallel."""
        semaphore = asyncio.Semaphore(self.config.max_parallel)

        async def process_with_semaphore(item: BatchItem[T]):
            async with semaphore:
                if not self._cancelled:
                    await self._process_item(item, result)

        tasks = [process_with_semaphore(item) for item in items]
        await asyncio.gather(*tasks)

    async def _process_chunked(
        self,
        items: List[BatchItem[T]],
        result: BatchResult[T, R],
    ) -> None:
        """Process items in chunks."""
        for i in range(0, len(items), self.config.chunk_size):
            if self._cancelled:
                break

            chunk = items[i:i + self.config.chunk_size]

            # Process chunk in parallel
            tasks = [self._process_item(item, result) for item in chunk]
            await asyncio.gather(*tasks)

            if self._progress_callback:
                self._progress_callback(result.metrics.processed, len(items))

            # Rate limit between chunks
            if i + self.config.chunk_size < len(items):
                await asyncio.sleep(self.config.rate_limit_delay)

    async def _process_item(
        self,
        item: BatchItem[T],
        result: BatchResult[T, R],
    ) -> None:
        """Process a single item with retries."""
        item.state = BatchState.RUNNING
        start_time = time.time()

        for attempt in range(self.config.max_retries):
            item.attempts = attempt + 1

            try:
                if asyncio.iscoroutinefunction(self._processor):
                    item.result = await asyncio.wait_for(
                        self._processor(item.data),
                        timeout=self.config.timeout_per_item,
                    )
                else:
                    item.result = self._processor(item.data)

                item.state = BatchState.COMPLETED
                item.duration_ms = (time.time() - start_time) * 1000
                result.metrics.record_success(item.duration_ms)
                return

            except asyncio.TimeoutError:
                item.error = "Timeout"
            except Exception as e:
                item.error = str(e)

            if attempt < self.config.max_retries - 1:
                result.metrics.record_retry()
                await asyncio.sleep(0.1 * (attempt + 1))

        # All retries failed
        item.state = BatchState.FAILED
        item.duration_ms = (time.time() - start_time) * 1000
        result.metrics.record_failure(item.duration_ms)

        if not self.config.continue_on_error:
            self._cancelled = True

    def cancel(self) -> None:
        """Cancel batch processing."""
        self._cancelled = True

    @property
    def is_running(self) -> bool:
        """Check if processing is running."""
        return self._running

    def get_metrics(self) -> BatchMetrics:
        """Get processing metrics."""
        return self._metrics


@dataclass
class BatchQueue(Generic[T]):
    """Queue for batch collection."""

    max_size: int = 100
    max_wait_time: float = 5.0
    _items: List[T] = field(default_factory=list)
    _last_add_time: float = 0.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self):
        """Initialize."""
        self._items = []
        self._last_add_time = time.time()
        self._lock = asyncio.Lock()

    async def add(self, item: T) -> bool:
        """Add item to queue."""
        async with self._lock:
            if len(self._items) >= self.max_size:
                return False
            self._items.append(item)
            self._last_add_time = time.time()
            return True

    async def add_many(self, items: List[T]) -> int:
        """Add multiple items."""
        async with self._lock:
            available = self.max_size - len(self._items)
            to_add = items[:available]
            self._items.extend(to_add)
            if to_add:
                self._last_add_time = time.time()
            return len(to_add)

    async def flush(self) -> List[T]:
        """Get and clear all items."""
        async with self._lock:
            items = list(self._items)
            self._items.clear()
            return items

    @property
    def size(self) -> int:
        """Get queue size."""
        return len(self._items)

    @property
    def is_full(self) -> bool:
        """Check if queue is full."""
        return len(self._items) >= self.max_size

    @property
    def should_flush(self) -> bool:
        """Check if queue should be flushed."""
        if len(self._items) >= self.max_size:
            return True
        if len(self._items) > 0:
            elapsed = time.time() - self._last_add_time
            if elapsed >= self.max_wait_time:
                return True
        return False


@dataclass
class BatchAggregator(Generic[T, R]):
    """Aggregator for batch results."""

    _results: List[R] = field(default_factory=list)
    _errors: List[str] = field(default_factory=list)
    _count: int = 0

    def __post_init__(self):
        """Initialize."""
        self._results = []
        self._errors = []
        self._count = 0

    def add_result(self, result: R) -> None:
        """Add successful result."""
        self._results.append(result)
        self._count += 1

    def add_error(self, error: str) -> None:
        """Add error."""
        self._errors.append(error)
        self._count += 1

    def add_batch_result(self, batch_result: BatchResult[T, R]) -> None:
        """Add results from batch."""
        self._results.extend(batch_result.results)
        for item in batch_result.failed_items:
            if item.error:
                self._errors.append(item.error)

    @property
    def success_count(self) -> int:
        """Count of successful results."""
        return len(self._results)

    @property
    def error_count(self) -> int:
        """Count of errors."""
        return len(self._errors)

    def get_results(self) -> List[R]:
        """Get all results."""
        return list(self._results)

    def get_errors(self) -> List[str]:
        """Get all errors."""
        return list(self._errors)

    def clear(self) -> None:
        """Clear aggregator."""
        self._results.clear()
        self._errors.clear()
        self._count = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_count": self._count,
            "success_count": self.success_count,
            "error_count": self.error_count,
        }
