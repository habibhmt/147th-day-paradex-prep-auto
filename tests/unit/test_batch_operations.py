"""Unit tests for Batch operations."""

import pytest
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock

from src.core.batch_operations import (
    BatchState,
    ProcessingMode,
    BatchConfig,
    BatchMetrics,
    BatchItem,
    BatchResult,
    BatchProcessor,
    BatchQueue,
    BatchAggregator,
)


class TestBatchState:
    """Tests for BatchState enum."""

    def test_state_values(self):
        """Should have expected state values."""
        assert BatchState.PENDING.value == "pending"
        assert BatchState.RUNNING.value == "running"
        assert BatchState.COMPLETED.value == "completed"
        assert BatchState.PARTIAL.value == "partial"
        assert BatchState.FAILED.value == "failed"


class TestProcessingMode:
    """Tests for ProcessingMode enum."""

    def test_mode_values(self):
        """Should have expected mode values."""
        assert ProcessingMode.SEQUENTIAL.value == "sequential"
        assert ProcessingMode.PARALLEL.value == "parallel"
        assert ProcessingMode.CHUNKED.value == "chunked"


class TestBatchConfig:
    """Tests for BatchConfig dataclass."""

    def test_default_config(self):
        """Should have correct defaults."""
        config = BatchConfig()

        assert config.chunk_size == 50
        assert config.max_parallel == 10
        assert config.continue_on_error is True
        assert config.max_retries == 3
        assert config.mode == ProcessingMode.CHUNKED

    def test_custom_config(self):
        """Should accept custom values."""
        config = BatchConfig(
            chunk_size=100,
            mode=ProcessingMode.PARALLEL,
        )

        assert config.chunk_size == 100
        assert config.mode == ProcessingMode.PARALLEL


class TestBatchMetrics:
    """Tests for BatchMetrics dataclass."""

    def test_default_metrics(self):
        """Should have correct defaults."""
        metrics = BatchMetrics()

        assert metrics.total_items == 0
        assert metrics.processed == 0
        assert metrics.successful == 0
        assert metrics.failed == 0

    def test_progress_pct_empty(self):
        """Should return 100% with no items."""
        metrics = BatchMetrics()

        assert metrics.progress_pct == 100.0

    def test_progress_pct(self):
        """Should calculate progress."""
        metrics = BatchMetrics(total_items=10, processed=5)

        assert metrics.progress_pct == 50.0

    def test_success_rate(self):
        """Should calculate success rate."""
        metrics = BatchMetrics(processed=10, successful=8)

        assert metrics.success_rate == 80.0

    def test_record_success(self):
        """Should record successful processing."""
        metrics = BatchMetrics()

        metrics.record_success(100.0)

        assert metrics.processed == 1
        assert metrics.successful == 1
        assert metrics.avg_time_per_item_ms == 100.0

    def test_record_failure(self):
        """Should record failed processing."""
        metrics = BatchMetrics()

        metrics.record_failure(50.0)

        assert metrics.processed == 1
        assert metrics.failed == 1

    def test_record_retry(self):
        """Should record retry."""
        metrics = BatchMetrics()

        metrics.record_retry()

        assert metrics.retried == 1

    def test_to_dict(self):
        """Should convert to dictionary."""
        metrics = BatchMetrics()
        metrics.record_success(100.0)

        d = metrics.to_dict()

        assert d["processed"] == 1
        assert d["successful"] == 1


class TestBatchItem:
    """Tests for BatchItem dataclass."""

    def test_create_item(self):
        """Should create batch item."""
        item = BatchItem(index=0, data="test_data")

        assert item.index == 0
        assert item.data == "test_data"
        assert item.state == BatchState.PENDING
        assert item.attempts == 0

    def test_to_dict(self):
        """Should convert to dictionary."""
        item = BatchItem(index=5, data="test")
        item.state = BatchState.COMPLETED
        item.duration_ms = 50.0

        d = item.to_dict()

        assert d["index"] == 5
        assert d["state"] == "completed"
        assert d["duration_ms"] == 50.0


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_create_result(self):
        """Should create batch result."""
        result = BatchResult(
            batch_id="batch_123",
            state=BatchState.PENDING,
        )

        assert result.batch_id == "batch_123"
        assert len(result.items) == 0

    def test_duration_ms(self):
        """Should calculate duration."""
        result = BatchResult(
            batch_id="test",
            state=BatchState.COMPLETED,
            start_time=100.0,
            end_time=100.5,
        )

        assert result.duration_ms == 500.0

    def test_successful_items(self):
        """Should filter successful items."""
        item1 = BatchItem(index=0, data="a")
        item1.state = BatchState.COMPLETED
        item2 = BatchItem(index=1, data="b")
        item2.state = BatchState.FAILED

        result = BatchResult(
            batch_id="test",
            state=BatchState.PARTIAL,
            items=[item1, item2],
        )

        assert len(result.successful_items) == 1
        assert result.successful_items[0] == item1

    def test_failed_items(self):
        """Should filter failed items."""
        item1 = BatchItem(index=0, data="a")
        item1.state = BatchState.COMPLETED
        item2 = BatchItem(index=1, data="b")
        item2.state = BatchState.FAILED

        result = BatchResult(
            batch_id="test",
            state=BatchState.PARTIAL,
            items=[item1, item2],
        )

        assert len(result.failed_items) == 1
        assert result.failed_items[0] == item2

    def test_to_dict(self):
        """Should convert to dictionary."""
        result = BatchResult(
            batch_id="batch_123",
            state=BatchState.COMPLETED,
        )

        d = result.to_dict()

        assert d["batch_id"] == "batch_123"
        assert d["state"] == "completed"


class TestBatchProcessor:
    """Tests for BatchProcessor."""

    @pytest.fixture
    def processor(self):
        """Create processor."""
        return BatchProcessor()

    @pytest.mark.asyncio
    async def test_process_without_processor(self, processor):
        """Should raise without processor set."""
        with pytest.raises(ValueError):
            await processor.process([1, 2, 3])

    @pytest.mark.asyncio
    async def test_process_simple(self, processor):
        """Should process items."""
        processor.set_processor(lambda x: x * 2)

        result = await processor.process([1, 2, 3])

        assert result.state == BatchState.COMPLETED
        assert result.results == [2, 4, 6]
        assert result.metrics.successful == 3

    @pytest.mark.asyncio
    async def test_process_async_processor(self, processor):
        """Should work with async processor."""
        async def async_double(x):
            await asyncio.sleep(0.01)
            return x * 2

        processor.set_processor(async_double)

        result = await processor.process([1, 2, 3])

        assert result.state == BatchState.COMPLETED
        assert result.results == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_process_sequential(self, processor):
        """Should process sequentially."""
        processor.config.mode = ProcessingMode.SEQUENTIAL
        processor.set_processor(lambda x: x)

        result = await processor.process([1, 2, 3])

        assert result.state == BatchState.COMPLETED

    @pytest.mark.asyncio
    async def test_process_parallel(self, processor):
        """Should process in parallel."""
        processor.config.mode = ProcessingMode.PARALLEL
        processor.config.max_parallel = 2
        processor.set_processor(lambda x: x)

        result = await processor.process([1, 2, 3, 4])

        assert result.state == BatchState.COMPLETED

    @pytest.mark.asyncio
    async def test_process_chunked(self, processor):
        """Should process in chunks."""
        processor.config.mode = ProcessingMode.CHUNKED
        processor.config.chunk_size = 2
        processor.config.rate_limit_delay = 0.01
        processor.set_processor(lambda x: x)

        result = await processor.process([1, 2, 3, 4, 5])

        assert result.state == BatchState.COMPLETED

    @pytest.mark.asyncio
    async def test_process_with_errors(self, processor):
        """Should handle errors."""
        def fail_on_2(x):
            if x == 2:
                raise ValueError("Error on 2")
            return x

        processor.set_processor(fail_on_2)
        processor.config.continue_on_error = True
        processor.config.max_retries = 1

        result = await processor.process([1, 2, 3])

        assert result.state == BatchState.PARTIAL
        assert result.metrics.failed == 1
        assert result.metrics.successful == 2

    @pytest.mark.asyncio
    async def test_process_stop_on_error(self, processor):
        """Should stop on error when configured."""
        call_count = 0

        def count_and_fail(x):
            nonlocal call_count
            call_count += 1
            if x == 2:
                raise ValueError("Error")
            return x

        processor.config.mode = ProcessingMode.SEQUENTIAL
        processor.config.continue_on_error = False
        processor.config.max_retries = 1
        processor.set_processor(count_and_fail)

        result = await processor.process([1, 2, 3])

        assert result.state == BatchState.CANCELLED

    @pytest.mark.asyncio
    async def test_process_with_retries(self, processor):
        """Should retry failed items."""
        attempt_counts = {}

        def fail_first_attempt(x):
            attempt_counts[x] = attempt_counts.get(x, 0) + 1
            if attempt_counts[x] < 2:
                raise ValueError("Retry")
            return x

        processor.config.max_retries = 3
        processor.set_processor(fail_first_attempt)

        result = await processor.process([1, 2])

        assert result.state == BatchState.COMPLETED
        assert result.metrics.retried > 0

    @pytest.mark.asyncio
    async def test_process_timeout(self, processor):
        """Should timeout slow items."""
        async def slow_processor(x):
            await asyncio.sleep(10)  # Very slow
            return x

        processor.config.timeout_per_item = 0.05
        processor.config.max_retries = 1
        processor.set_processor(slow_processor)

        result = await processor.process([1])

        assert result.metrics.failed == 1
        assert "Timeout" in result.failed_items[0].error

    @pytest.mark.asyncio
    async def test_progress_callback(self, processor):
        """Should call progress callback."""
        progress_calls = []

        def on_progress(processed, total):
            progress_calls.append((processed, total))

        processor.config.mode = ProcessingMode.SEQUENTIAL
        processor.set_processor(lambda x: x)
        processor.set_progress_callback(on_progress)

        await processor.process([1, 2, 3])

        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)

    @pytest.mark.asyncio
    async def test_cancel(self, processor):
        """Should cancel processing."""
        processed = []

        async def slow_processor(x):
            await asyncio.sleep(0.1)
            processed.append(x)
            return x

        processor.config.mode = ProcessingMode.SEQUENTIAL
        processor.set_processor(slow_processor)

        # Start processing
        task = asyncio.create_task(processor.process([1, 2, 3, 4, 5]))
        await asyncio.sleep(0.15)  # Let some process
        processor.cancel()

        result = await task

        assert result.state == BatchState.CANCELLED
        assert len(processed) < 5

    def test_is_running(self, processor):
        """Should track running state."""
        assert processor.is_running is False

    @pytest.mark.asyncio
    async def test_batch_id(self, processor):
        """Should use provided batch ID."""
        processor.set_processor(lambda x: x)

        result = await processor.process([1], batch_id="custom_id")

        assert result.batch_id == "custom_id"


class TestBatchQueue:
    """Tests for BatchQueue."""

    @pytest.fixture
    def queue(self):
        """Create queue."""
        return BatchQueue(max_size=5, max_wait_time=1.0)

    @pytest.mark.asyncio
    async def test_add(self, queue):
        """Should add item."""
        result = await queue.add("item1")

        assert result is True
        assert queue.size == 1

    @pytest.mark.asyncio
    async def test_add_full(self, queue):
        """Should reject when full."""
        for i in range(5):
            await queue.add(f"item{i}")

        result = await queue.add("overflow")

        assert result is False
        assert queue.size == 5

    @pytest.mark.asyncio
    async def test_add_many(self, queue):
        """Should add multiple items."""
        added = await queue.add_many(["a", "b", "c"])

        assert added == 3
        assert queue.size == 3

    @pytest.mark.asyncio
    async def test_add_many_partial(self, queue):
        """Should add as many as fit."""
        await queue.add_many(["a", "b", "c"])

        added = await queue.add_many(["d", "e", "f", "g"])

        assert added == 2  # Only 2 fit
        assert queue.size == 5

    @pytest.mark.asyncio
    async def test_flush(self, queue):
        """Should flush and return items."""
        await queue.add_many(["a", "b", "c"])

        items = await queue.flush()

        assert items == ["a", "b", "c"]
        assert queue.size == 0

    def test_is_full(self, queue):
        """Should detect full queue."""
        assert queue.is_full is False

    @pytest.mark.asyncio
    async def test_is_full_true(self, queue):
        """Should detect full queue."""
        for i in range(5):
            await queue.add(f"item{i}")

        assert queue.is_full is True

    @pytest.mark.asyncio
    async def test_should_flush_full(self, queue):
        """Should flush when full."""
        for i in range(5):
            await queue.add(f"item{i}")

        assert queue.should_flush is True

    @pytest.mark.asyncio
    async def test_should_flush_timeout(self, queue):
        """Should flush after timeout."""
        queue.max_wait_time = 0.05
        await queue.add("item")

        await asyncio.sleep(0.1)

        assert queue.should_flush is True


class TestBatchAggregator:
    """Tests for BatchAggregator."""

    @pytest.fixture
    def aggregator(self):
        """Create aggregator."""
        return BatchAggregator()

    def test_add_result(self, aggregator):
        """Should add result."""
        aggregator.add_result("result1")
        aggregator.add_result("result2")

        assert aggregator.success_count == 2

    def test_add_error(self, aggregator):
        """Should add error."""
        aggregator.add_error("Error 1")

        assert aggregator.error_count == 1

    def test_add_batch_result(self, aggregator):
        """Should add batch result."""
        item1 = BatchItem(index=0, data="a")
        item1.state = BatchState.COMPLETED
        item1.result = "result_a"

        item2 = BatchItem(index=1, data="b")
        item2.state = BatchState.FAILED
        item2.error = "Failed"

        batch_result = BatchResult(
            batch_id="test",
            state=BatchState.PARTIAL,
            items=[item1, item2],
            results=["result_a"],
        )

        aggregator.add_batch_result(batch_result)

        assert aggregator.success_count == 1
        assert aggregator.error_count == 1

    def test_get_results(self, aggregator):
        """Should get all results."""
        aggregator.add_result("a")
        aggregator.add_result("b")

        results = aggregator.get_results()

        assert results == ["a", "b"]

    def test_get_errors(self, aggregator):
        """Should get all errors."""
        aggregator.add_error("Error 1")
        aggregator.add_error("Error 2")

        errors = aggregator.get_errors()

        assert errors == ["Error 1", "Error 2"]

    def test_clear(self, aggregator):
        """Should clear all data."""
        aggregator.add_result("result")
        aggregator.add_error("error")

        aggregator.clear()

        assert aggregator.success_count == 0
        assert aggregator.error_count == 0

    def test_to_dict(self, aggregator):
        """Should convert to dictionary."""
        aggregator.add_result("result")
        aggregator.add_error("error")

        d = aggregator.to_dict()

        assert d["success_count"] == 1
        assert d["error_count"] == 1
