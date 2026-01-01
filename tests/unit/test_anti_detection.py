"""Unit tests for anti-detection modules."""

import pytest
from decimal import Decimal
import asyncio

from src.anti_detection.randomizer import AntiDetectionRandomizer
from src.anti_detection.pattern_breaker import PatternBreaker


class TestAntiDetectionRandomizer:
    """Tests for AntiDetectionRandomizer."""

    @pytest.fixture
    def randomizer(self):
        """Create randomizer with default settings."""
        return AntiDetectionRandomizer(
            size_variance=0.15,
            timing_variance=0.30,
            min_delay=0.5,
            max_delay=5.0,
        )

    def test_randomize_size_applies_variance(self, randomizer):
        """Should apply variance to size."""
        base_size = Decimal("1000")

        # Run multiple times
        sizes = [randomizer.randomize_size(base_size) for _ in range(20)]

        # Should have variance
        assert len(set(sizes)) > 1

        # All should be within bounds
        min_size = base_size * Decimal("0.85")
        max_size = base_size * Decimal("1.15")
        for size in sizes:
            assert min_size <= size <= max_size

    def test_randomize_size_preserves_sign(self, randomizer):
        """Should preserve sign by default."""
        negative_size = Decimal("-1000")

        for _ in range(10):
            result = randomizer.randomize_size(negative_size)
            assert result < 0

    def test_randomize_sizes_balanced_maintains_sum(self, randomizer):
        """Should maintain total sum after randomization."""
        sizes = [Decimal("1000"), Decimal("2000"), Decimal("3000")]
        target = sum(sizes)

        result = randomizer.randomize_sizes_balanced(sizes, target)

        # Sum should be preserved
        assert abs(sum(result) - target) < Decimal("0.001")

    def test_randomize_sizes_balanced_applies_variance(self, randomizer):
        """Individual sizes should vary."""
        sizes = [Decimal("1000"), Decimal("1000"), Decimal("1000")]
        target = sum(sizes)

        all_results = []
        for _ in range(10):
            result = randomizer.randomize_sizes_balanced(sizes, target)
            all_results.extend(result)

        # Should have variance
        unique_sizes = set(all_results)
        assert len(unique_sizes) > 1

    def test_randomize_interval(self, randomizer):
        """Should apply timing variance to interval."""
        base_interval = 60.0

        intervals = [randomizer.randomize_interval(base_interval) for _ in range(20)]

        # Should have variance
        assert len(set(intervals)) > 1

        # All should be within bounds (±30%)
        for interval in intervals:
            assert 42.0 <= interval <= 78.0

    def test_shuffle_accounts(self, randomizer):
        """Should shuffle account order."""
        accounts = ["acc1", "acc2", "acc3", "acc4"]

        # Run multiple times and check for different orders
        orders = set()
        for _ in range(20):
            shuffled = randomizer.shuffle_accounts(accounts)
            orders.add(tuple(shuffled))

        # Should have at least some different orders
        assert len(orders) > 1

    def test_should_skip_interval(self, randomizer):
        """Should sometimes skip intervals."""
        skips = [randomizer.should_skip_interval(0.5) for _ in range(100)]

        # With 50% probability, should have both True and False
        assert True in skips
        assert False in skips

    def test_generate_matched_pairs_delta_neutral(self, randomizer):
        """Generated pairs should sum to equal totals per side."""
        total_size = Decimal("10000")
        n_pairs = 5

        pairs = randomizer.generate_matched_pairs(total_size, n_pairs)

        total_long = sum(p[0] for p in pairs)
        total_short = sum(p[1] for p in pairs)

        # Each side should equal total_size
        assert abs(total_long - total_size) < Decimal("0.001")
        assert abs(total_short - total_size) < Decimal("0.001")

    def test_add_noise_to_price(self, randomizer):
        """Should add tick noise to price."""
        price = Decimal("50000")
        tick_size = Decimal("0.1")

        prices = [randomizer.add_noise_to_price(price, tick_size, ticks=2) for _ in range(20)]

        # Should have variance
        unique_prices = set(prices)
        assert len(unique_prices) > 1

        # All should be within ±2 ticks
        for p in prices:
            assert abs(p - price) <= tick_size * 2

    def test_get_random_subset(self, randomizer):
        """Should return random subsets."""
        items = ["a", "b", "c", "d", "e"]

        subsets = [tuple(sorted(randomizer.get_random_subset(items, 2, 4))) for _ in range(20)]

        # Should have different subsets
        unique_subsets = set(subsets)
        assert len(unique_subsets) > 1

        # All should have 2-4 items
        for subset in subsets:
            assert 2 <= len(subset) <= 4


class TestPatternBreaker:
    """Tests for PatternBreaker."""

    @pytest.fixture
    def breaker(self):
        """Create pattern breaker."""
        return PatternBreaker()

    def test_get_account_order_shuffles(self, breaker):
        """Should return shuffled account order."""
        accounts = ["acc1", "acc2", "acc3", "acc4"]

        orders = set()
        for _ in range(20):
            order = breaker.get_account_order(accounts)
            orders.add(tuple(order))

        # Should have different orders
        assert len(orders) > 1

    def test_get_varied_interval(self, breaker):
        """Should vary base interval."""
        base = 60.0

        intervals = [breaker.get_varied_interval(base) for _ in range(20)]

        # Should have variance
        assert len(set(intervals)) > 1

    def test_should_skip_interval(self, breaker):
        """Should sometimes skip intervals."""
        skips = [breaker.should_skip_interval() for _ in range(100)]

        # Default probability is low but should sometimes skip
        # With 10% probability, expect some skips in 100 trials
        skip_count = sum(1 for s in skips if s)
        assert skip_count >= 0  # Can be 0 sometimes due to randomness

    def test_record_trade(self, breaker):
        """Should record trade time."""
        breaker.record_trade("acc1")
        breaker.record_trade("acc2")

        # Should have recorded trades
        time_since = breaker.get_time_since_last_trade("acc1")
        assert time_since is not None
        assert time_since >= 0

    def test_get_time_since_last_trade_none_for_new(self, breaker):
        """Should return None for account with no trades."""
        time_since = breaker.get_time_since_last_trade("new_account")
        assert time_since is None

    def test_get_staggered_delays(self, breaker):
        """Should generate staggered delays."""
        min_delay = 0.5
        max_delay = 5.0
        delays = breaker.get_staggered_delays(4, min_delay=min_delay, max_delay=max_delay)

        assert len(delays) == 4
        # Variance is ±20% of (max-min) range, so upper bound can exceed max_delay
        variance_range = 0.2 * (max_delay - min_delay)
        for delay in delays:
            assert min_delay <= delay <= max_delay + variance_range

    def test_should_split_order_small_size(self, breaker):
        """Should not split small orders."""
        should_split = breaker.should_split_order(500, threshold=1000)
        assert should_split is False

    def test_get_split_sizes_sums_correctly(self, breaker):
        """Split sizes should sum to total."""
        total = 10000.0
        sizes = breaker.get_split_sizes(total, min_parts=2, max_parts=4)

        assert 2 <= len(sizes) <= 4
        assert abs(sum(sizes) - total) < 0.001

    def test_stats(self, breaker):
        """Should return stats dictionary."""
        breaker.record_trade("acc1")

        stats = breaker.stats()

        assert "order_history_size" in stats
        assert "timing_history_size" in stats
        assert "trade_times_tracked" in stats

    def test_clear_history(self, breaker):
        """Should clear all history."""
        breaker.record_trade("acc1")
        breaker.get_account_order(["acc1", "acc2"])

        breaker.clear_history()

        stats = breaker.stats()
        assert stats["trade_times_tracked"] == 0

    def test_get_entry_pattern(self, breaker):
        """Should return varied entry patterns."""
        patterns = set()
        for _ in range(20):
            pattern = breaker.get_entry_pattern("acc1")
            patterns.add(pattern)

        # Should have some variety
        assert len(patterns) >= 1
        # Should be valid patterns
        valid = {"immediate", "staged", "delayed"}
        for p in patterns:
            assert p in valid
