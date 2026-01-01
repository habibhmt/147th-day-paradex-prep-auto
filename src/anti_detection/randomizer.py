"""Randomization utilities for anti-detection."""

import asyncio
import logging
import random
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AntiDetectionRandomizer:
    """Randomizes trading patterns to avoid detection.

    Applies variance to:
    - Order sizes
    - Timing between operations
    - Execution order of accounts

    Maintains delta neutrality while adding noise.
    """

    size_variance: float = 0.15  # +/- 15%
    timing_variance: float = 0.30  # +/- 30%
    min_delay: float = 0.5  # Minimum delay seconds
    max_delay: float = 5.0  # Maximum delay seconds

    def randomize_size(
        self,
        base_size: Decimal,
        preserve_sign: bool = True,
    ) -> Decimal:
        """Add random variance to order size.

        Args:
            base_size: Base size to randomize
            preserve_sign: Keep original sign

        Returns:
            Randomized size
        """
        sign = 1 if base_size >= 0 else -1
        abs_size = abs(base_size)

        variance = Decimal(
            str(1 + random.uniform(-self.size_variance, self.size_variance))
        )
        result = abs_size * variance

        if preserve_sign:
            return sign * result
        return result

    def randomize_sizes_balanced(
        self,
        sizes: List[Decimal],
        target_sum: Decimal,
    ) -> List[Decimal]:
        """Randomize sizes while maintaining sum.

        Useful for delta-neutral pairs where total must match.

        Args:
            sizes: List of sizes to randomize
            target_sum: Required sum of all sizes

        Returns:
            Randomized sizes summing to target
        """
        if not sizes:
            return []

        # Apply variance
        randomized = [self.randomize_size(s) for s in sizes]

        # Normalize to maintain sum
        current_sum = sum(randomized)
        if current_sum == 0:
            return sizes

        factor = target_sum / current_sum
        return [s * factor for s in randomized]

    async def random_delay(self) -> float:
        """Add random delay between operations.

        Returns:
            Actual delay applied
        """
        delay = random.uniform(self.min_delay, self.max_delay)
        await asyncio.sleep(delay)
        return delay

    async def variable_delay(self, base_delay: float) -> float:
        """Add variance to a base delay.

        Args:
            base_delay: Base delay to vary

        Returns:
            Actual delay applied
        """
        variance = random.uniform(1 - self.timing_variance, 1 + self.timing_variance)
        delay = base_delay * variance
        delay = max(self.min_delay, min(self.max_delay * 2, delay))
        await asyncio.sleep(delay)
        return delay

    def randomize_interval(self, base_interval: float) -> float:
        """Randomize a time interval.

        Args:
            base_interval: Base interval

        Returns:
            Randomized interval
        """
        variance = random.uniform(1 - self.timing_variance, 1 + self.timing_variance)
        return base_interval * variance

    def shuffle_accounts(self, account_ids: List[str]) -> List[str]:
        """Randomly shuffle account execution order.

        Args:
            account_ids: List of account IDs

        Returns:
            Shuffled list
        """
        shuffled = account_ids.copy()
        random.shuffle(shuffled)
        return shuffled

    def should_skip_interval(self, probability: float = 0.1) -> bool:
        """Randomly decide to skip an interval.

        Useful for breaking regular patterns.

        Args:
            probability: Probability to skip (0-1)

        Returns:
            True if should skip
        """
        return random.random() < probability

    def generate_matched_pairs(
        self,
        total_size: Decimal,
        n_pairs: int,
    ) -> List[Tuple[Decimal, Decimal]]:
        """Generate random long/short pairs that sum to equal totals.

        Args:
            total_size: Total size per side
            n_pairs: Number of pairs to generate

        Returns:
            List of (long_size, short_size) tuples
        """
        if n_pairs <= 0:
            return []

        # Generate random weights
        long_weights = [random.random() for _ in range(n_pairs)]
        short_weights = [random.random() for _ in range(n_pairs)]

        # Normalize
        long_sum = sum(long_weights)
        short_sum = sum(short_weights)

        pairs = []
        for i in range(n_pairs):
            long_size = total_size * Decimal(str(long_weights[i] / long_sum))
            short_size = total_size * Decimal(str(short_weights[i] / short_sum))
            pairs.append((long_size, short_size))

        return pairs

    def add_noise_to_price(
        self,
        price: Decimal,
        tick_size: Decimal,
        ticks: int = 2,
    ) -> Decimal:
        """Add random tick noise to price.

        Args:
            price: Base price
            tick_size: Minimum price increment
            ticks: Maximum ticks to add/subtract

        Returns:
            Price with noise
        """
        tick_offset = random.randint(-ticks, ticks)
        return price + (tick_size * tick_offset)

    def get_random_subset(
        self,
        items: List,
        min_count: int = 1,
        max_count: int = None,
    ) -> List:
        """Get random subset of items.

        Args:
            items: Items to select from
            min_count: Minimum items to select
            max_count: Maximum items (default: all)

        Returns:
            Random subset
        """
        if not items:
            return []

        if max_count is None:
            max_count = len(items)

        count = random.randint(min_count, min(max_count, len(items)))
        return random.sample(items, count)
