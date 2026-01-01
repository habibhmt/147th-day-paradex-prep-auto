"""Pattern breaking utilities for anti-detection."""

import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PatternBreaker:
    """Breaks predictable trading patterns.

    Tracks recent patterns and ensures variation:
    - Account execution order
    - Trade timing
    - Entry/exit patterns
    - Skip intervals

    Helps avoid detection as coordinated accounts.
    """

    history_size: int = 50  # Remember last N patterns

    # Pattern history per type
    _account_order_history: Deque[List[str]] = field(default_factory=lambda: deque(maxlen=50))
    _timing_history: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    _entry_pattern_history: Dict[str, Deque[str]] = field(default_factory=dict)
    _last_trade_time: Dict[str, float] = field(default_factory=dict)
    _skip_counter: int = 0

    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._account_order_history = deque(maxlen=self.history_size)
        self._timing_history = deque(maxlen=self.history_size)
        self._entry_pattern_history = {}
        self._last_trade_time = {}
        self._skip_counter = 0

    def get_account_order(self, accounts: List[str]) -> List[str]:
        """Get varied account execution order.

        Avoids repeating recent orders.

        Args:
            accounts: List of account IDs

        Returns:
            Shuffled account order
        """
        if len(accounts) <= 1:
            return accounts

        # Try to avoid recent patterns
        for _ in range(10):  # Max attempts
            shuffled = accounts.copy()
            random.shuffle(shuffled)

            # Check if this order was used recently
            if shuffled not in list(self._account_order_history)[-5:]:
                break

        self._account_order_history.append(shuffled)
        return shuffled

    def should_skip_interval(
        self,
        base_probability: float = 0.1,
        max_consecutive_skips: int = 2,
    ) -> bool:
        """Decide whether to skip a trading interval.

        Avoids too many consecutive skips.

        Args:
            base_probability: Base skip probability
            max_consecutive_skips: Max skips in a row

        Returns:
            True if should skip
        """
        if self._skip_counter >= max_consecutive_skips:
            self._skip_counter = 0
            return False

        if random.random() < base_probability:
            self._skip_counter += 1
            logger.debug(f"Skipping interval ({self._skip_counter} consecutive)")
            return True

        self._skip_counter = 0
        return False

    def get_entry_pattern(
        self,
        account_id: str,
        patterns: List[str] = None,
    ) -> str:
        """Get varied entry pattern for account.

        Args:
            account_id: Account ID
            patterns: Available patterns (default: immediate, staged, delayed)

        Returns:
            Selected pattern
        """
        if patterns is None:
            patterns = ["immediate", "staged", "delayed"]

        if account_id not in self._entry_pattern_history:
            self._entry_pattern_history[account_id] = deque(maxlen=10)

        recent = list(self._entry_pattern_history[account_id])

        # Avoid last 2 patterns
        available = [p for p in patterns if p not in recent[-2:]]
        if not available:
            available = patterns

        pattern = random.choice(available)
        self._entry_pattern_history[account_id].append(pattern)

        return pattern

    def get_varied_interval(
        self,
        base_interval: float,
        variance: float = 0.3,
    ) -> float:
        """Get varied interval avoiding recent patterns.

        Args:
            base_interval: Base interval in seconds
            variance: Variance factor

        Returns:
            Varied interval
        """
        # Generate candidate
        factor = random.uniform(1 - variance, 1 + variance)
        candidate = base_interval * factor

        # Avoid being too similar to recent intervals
        recent = list(self._timing_history)[-5:]
        for _ in range(5):  # Max retries
            too_similar = any(
                abs(candidate - r) < base_interval * 0.1 for r in recent
            )
            if not too_similar:
                break
            factor = random.uniform(1 - variance, 1 + variance)
            candidate = base_interval * factor

        self._timing_history.append(candidate)
        return candidate

    def get_time_since_last_trade(self, account_id: str) -> Optional[float]:
        """Get time since last trade for account.

        Args:
            account_id: Account ID

        Returns:
            Seconds since last trade, or None if no history
        """
        last = self._last_trade_time.get(account_id)
        if last is None:
            return None
        return time.time() - last

    def record_trade(self, account_id: str) -> None:
        """Record trade time for account.

        Args:
            account_id: Account that traded
        """
        self._last_trade_time[account_id] = time.time()

    def get_staggered_delays(
        self,
        n_accounts: int,
        min_delay: float = 0.5,
        max_delay: float = 5.0,
    ) -> List[float]:
        """Get staggered delays for multiple accounts.

        Ensures accounts don't trade at exactly the same time.

        Args:
            n_accounts: Number of accounts
            min_delay: Minimum delay
            max_delay: Maximum delay

        Returns:
            List of delays for each account
        """
        if n_accounts <= 0:
            return []

        # Generate base delays
        delays = []
        for i in range(n_accounts):
            # Stagger based on position
            base = min_delay + (max_delay - min_delay) * (i / max(1, n_accounts - 1))
            # Add random variance
            variance = random.uniform(-0.2, 0.2) * (max_delay - min_delay)
            delay = max(min_delay, base + variance)
            delays.append(delay)

        # Shuffle to avoid predictable order
        random.shuffle(delays)
        return delays

    def should_split_order(
        self,
        size: float,
        threshold: float = 1000,
        probability: float = 0.3,
    ) -> bool:
        """Decide whether to split a large order.

        Splitting breaks patterns of large single orders.

        Args:
            size: Order size
            threshold: Size threshold for considering split
            probability: Probability to split if above threshold

        Returns:
            True if should split
        """
        if size < threshold:
            return False
        return random.random() < probability

    def get_split_sizes(
        self,
        total_size: float,
        min_parts: int = 2,
        max_parts: int = 4,
    ) -> List[float]:
        """Split order into multiple parts.

        Args:
            total_size: Total size to split
            min_parts: Minimum parts
            max_parts: Maximum parts

        Returns:
            List of part sizes summing to total
        """
        n_parts = random.randint(min_parts, max_parts)

        # Generate random weights
        weights = [random.random() for _ in range(n_parts)]
        total_weight = sum(weights)

        # Convert to sizes
        sizes = [total_size * (w / total_weight) for w in weights]

        return sizes

    def clear_history(self) -> None:
        """Clear all pattern history."""
        self._account_order_history.clear()
        self._timing_history.clear()
        self._entry_pattern_history.clear()
        self._last_trade_time.clear()
        self._skip_counter = 0
        logger.info("Pattern history cleared")

    def stats(self) -> dict:
        """Get pattern breaker statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "order_history_size": len(self._account_order_history),
            "timing_history_size": len(self._timing_history),
            "accounts_tracked": len(self._entry_pattern_history),
            "trade_times_tracked": len(self._last_trade_time),
            "consecutive_skips": self._skip_counter,
        }
