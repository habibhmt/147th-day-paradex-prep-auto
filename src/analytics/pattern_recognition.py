"""
Chart Pattern Recognition for Paradex.

Detects classic chart patterns: head and shoulders, double top/bottom, triangles,
flags, wedges, channels, and candlestick patterns.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional
import logging


logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of chart patterns."""
    # Reversal patterns
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    ROUNDING_TOP = "rounding_top"
    ROUNDING_BOTTOM = "rounding_bottom"

    # Continuation patterns
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    BULL_FLAG = "bull_flag"
    BEAR_FLAG = "bear_flag"
    BULL_PENNANT = "bull_pennant"
    BEAR_PENNANT = "bear_pennant"
    RISING_WEDGE = "rising_wedge"
    FALLING_WEDGE = "falling_wedge"
    ASCENDING_CHANNEL = "ascending_channel"
    DESCENDING_CHANNEL = "descending_channel"
    HORIZONTAL_CHANNEL = "horizontal_channel"

    # Candlestick patterns
    DOJI = "doji"
    HAMMER = "hammer"
    INVERTED_HAMMER = "inverted_hammer"
    HANGING_MAN = "hanging_man"
    SHOOTING_STAR = "shooting_star"
    ENGULFING_BULLISH = "engulfing_bullish"
    ENGULFING_BEARISH = "engulfing_bearish"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"
    HARAMI_BULLISH = "harami_bullish"
    HARAMI_BEARISH = "harami_bearish"
    PIERCING_LINE = "piercing_line"
    DARK_CLOUD_COVER = "dark_cloud_cover"
    TWEEZER_TOP = "tweezer_top"
    TWEEZER_BOTTOM = "tweezer_bottom"
    MARUBOZU_BULLISH = "marubozu_bullish"
    MARUBOZU_BEARISH = "marubozu_bearish"


class PatternDirection(Enum):
    """Pattern direction/bias."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class PatternStrength(Enum):
    """Pattern strength/reliability."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class PatternStatus(Enum):
    """Pattern status."""
    FORMING = "forming"
    COMPLETE = "complete"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    INVALIDATED = "invalidated"


@dataclass
class OHLCV:
    """OHLCV data point."""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": str(self.open),
            "high": str(self.high),
            "low": str(self.low),
            "close": str(self.close),
            "volume": str(self.volume),
        }

    @property
    def is_bullish(self) -> bool:
        """Check if candle is bullish."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """Check if candle is bearish."""
        return self.close < self.open

    @property
    def body_size(self) -> Decimal:
        """Get candle body size."""
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> Decimal:
        """Get upper wick size."""
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> Decimal:
        """Get lower wick size."""
        return min(self.open, self.close) - self.low

    @property
    def range(self) -> Decimal:
        """Get candle range."""
        return self.high - self.low


@dataclass
class PatternPoint:
    """Key point in a pattern."""
    index: int
    price: Decimal
    timestamp: datetime
    point_type: str  # e.g., "high", "low", "neckline"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "price": str(self.price),
            "timestamp": self.timestamp.isoformat(),
            "point_type": self.point_type,
        }


@dataclass
class Pattern:
    """Detected chart pattern."""
    id: str
    pattern_type: PatternType
    symbol: str
    direction: PatternDirection
    strength: PatternStrength
    status: PatternStatus
    start_index: int
    end_index: int
    start_time: datetime
    end_time: datetime
    entry_price: Optional[Decimal] = None
    target_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    confidence: Decimal = Decimal("0.5")
    points: list[PatternPoint] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "pattern_type": self.pattern_type.value,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "strength": self.strength.value,
            "status": self.status.value,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "entry_price": str(self.entry_price) if self.entry_price else None,
            "target_price": str(self.target_price) if self.target_price else None,
            "stop_loss": str(self.stop_loss) if self.stop_loss else None,
            "confidence": str(self.confidence),
            "points": [p.to_dict() for p in self.points],
            "metadata": self.metadata,
            "detected_at": self.detected_at.isoformat(),
        }

    @property
    def risk_reward(self) -> Optional[Decimal]:
        """Calculate risk/reward ratio."""
        if not all([self.entry_price, self.target_price, self.stop_loss]):
            return None
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.target_price - self.entry_price)
        if risk == 0:
            return None
        return reward / risk


class PivotFinder:
    """Find pivot highs and lows in price data."""

    def __init__(self, left_bars: int = 5, right_bars: int = 5):
        """Initialize pivot finder."""
        self.left_bars = left_bars
        self.right_bars = right_bars

    def find_pivot_highs(self, candles: list[OHLCV]) -> list[tuple[int, Decimal]]:
        """Find pivot high points."""
        pivots = []
        for i in range(self.left_bars, len(candles) - self.right_bars):
            is_pivot = True
            high = candles[i].high

            for j in range(i - self.left_bars, i):
                if candles[j].high >= high:
                    is_pivot = False
                    break

            if is_pivot:
                for j in range(i + 1, i + self.right_bars + 1):
                    if candles[j].high > high:
                        is_pivot = False
                        break

            if is_pivot:
                pivots.append((i, high))

        return pivots

    def find_pivot_lows(self, candles: list[OHLCV]) -> list[tuple[int, Decimal]]:
        """Find pivot low points."""
        pivots = []
        for i in range(self.left_bars, len(candles) - self.right_bars):
            is_pivot = True
            low = candles[i].low

            for j in range(i - self.left_bars, i):
                if candles[j].low <= low:
                    is_pivot = False
                    break

            if is_pivot:
                for j in range(i + 1, i + self.right_bars + 1):
                    if candles[j].low < low:
                        is_pivot = False
                        break

            if is_pivot:
                pivots.append((i, low))

        return pivots


class TrendlineAnalyzer:
    """Analyze trendlines in price data."""

    def __init__(self, min_touches: int = 2, tolerance_pct: Decimal = Decimal("0.01")):
        """Initialize trendline analyzer."""
        self.min_touches = min_touches
        self.tolerance_pct = tolerance_pct

    def find_support_line(
        self,
        candles: list[OHLCV],
        pivot_lows: list[tuple[int, Decimal]],
    ) -> Optional[tuple[Decimal, Decimal]]:
        """Find support trendline (slope, intercept)."""
        if len(pivot_lows) < 2:
            return None

        best_line = None
        best_touches = 0

        for i in range(len(pivot_lows) - 1):
            for j in range(i + 1, len(pivot_lows)):
                idx1, price1 = pivot_lows[i]
                idx2, price2 = pivot_lows[j]

                if idx2 == idx1:
                    continue

                slope = (price2 - price1) / Decimal(idx2 - idx1)
                intercept = price1 - slope * Decimal(idx1)

                touches = self._count_touches(candles, slope, intercept, "low")

                if touches >= self.min_touches and touches > best_touches:
                    best_touches = touches
                    best_line = (slope, intercept)

        return best_line

    def find_resistance_line(
        self,
        candles: list[OHLCV],
        pivot_highs: list[tuple[int, Decimal]],
    ) -> Optional[tuple[Decimal, Decimal]]:
        """Find resistance trendline (slope, intercept)."""
        if len(pivot_highs) < 2:
            return None

        best_line = None
        best_touches = 0

        for i in range(len(pivot_highs) - 1):
            for j in range(i + 1, len(pivot_highs)):
                idx1, price1 = pivot_highs[i]
                idx2, price2 = pivot_highs[j]

                if idx2 == idx1:
                    continue

                slope = (price2 - price1) / Decimal(idx2 - idx1)
                intercept = price1 - slope * Decimal(idx1)

                touches = self._count_touches(candles, slope, intercept, "high")

                if touches >= self.min_touches and touches > best_touches:
                    best_touches = touches
                    best_line = (slope, intercept)

        return best_line

    def _count_touches(
        self,
        candles: list[OHLCV],
        slope: Decimal,
        intercept: Decimal,
        price_type: str,
    ) -> int:
        """Count touches on trendline."""
        touches = 0
        for i, candle in enumerate(candles):
            line_price = slope * Decimal(i) + intercept
            actual_price = candle.low if price_type == "low" else candle.high
            tolerance = line_price * self.tolerance_pct

            if abs(actual_price - line_price) <= tolerance:
                touches += 1

        return touches


class CandlestickPatternDetector:
    """Detect candlestick patterns."""

    def __init__(self, body_threshold: Decimal = Decimal("0.1")):
        """Initialize candlestick pattern detector."""
        self.body_threshold = body_threshold

    def detect_all(self, candles: list[OHLCV], symbol: str) -> list[Pattern]:
        """Detect all candlestick patterns."""
        patterns = []

        if len(candles) >= 1:
            patterns.extend(self._detect_single_candle_patterns(candles, symbol))

        if len(candles) >= 2:
            patterns.extend(self._detect_double_candle_patterns(candles, symbol))

        if len(candles) >= 3:
            patterns.extend(self._detect_triple_candle_patterns(candles, symbol))

        return patterns

    def _detect_single_candle_patterns(
        self,
        candles: list[OHLCV],
        symbol: str,
    ) -> list[Pattern]:
        """Detect single candlestick patterns."""
        patterns = []

        for i, candle in enumerate(candles):
            if self._is_doji(candle):
                patterns.append(self._create_pattern(
                    PatternType.DOJI, symbol, i, candle,
                    PatternDirection.NEUTRAL, PatternStrength.MODERATE,
                ))

            if self._is_hammer(candle):
                patterns.append(self._create_pattern(
                    PatternType.HAMMER, symbol, i, candle,
                    PatternDirection.BULLISH, PatternStrength.MODERATE,
                ))

            if self._is_inverted_hammer(candle):
                patterns.append(self._create_pattern(
                    PatternType.INVERTED_HAMMER, symbol, i, candle,
                    PatternDirection.BULLISH, PatternStrength.WEAK,
                ))

            if self._is_shooting_star(candle):
                patterns.append(self._create_pattern(
                    PatternType.SHOOTING_STAR, symbol, i, candle,
                    PatternDirection.BEARISH, PatternStrength.MODERATE,
                ))

            if self._is_marubozu_bullish(candle):
                patterns.append(self._create_pattern(
                    PatternType.MARUBOZU_BULLISH, symbol, i, candle,
                    PatternDirection.BULLISH, PatternStrength.STRONG,
                ))

            if self._is_marubozu_bearish(candle):
                patterns.append(self._create_pattern(
                    PatternType.MARUBOZU_BEARISH, symbol, i, candle,
                    PatternDirection.BEARISH, PatternStrength.STRONG,
                ))

        return patterns

    def _detect_double_candle_patterns(
        self,
        candles: list[OHLCV],
        symbol: str,
    ) -> list[Pattern]:
        """Detect double candlestick patterns."""
        patterns = []

        for i in range(1, len(candles)):
            prev = candles[i - 1]
            curr = candles[i]

            if self._is_engulfing_bullish(prev, curr):
                patterns.append(self._create_pattern(
                    PatternType.ENGULFING_BULLISH, symbol, i - 1, curr,
                    PatternDirection.BULLISH, PatternStrength.STRONG,
                    start_candle=prev,
                ))

            if self._is_engulfing_bearish(prev, curr):
                patterns.append(self._create_pattern(
                    PatternType.ENGULFING_BEARISH, symbol, i - 1, curr,
                    PatternDirection.BEARISH, PatternStrength.STRONG,
                    start_candle=prev,
                ))

            if self._is_harami_bullish(prev, curr):
                patterns.append(self._create_pattern(
                    PatternType.HARAMI_BULLISH, symbol, i - 1, curr,
                    PatternDirection.BULLISH, PatternStrength.MODERATE,
                    start_candle=prev,
                ))

            if self._is_harami_bearish(prev, curr):
                patterns.append(self._create_pattern(
                    PatternType.HARAMI_BEARISH, symbol, i - 1, curr,
                    PatternDirection.BEARISH, PatternStrength.MODERATE,
                    start_candle=prev,
                ))

            if self._is_piercing_line(prev, curr):
                patterns.append(self._create_pattern(
                    PatternType.PIERCING_LINE, symbol, i - 1, curr,
                    PatternDirection.BULLISH, PatternStrength.MODERATE,
                    start_candle=prev,
                ))

            if self._is_dark_cloud_cover(prev, curr):
                patterns.append(self._create_pattern(
                    PatternType.DARK_CLOUD_COVER, symbol, i - 1, curr,
                    PatternDirection.BEARISH, PatternStrength.MODERATE,
                    start_candle=prev,
                ))

            if self._is_tweezer_top(prev, curr):
                patterns.append(self._create_pattern(
                    PatternType.TWEEZER_TOP, symbol, i - 1, curr,
                    PatternDirection.BEARISH, PatternStrength.MODERATE,
                    start_candle=prev,
                ))

            if self._is_tweezer_bottom(prev, curr):
                patterns.append(self._create_pattern(
                    PatternType.TWEEZER_BOTTOM, symbol, i - 1, curr,
                    PatternDirection.BULLISH, PatternStrength.MODERATE,
                    start_candle=prev,
                ))

        return patterns

    def _detect_triple_candle_patterns(
        self,
        candles: list[OHLCV],
        symbol: str,
    ) -> list[Pattern]:
        """Detect triple candlestick patterns."""
        patterns = []

        for i in range(2, len(candles)):
            c1, c2, c3 = candles[i - 2], candles[i - 1], candles[i]

            if self._is_morning_star(c1, c2, c3):
                patterns.append(self._create_pattern(
                    PatternType.MORNING_STAR, symbol, i - 2, c3,
                    PatternDirection.BULLISH, PatternStrength.STRONG,
                    start_candle=c1,
                ))

            if self._is_evening_star(c1, c2, c3):
                patterns.append(self._create_pattern(
                    PatternType.EVENING_STAR, symbol, i - 2, c3,
                    PatternDirection.BEARISH, PatternStrength.STRONG,
                    start_candle=c1,
                ))

            if self._is_three_white_soldiers(c1, c2, c3):
                patterns.append(self._create_pattern(
                    PatternType.THREE_WHITE_SOLDIERS, symbol, i - 2, c3,
                    PatternDirection.BULLISH, PatternStrength.VERY_STRONG,
                    start_candle=c1,
                ))

            if self._is_three_black_crows(c1, c2, c3):
                patterns.append(self._create_pattern(
                    PatternType.THREE_BLACK_CROWS, symbol, i - 2, c3,
                    PatternDirection.BEARISH, PatternStrength.VERY_STRONG,
                    start_candle=c1,
                ))

        return patterns

    def _is_doji(self, candle: OHLCV) -> bool:
        """Check if candle is a doji."""
        if candle.range == 0:
            return True
        body_pct = candle.body_size / candle.range
        return body_pct < self.body_threshold

    def _is_hammer(self, candle: OHLCV) -> bool:
        """Check if candle is a hammer."""
        if candle.range == 0:
            return False
        body_pct = candle.body_size / candle.range
        lower_wick_pct = candle.lower_wick / candle.range
        upper_wick_pct = candle.upper_wick / candle.range

        return (
            body_pct < Decimal("0.35") and
            lower_wick_pct > Decimal("0.6") and
            upper_wick_pct < Decimal("0.1")
        )

    def _is_inverted_hammer(self, candle: OHLCV) -> bool:
        """Check if candle is an inverted hammer."""
        if candle.range == 0:
            return False
        body_pct = candle.body_size / candle.range
        upper_wick_pct = candle.upper_wick / candle.range
        lower_wick_pct = candle.lower_wick / candle.range

        return (
            body_pct < Decimal("0.35") and
            upper_wick_pct > Decimal("0.6") and
            lower_wick_pct < Decimal("0.1")
        )

    def _is_shooting_star(self, candle: OHLCV) -> bool:
        """Check if candle is a shooting star."""
        return self._is_inverted_hammer(candle) and candle.is_bearish

    def _is_marubozu_bullish(self, candle: OHLCV) -> bool:
        """Check if candle is a bullish marubozu."""
        if candle.range == 0:
            return False
        body_pct = candle.body_size / candle.range
        upper_wick_pct = candle.upper_wick / candle.range
        lower_wick_pct = candle.lower_wick / candle.range

        return (
            candle.is_bullish and
            body_pct > Decimal("0.95") and
            upper_wick_pct < Decimal("0.025") and
            lower_wick_pct < Decimal("0.025")
        )

    def _is_marubozu_bearish(self, candle: OHLCV) -> bool:
        """Check if candle is a bearish marubozu."""
        if candle.range == 0:
            return False
        body_pct = candle.body_size / candle.range
        upper_wick_pct = candle.upper_wick / candle.range
        lower_wick_pct = candle.lower_wick / candle.range

        return (
            candle.is_bearish and
            body_pct > Decimal("0.95") and
            upper_wick_pct < Decimal("0.025") and
            lower_wick_pct < Decimal("0.025")
        )

    def _is_engulfing_bullish(self, prev: OHLCV, curr: OHLCV) -> bool:
        """Check for bullish engulfing pattern."""
        return (
            prev.is_bearish and
            curr.is_bullish and
            curr.open < prev.close and
            curr.close > prev.open
        )

    def _is_engulfing_bearish(self, prev: OHLCV, curr: OHLCV) -> bool:
        """Check for bearish engulfing pattern."""
        return (
            prev.is_bullish and
            curr.is_bearish and
            curr.open > prev.close and
            curr.close < prev.open
        )

    def _is_harami_bullish(self, prev: OHLCV, curr: OHLCV) -> bool:
        """Check for bullish harami pattern."""
        return (
            prev.is_bearish and
            curr.is_bullish and
            curr.open > prev.close and
            curr.close < prev.open
        )

    def _is_harami_bearish(self, prev: OHLCV, curr: OHLCV) -> bool:
        """Check for bearish harami pattern."""
        return (
            prev.is_bullish and
            curr.is_bearish and
            curr.open < prev.close and
            curr.close > prev.open
        )

    def _is_piercing_line(self, prev: OHLCV, curr: OHLCV) -> bool:
        """Check for piercing line pattern."""
        if not prev.is_bearish or not curr.is_bullish:
            return False

        midpoint = (prev.open + prev.close) / 2
        return curr.open < prev.close and curr.close > midpoint

    def _is_dark_cloud_cover(self, prev: OHLCV, curr: OHLCV) -> bool:
        """Check for dark cloud cover pattern."""
        if not prev.is_bullish or not curr.is_bearish:
            return False

        midpoint = (prev.open + prev.close) / 2
        return curr.open > prev.close and curr.close < midpoint

    def _is_tweezer_top(self, prev: OHLCV, curr: OHLCV) -> bool:
        """Check for tweezer top pattern."""
        tolerance = (prev.high + curr.high) / 2 * Decimal("0.001")
        return (
            prev.is_bullish and
            curr.is_bearish and
            abs(prev.high - curr.high) <= tolerance
        )

    def _is_tweezer_bottom(self, prev: OHLCV, curr: OHLCV) -> bool:
        """Check for tweezer bottom pattern."""
        tolerance = (prev.low + curr.low) / 2 * Decimal("0.001")
        return (
            prev.is_bearish and
            curr.is_bullish and
            abs(prev.low - curr.low) <= tolerance
        )

    def _is_morning_star(self, c1: OHLCV, c2: OHLCV, c3: OHLCV) -> bool:
        """Check for morning star pattern."""
        if not c1.is_bearish or not c3.is_bullish:
            return False

        # Small middle body
        if c1.body_size > 0 and c2.body_size / c1.body_size > Decimal("0.5"):
            return False

        # Third closes above midpoint of first
        midpoint = (c1.open + c1.close) / 2
        return c3.close > midpoint

    def _is_evening_star(self, c1: OHLCV, c2: OHLCV, c3: OHLCV) -> bool:
        """Check for evening star pattern."""
        if not c1.is_bullish or not c3.is_bearish:
            return False

        # Small middle body
        if c1.body_size > 0 and c2.body_size / c1.body_size > Decimal("0.5"):
            return False

        # Third closes below midpoint of first
        midpoint = (c1.open + c1.close) / 2
        return c3.close < midpoint

    def _is_three_white_soldiers(self, c1: OHLCV, c2: OHLCV, c3: OHLCV) -> bool:
        """Check for three white soldiers pattern."""
        return (
            c1.is_bullish and c2.is_bullish and c3.is_bullish and
            c2.open > c1.open and c2.close > c1.close and
            c3.open > c2.open and c3.close > c2.close
        )

    def _is_three_black_crows(self, c1: OHLCV, c2: OHLCV, c3: OHLCV) -> bool:
        """Check for three black crows pattern."""
        return (
            c1.is_bearish and c2.is_bearish and c3.is_bearish and
            c2.open < c1.open and c2.close < c1.close and
            c3.open < c2.open and c3.close < c2.close
        )

    def _create_pattern(
        self,
        pattern_type: PatternType,
        symbol: str,
        start_idx: int,
        end_candle: OHLCV,
        direction: PatternDirection,
        strength: PatternStrength,
        start_candle: Optional[OHLCV] = None,
    ) -> Pattern:
        """Create pattern object."""
        import hashlib
        pattern_id = hashlib.md5(
            f"{pattern_type.value}:{symbol}:{start_idx}:{end_candle.timestamp.isoformat()}".encode()
        ).hexdigest()[:12]

        start_time = start_candle.timestamp if start_candle else end_candle.timestamp
        end_idx = start_idx if start_candle is None else start_idx + 1

        return Pattern(
            id=pattern_id,
            pattern_type=pattern_type,
            symbol=symbol,
            direction=direction,
            strength=strength,
            status=PatternStatus.COMPLETE,
            start_index=start_idx,
            end_index=end_idx,
            start_time=start_time,
            end_time=end_candle.timestamp,
            entry_price=end_candle.close,
            confidence=Decimal("0.7") if strength == PatternStrength.STRONG else Decimal("0.5"),
        )


class ChartPatternDetector:
    """Detect chart patterns like head and shoulders, triangles, etc."""

    def __init__(
        self,
        pivot_left: int = 5,
        pivot_right: int = 5,
        tolerance_pct: Decimal = Decimal("0.02"),
    ):
        """Initialize chart pattern detector."""
        self.pivot_finder = PivotFinder(pivot_left, pivot_right)
        self.trendline_analyzer = TrendlineAnalyzer(tolerance_pct=tolerance_pct)
        self.tolerance_pct = tolerance_pct

    def detect_all(self, candles: list[OHLCV], symbol: str) -> list[Pattern]:
        """Detect all chart patterns."""
        if len(candles) < 20:
            return []

        patterns = []
        pivot_highs = self.pivot_finder.find_pivot_highs(candles)
        pivot_lows = self.pivot_finder.find_pivot_lows(candles)

        patterns.extend(self._detect_double_top(candles, symbol, pivot_highs))
        patterns.extend(self._detect_double_bottom(candles, symbol, pivot_lows))
        patterns.extend(self._detect_head_shoulders(candles, symbol, pivot_highs, pivot_lows))
        patterns.extend(self._detect_triangles(candles, symbol, pivot_highs, pivot_lows))
        patterns.extend(self._detect_channels(candles, symbol, pivot_highs, pivot_lows))

        return patterns

    def _detect_double_top(
        self,
        candles: list[OHLCV],
        symbol: str,
        pivot_highs: list[tuple[int, Decimal]],
    ) -> list[Pattern]:
        """Detect double top patterns."""
        patterns = []

        if len(pivot_highs) < 2:
            return patterns

        for i in range(len(pivot_highs) - 1):
            idx1, price1 = pivot_highs[i]
            idx2, price2 = pivot_highs[i + 1]

            # Peaks should be roughly equal
            tolerance = max(price1, price2) * self.tolerance_pct
            if abs(price1 - price2) > tolerance:
                continue

            # Should have at least some bars between peaks
            if idx2 - idx1 < 5:
                continue

            # Find the neckline (lowest low between peaks)
            lows_between = [c.low for c in candles[idx1:idx2 + 1]]
            neckline = min(lows_between)

            pattern = self._create_chart_pattern(
                PatternType.DOUBLE_TOP,
                symbol,
                candles,
                idx1,
                idx2,
                PatternDirection.BEARISH,
                entry_price=neckline,
                stop_loss=max(price1, price2),
                target_price=neckline - (max(price1, price2) - neckline),
            )
            patterns.append(pattern)

        return patterns

    def _detect_double_bottom(
        self,
        candles: list[OHLCV],
        symbol: str,
        pivot_lows: list[tuple[int, Decimal]],
    ) -> list[Pattern]:
        """Detect double bottom patterns."""
        patterns = []

        if len(pivot_lows) < 2:
            return patterns

        for i in range(len(pivot_lows) - 1):
            idx1, price1 = pivot_lows[i]
            idx2, price2 = pivot_lows[i + 1]

            # Bottoms should be roughly equal
            tolerance = max(price1, price2) * self.tolerance_pct
            if abs(price1 - price2) > tolerance:
                continue

            # Should have at least some bars between bottoms
            if idx2 - idx1 < 5:
                continue

            # Find the neckline (highest high between bottoms)
            highs_between = [c.high for c in candles[idx1:idx2 + 1]]
            neckline = max(highs_between)

            pattern = self._create_chart_pattern(
                PatternType.DOUBLE_BOTTOM,
                symbol,
                candles,
                idx1,
                idx2,
                PatternDirection.BULLISH,
                entry_price=neckline,
                stop_loss=min(price1, price2),
                target_price=neckline + (neckline - min(price1, price2)),
            )
            patterns.append(pattern)

        return patterns

    def _detect_head_shoulders(
        self,
        candles: list[OHLCV],
        symbol: str,
        pivot_highs: list[tuple[int, Decimal]],
        pivot_lows: list[tuple[int, Decimal]],
    ) -> list[Pattern]:
        """Detect head and shoulders patterns."""
        patterns = []

        if len(pivot_highs) < 3:
            return patterns

        for i in range(len(pivot_highs) - 2):
            idx1, left_shoulder = pivot_highs[i]
            idx2, head = pivot_highs[i + 1]
            idx3, right_shoulder = pivot_highs[i + 2]

            # Head must be higher than shoulders
            if not (head > left_shoulder and head > right_shoulder):
                continue

            # Shoulders should be roughly equal
            tolerance = max(left_shoulder, right_shoulder) * self.tolerance_pct
            if abs(left_shoulder - right_shoulder) > tolerance * 2:
                continue

            # Find neckline
            neckline = min([c.low for c in candles[idx1:idx3 + 1]])

            pattern = self._create_chart_pattern(
                PatternType.HEAD_AND_SHOULDERS,
                symbol,
                candles,
                idx1,
                idx3,
                PatternDirection.BEARISH,
                entry_price=neckline,
                stop_loss=head,
                target_price=neckline - (head - neckline),
            )
            pattern.points = [
                PatternPoint(idx1, left_shoulder, candles[idx1].timestamp, "left_shoulder"),
                PatternPoint(idx2, head, candles[idx2].timestamp, "head"),
                PatternPoint(idx3, right_shoulder, candles[idx3].timestamp, "right_shoulder"),
            ]
            patterns.append(pattern)

        return patterns

    def _detect_triangles(
        self,
        candles: list[OHLCV],
        symbol: str,
        pivot_highs: list[tuple[int, Decimal]],
        pivot_lows: list[tuple[int, Decimal]],
    ) -> list[Pattern]:
        """Detect triangle patterns."""
        patterns = []

        if len(pivot_highs) < 2 or len(pivot_lows) < 2:
            return patterns

        support_line = self.trendline_analyzer.find_support_line(candles, pivot_lows)
        resistance_line = self.trendline_analyzer.find_resistance_line(candles, pivot_highs)

        if not support_line or not resistance_line:
            return patterns

        support_slope, _ = support_line
        resistance_slope, _ = resistance_line

        start_idx = min(pivot_highs[0][0], pivot_lows[0][0])
        end_idx = max(pivot_highs[-1][0], pivot_lows[-1][0])

        # Ascending triangle: flat resistance, rising support
        if abs(resistance_slope) < self.tolerance_pct and support_slope > Decimal("0"):
            patterns.append(self._create_chart_pattern(
                PatternType.ASCENDING_TRIANGLE,
                symbol,
                candles,
                start_idx,
                end_idx,
                PatternDirection.BULLISH,
            ))

        # Descending triangle: falling resistance, flat support
        elif resistance_slope < Decimal("0") and abs(support_slope) < self.tolerance_pct:
            patterns.append(self._create_chart_pattern(
                PatternType.DESCENDING_TRIANGLE,
                symbol,
                candles,
                start_idx,
                end_idx,
                PatternDirection.BEARISH,
            ))

        # Symmetrical triangle: converging lines
        elif (
            resistance_slope < Decimal("0") and
            support_slope > Decimal("0") and
            abs(resistance_slope) > self.tolerance_pct and
            abs(support_slope) > self.tolerance_pct
        ):
            patterns.append(self._create_chart_pattern(
                PatternType.SYMMETRICAL_TRIANGLE,
                symbol,
                candles,
                start_idx,
                end_idx,
                PatternDirection.NEUTRAL,
            ))

        return patterns

    def _detect_channels(
        self,
        candles: list[OHLCV],
        symbol: str,
        pivot_highs: list[tuple[int, Decimal]],
        pivot_lows: list[tuple[int, Decimal]],
    ) -> list[Pattern]:
        """Detect channel patterns."""
        patterns = []

        if len(pivot_highs) < 2 or len(pivot_lows) < 2:
            return patterns

        support_line = self.trendline_analyzer.find_support_line(candles, pivot_lows)
        resistance_line = self.trendline_analyzer.find_resistance_line(candles, pivot_highs)

        if not support_line or not resistance_line:
            return patterns

        support_slope, _ = support_line
        resistance_slope, _ = resistance_line

        start_idx = min(pivot_highs[0][0], pivot_lows[0][0])
        end_idx = max(pivot_highs[-1][0], pivot_lows[-1][0])

        # Check if slopes are parallel (similar slope)
        slope_diff = abs(resistance_slope - support_slope)
        avg_slope = (abs(resistance_slope) + abs(support_slope)) / 2

        if avg_slope > 0 and slope_diff / avg_slope < Decimal("0.3"):
            # Slopes are roughly parallel
            avg = (resistance_slope + support_slope) / 2

            if avg > self.tolerance_pct:
                patterns.append(self._create_chart_pattern(
                    PatternType.ASCENDING_CHANNEL,
                    symbol,
                    candles,
                    start_idx,
                    end_idx,
                    PatternDirection.BULLISH,
                ))
            elif avg < -self.tolerance_pct:
                patterns.append(self._create_chart_pattern(
                    PatternType.DESCENDING_CHANNEL,
                    symbol,
                    candles,
                    start_idx,
                    end_idx,
                    PatternDirection.BEARISH,
                ))
            else:
                patterns.append(self._create_chart_pattern(
                    PatternType.HORIZONTAL_CHANNEL,
                    symbol,
                    candles,
                    start_idx,
                    end_idx,
                    PatternDirection.NEUTRAL,
                ))

        return patterns

    def _create_chart_pattern(
        self,
        pattern_type: PatternType,
        symbol: str,
        candles: list[OHLCV],
        start_idx: int,
        end_idx: int,
        direction: PatternDirection,
        entry_price: Optional[Decimal] = None,
        stop_loss: Optional[Decimal] = None,
        target_price: Optional[Decimal] = None,
    ) -> Pattern:
        """Create chart pattern object."""
        import hashlib
        pattern_id = hashlib.md5(
            f"{pattern_type.value}:{symbol}:{start_idx}:{end_idx}".encode()
        ).hexdigest()[:12]

        return Pattern(
            id=pattern_id,
            pattern_type=pattern_type,
            symbol=symbol,
            direction=direction,
            strength=PatternStrength.MODERATE,
            status=PatternStatus.FORMING,
            start_index=start_idx,
            end_index=end_idx,
            start_time=candles[start_idx].timestamp,
            end_time=candles[end_idx].timestamp,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            confidence=Decimal("0.6"),
        )


class PatternRecognizer:
    """Main pattern recognition engine."""

    def __init__(self):
        """Initialize pattern recognizer."""
        self.candlestick_detector = CandlestickPatternDetector()
        self.chart_detector = ChartPatternDetector()
        self._detected_patterns: dict[str, list[Pattern]] = {}
        self._callbacks: list = []

    def analyze(self, candles: list[OHLCV], symbol: str) -> list[Pattern]:
        """Analyze candles for all patterns."""
        patterns = []

        # Detect candlestick patterns
        candlestick_patterns = self.candlestick_detector.detect_all(candles, symbol)
        patterns.extend(candlestick_patterns)

        # Detect chart patterns
        chart_patterns = self.chart_detector.detect_all(candles, symbol)
        patterns.extend(chart_patterns)

        # Store detected patterns
        self._detected_patterns[symbol] = patterns

        # Trigger callbacks
        for callback in self._callbacks:
            for pattern in patterns:
                callback(pattern)

        return patterns

    def get_patterns(
        self,
        symbol: Optional[str] = None,
        pattern_type: Optional[PatternType] = None,
        direction: Optional[PatternDirection] = None,
        min_strength: Optional[PatternStrength] = None,
    ) -> list[Pattern]:
        """Get detected patterns with filters."""
        if symbol:
            patterns = self._detected_patterns.get(symbol, [])
        else:
            patterns = []
            for p_list in self._detected_patterns.values():
                patterns.extend(p_list)

        if pattern_type:
            patterns = [p for p in patterns if p.pattern_type == pattern_type]

        if direction:
            patterns = [p for p in patterns if p.direction == direction]

        if min_strength:
            strength_order = [
                PatternStrength.WEAK,
                PatternStrength.MODERATE,
                PatternStrength.STRONG,
                PatternStrength.VERY_STRONG,
            ]
            min_idx = strength_order.index(min_strength)
            patterns = [
                p for p in patterns
                if strength_order.index(p.strength) >= min_idx
            ]

        return patterns

    def get_bullish_patterns(self, symbol: Optional[str] = None) -> list[Pattern]:
        """Get bullish patterns."""
        return self.get_patterns(symbol=symbol, direction=PatternDirection.BULLISH)

    def get_bearish_patterns(self, symbol: Optional[str] = None) -> list[Pattern]:
        """Get bearish patterns."""
        return self.get_patterns(symbol=symbol, direction=PatternDirection.BEARISH)

    def get_strong_patterns(self, symbol: Optional[str] = None) -> list[Pattern]:
        """Get strong patterns."""
        return self.get_patterns(
            symbol=symbol,
            min_strength=PatternStrength.STRONG,
        )

    def add_callback(self, callback) -> None:
        """Add pattern detection callback."""
        self._callbacks.append(callback)

    def remove_callback(self, callback) -> bool:
        """Remove pattern detection callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            return True
        return False

    def clear(self, symbol: Optional[str] = None) -> None:
        """Clear detected patterns."""
        if symbol:
            self._detected_patterns.pop(symbol, None)
        else:
            self._detected_patterns.clear()

    def get_summary(self, symbol: Optional[str] = None) -> dict:
        """Get pattern detection summary."""
        patterns = self.get_patterns(symbol=symbol)

        by_type = {}
        by_direction = {"bullish": 0, "bearish": 0, "neutral": 0}
        by_strength = {"weak": 0, "moderate": 0, "strong": 0, "very_strong": 0}

        for pattern in patterns:
            ptype = pattern.pattern_type.value
            by_type[ptype] = by_type.get(ptype, 0) + 1
            by_direction[pattern.direction.value] += 1
            by_strength[pattern.strength.value] += 1

        return {
            "total_patterns": len(patterns),
            "by_type": by_type,
            "by_direction": by_direction,
            "by_strength": by_strength,
            "symbols": list(self._detected_patterns.keys()),
        }


# Global instance
_pattern_recognizer: Optional[PatternRecognizer] = None


def get_pattern_recognizer() -> PatternRecognizer:
    """Get global pattern recognizer instance."""
    global _pattern_recognizer
    if _pattern_recognizer is None:
        _pattern_recognizer = PatternRecognizer()
    return _pattern_recognizer


def set_pattern_recognizer(recognizer: PatternRecognizer) -> None:
    """Set global pattern recognizer instance."""
    global _pattern_recognizer
    _pattern_recognizer = recognizer
