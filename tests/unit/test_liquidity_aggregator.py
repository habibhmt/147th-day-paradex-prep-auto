"""Tests for Liquidity Aggregator module."""

from datetime import datetime
from decimal import Decimal

import pytest

from src.analytics.liquidity_aggregator import (
    AggregatedLevel,
    AggregationMode,
    FillEstimate,
    LiquidityAggregator,
    LiquidityDepth,
    LiquidityLevel,
    LiquidityMetrics,
    LiquidityQuality,
    LiquiditySnapshot,
    LiquiditySource,
    LiquidityType,
    MultiMarketAggregator,
    get_aggregator,
    get_multi_aggregator,
    reset_aggregators,
)


class TestLiquiditySourceEnum:
    """Tests for LiquiditySource enum."""

    def test_all_sources(self):
        """Test all liquidity sources."""
        sources = [
            LiquiditySource.ORDER_BOOK,
            LiquiditySource.MARKET_MAKER,
            LiquiditySource.DARK_POOL,
            LiquiditySource.INTERNAL,
            LiquiditySource.EXTERNAL,
        ]
        assert len(sources) == 5

    def test_source_values(self):
        """Test source values."""
        assert LiquiditySource.ORDER_BOOK.value == "order_book"
        assert LiquiditySource.MARKET_MAKER.value == "market_maker"
        assert LiquiditySource.DARK_POOL.value == "dark_pool"


class TestLiquidityTypeEnum:
    """Tests for LiquidityType enum."""

    def test_all_types(self):
        """Test all liquidity types."""
        types = [LiquidityType.BID, LiquidityType.ASK, LiquidityType.BOTH]
        assert len(types) == 3

    def test_type_values(self):
        """Test type values."""
        assert LiquidityType.BID.value == "bid"
        assert LiquidityType.ASK.value == "ask"


class TestAggregationModeEnum:
    """Tests for AggregationMode enum."""

    def test_all_modes(self):
        """Test all aggregation modes."""
        modes = [
            AggregationMode.SUM,
            AggregationMode.WEIGHTED,
            AggregationMode.BEST_PRICE,
            AggregationMode.DEPTH_BASED,
        ]
        assert len(modes) == 4


class TestLiquidityQualityEnum:
    """Tests for LiquidityQuality enum."""

    def test_all_qualities(self):
        """Test all quality ratings."""
        qualities = [
            LiquidityQuality.HIGH,
            LiquidityQuality.MEDIUM,
            LiquidityQuality.LOW,
            LiquidityQuality.INDICATIVE,
        ]
        assert len(qualities) == 4


class TestLiquidityLevel:
    """Tests for LiquidityLevel dataclass."""

    def test_create_level(self):
        """Test creating a liquidity level."""
        level = LiquidityLevel(
            price=Decimal("50000"),
            size=Decimal("10"),
            source=LiquiditySource.ORDER_BOOK,
        )
        assert level.price == Decimal("50000")
        assert level.size == Decimal("10")
        assert level.source == LiquiditySource.ORDER_BOOK
        assert level.quality == LiquidityQuality.MEDIUM

    def test_level_with_all_fields(self):
        """Test level with all fields."""
        now = datetime.now()
        level = LiquidityLevel(
            price=Decimal("50000"),
            size=Decimal("10"),
            source=LiquiditySource.MARKET_MAKER,
            quality=LiquidityQuality.HIGH,
            timestamp=now,
            order_count=5,
            metadata={"firm": "test"},
        )
        assert level.quality == LiquidityQuality.HIGH
        assert level.timestamp == now
        assert level.order_count == 5
        assert level.metadata["firm"] == "test"

    def test_to_dict(self):
        """Test converting to dict."""
        level = LiquidityLevel(
            price=Decimal("50000"),
            size=Decimal("10"),
            source=LiquiditySource.ORDER_BOOK,
        )
        d = level.to_dict()
        assert d["price"] == 50000.0
        assert d["size"] == 10.0
        assert d["source"] == "order_book"


class TestAggregatedLevel:
    """Tests for AggregatedLevel dataclass."""

    def test_create_aggregated_level(self):
        """Test creating aggregated level."""
        level = AggregatedLevel(
            price=Decimal("50000"),
            total_size=Decimal("100"),
        )
        assert level.price == Decimal("50000")
        assert level.total_size == Decimal("100")
        assert level.sources == []

    def test_aggregated_with_sources(self):
        """Test aggregated level with sources."""
        level = AggregatedLevel(
            price=Decimal("50000"),
            total_size=Decimal("100"),
            sources=[LiquiditySource.ORDER_BOOK, LiquiditySource.MARKET_MAKER],
            weighted_quality=0.85,
            order_count=10,
        )
        assert len(level.sources) == 2
        assert level.weighted_quality == 0.85
        assert level.order_count == 10

    def test_to_dict(self):
        """Test converting to dict."""
        level = AggregatedLevel(
            price=Decimal("50000"),
            total_size=Decimal("100"),
            sources=[LiquiditySource.ORDER_BOOK],
        )
        d = level.to_dict()
        assert d["price"] == 50000.0
        assert d["total_size"] == 100.0
        assert "order_book" in d["sources"]


class TestLiquiditySnapshot:
    """Tests for LiquiditySnapshot dataclass."""

    def test_create_snapshot(self):
        """Test creating snapshot."""
        snapshot = LiquiditySnapshot(
            market="BTC-USD-PERP",
            timestamp=datetime.now(),
            bids=[],
            asks=[],
        )
        assert snapshot.market == "BTC-USD-PERP"
        assert snapshot.total_bid_liquidity == Decimal("0")

    def test_snapshot_with_data(self):
        """Test snapshot with bid/ask data."""
        bids = [AggregatedLevel(price=Decimal("50000"), total_size=Decimal("100"))]
        asks = [AggregatedLevel(price=Decimal("50010"), total_size=Decimal("80"))]

        snapshot = LiquiditySnapshot(
            market="BTC-USD-PERP",
            timestamp=datetime.now(),
            bids=bids,
            asks=asks,
            total_bid_liquidity=Decimal("100"),
            total_ask_liquidity=Decimal("80"),
            spread_bps=2.0,
            imbalance_ratio=0.556,
        )
        assert len(snapshot.bids) == 1
        assert len(snapshot.asks) == 1
        assert snapshot.spread_bps == 2.0

    def test_to_dict(self):
        """Test converting to dict."""
        snapshot = LiquiditySnapshot(
            market="BTC-USD-PERP",
            timestamp=datetime.now(),
            bids=[],
            asks=[],
        )
        d = snapshot.to_dict()
        assert d["market"] == "BTC-USD-PERP"
        assert "timestamp" in d


class TestLiquidityMetrics:
    """Tests for LiquidityMetrics dataclass."""

    def test_create_metrics(self):
        """Test creating metrics."""
        metrics = LiquidityMetrics(
            market="BTC-USD-PERP",
            timestamp=datetime.now(),
            best_bid=Decimal("50000"),
            best_ask=Decimal("50010"),
            mid_price=Decimal("50005"),
            spread_bps=2.0,
            bid_depth_1pct=Decimal("500"),
            ask_depth_1pct=Decimal("400"),
            bid_depth_5pct=Decimal("2000"),
            ask_depth_5pct=Decimal("1800"),
            total_bid_liquidity=Decimal("5000"),
            total_ask_liquidity=Decimal("4500"),
            imbalance_ratio=0.526,
            bid_levels=20,
            ask_levels=20,
            avg_bid_quality=0.8,
            avg_ask_quality=0.75,
        )
        assert metrics.mid_price == Decimal("50005")
        assert metrics.spread_bps == 2.0

    def test_to_dict(self):
        """Test converting to dict."""
        metrics = LiquidityMetrics(
            market="BTC-USD-PERP",
            timestamp=datetime.now(),
            best_bid=Decimal("50000"),
            best_ask=Decimal("50010"),
            mid_price=Decimal("50005"),
            spread_bps=2.0,
            bid_depth_1pct=Decimal("500"),
            ask_depth_1pct=Decimal("400"),
            bid_depth_5pct=Decimal("2000"),
            ask_depth_5pct=Decimal("1800"),
            total_bid_liquidity=Decimal("5000"),
            total_ask_liquidity=Decimal("4500"),
            imbalance_ratio=0.526,
            bid_levels=20,
            ask_levels=20,
            avg_bid_quality=0.8,
            avg_ask_quality=0.75,
        )
        d = metrics.to_dict()
        assert d["market"] == "BTC-USD-PERP"
        assert d["spread_bps"] == 2.0


class TestFillEstimate:
    """Tests for FillEstimate dataclass."""

    def test_create_fill_estimate(self):
        """Test creating fill estimate."""
        estimate = FillEstimate(
            market="BTC-USD-PERP",
            side=LiquidityType.BID,
            size=Decimal("10"),
            avg_price=Decimal("50005"),
            worst_price=Decimal("50010"),
            slippage_bps=1.0,
            levels_consumed=3,
        )
        assert estimate.market == "BTC-USD-PERP"
        assert estimate.levels_consumed == 3

    def test_to_dict(self):
        """Test converting to dict."""
        estimate = FillEstimate(
            market="BTC-USD-PERP",
            side=LiquidityType.BID,
            size=Decimal("10"),
            avg_price=Decimal("50005"),
            worst_price=Decimal("50010"),
            slippage_bps=1.0,
            levels_consumed=3,
            fills=[(Decimal("50000"), Decimal("5")), (Decimal("50010"), Decimal("5"))],
        )
        d = estimate.to_dict()
        assert d["side"] == "bid"
        assert len(d["fills"]) == 2


class TestLiquidityDepth:
    """Tests for LiquidityDepth dataclass."""

    def test_create_depth(self):
        """Test creating depth profile."""
        depth = LiquidityDepth(
            market="BTC-USD-PERP",
            timestamp=datetime.now(),
            price_levels=[Decimal("49900"), Decimal("50000"), Decimal("50100")],
            bid_cumulative=[Decimal("100"), Decimal("50"), Decimal("0")],
            ask_cumulative=[Decimal("0"), Decimal("50"), Decimal("100")],
        )
        assert len(depth.price_levels) == 3

    def test_to_dict(self):
        """Test converting to dict."""
        depth = LiquidityDepth(
            market="BTC-USD-PERP",
            timestamp=datetime.now(),
            price_levels=[Decimal("50000")],
            bid_cumulative=[Decimal("100")],
            ask_cumulative=[Decimal("100")],
        )
        d = depth.to_dict()
        assert d["market"] == "BTC-USD-PERP"


class TestLiquidityAggregator:
    """Tests for LiquidityAggregator class."""

    @pytest.fixture
    def aggregator(self):
        """Create aggregator."""
        return LiquidityAggregator()

    @pytest.fixture
    def sample_book(self, aggregator):
        """Create aggregator with sample order book."""
        bids = [
            (Decimal("50000"), Decimal("10")),
            (Decimal("49990"), Decimal("20")),
            (Decimal("49980"), Decimal("30")),
        ]
        asks = [
            (Decimal("50010"), Decimal("10")),
            (Decimal("50020"), Decimal("20")),
            (Decimal("50030"), Decimal("30")),
        ]
        aggregator.update_order_book("BTC-USD-PERP", bids, asks)
        return aggregator

    def test_init_defaults(self):
        """Test default initialization."""
        agg = LiquidityAggregator()
        assert agg.max_levels == 50
        assert agg.aggregation_mode == AggregationMode.SUM
        assert agg.stale_threshold_ms == 5000

    def test_init_custom(self):
        """Test custom initialization."""
        agg = LiquidityAggregator(
            max_levels=100,
            aggregation_mode=AggregationMode.WEIGHTED,
            stale_threshold_ms=10000,
        )
        assert agg.max_levels == 100
        assert agg.aggregation_mode == AggregationMode.WEIGHTED

    def test_update_order_book(self, aggregator):
        """Test updating order book."""
        bids = [(Decimal("50000"), Decimal("10"))]
        asks = [(Decimal("50010"), Decimal("10"))]
        aggregator.update_order_book("BTC-USD-PERP", bids, asks)

        snapshot = aggregator.get_snapshot("BTC-USD-PERP")
        assert snapshot is not None
        assert len(snapshot.bids) == 1
        assert len(snapshot.asks) == 1

    def test_get_snapshot(self, sample_book):
        """Test getting liquidity snapshot."""
        snapshot = sample_book.get_snapshot("BTC-USD-PERP")
        assert snapshot is not None
        assert snapshot.market == "BTC-USD-PERP"
        assert len(snapshot.bids) == 3
        assert len(snapshot.asks) == 3

    def test_get_snapshot_nonexistent(self, aggregator):
        """Test getting snapshot for non-existent market."""
        snapshot = aggregator.get_snapshot("UNKNOWN")
        assert snapshot is None

    def test_snapshot_totals(self, sample_book):
        """Test snapshot liquidity totals."""
        snapshot = sample_book.get_snapshot("BTC-USD-PERP")
        assert snapshot.total_bid_liquidity == Decimal("60")
        assert snapshot.total_ask_liquidity == Decimal("60")

    def test_snapshot_spread(self, sample_book):
        """Test snapshot spread calculation."""
        snapshot = sample_book.get_snapshot("BTC-USD-PERP")
        # Spread = (50010 - 50000) / 50005 * 10000 ≈ 2 bps
        assert snapshot.spread_bps > 0
        assert snapshot.spread_bps < 5

    def test_snapshot_imbalance(self, sample_book):
        """Test snapshot imbalance ratio."""
        snapshot = sample_book.get_snapshot("BTC-USD-PERP")
        # Equal bid/ask liquidity
        assert snapshot.imbalance_ratio == pytest.approx(0.5, rel=0.01)

    def test_get_metrics(self, sample_book):
        """Test getting liquidity metrics."""
        metrics = sample_book.get_metrics("BTC-USD-PERP")
        assert metrics is not None
        assert metrics.best_bid == Decimal("50000")
        assert metrics.best_ask == Decimal("50010")

    def test_get_metrics_nonexistent(self, aggregator):
        """Test getting metrics for non-existent market."""
        metrics = aggregator.get_metrics("UNKNOWN")
        assert metrics is None

    def test_metrics_mid_price(self, sample_book):
        """Test metrics mid price calculation."""
        metrics = sample_book.get_metrics("BTC-USD-PERP")
        expected_mid = (Decimal("50000") + Decimal("50010")) / 2
        assert metrics.mid_price == expected_mid

    def test_metrics_depth(self, sample_book):
        """Test metrics depth calculations."""
        metrics = sample_book.get_metrics("BTC-USD-PERP")
        # Within 1% of mid (~50005): 50005 * 0.01 = 500
        # All levels should be within 1%
        assert metrics.bid_depth_1pct > 0
        assert metrics.ask_depth_1pct > 0


class TestLiquidityAggregatorFillEstimate:
    """Tests for fill estimation."""

    @pytest.fixture
    def aggregator(self):
        """Create aggregator with order book."""
        agg = LiquidityAggregator()
        bids = [
            (Decimal("50000"), Decimal("10")),
            (Decimal("49990"), Decimal("20")),
            (Decimal("49980"), Decimal("30")),
        ]
        asks = [
            (Decimal("50010"), Decimal("10")),
            (Decimal("50020"), Decimal("20")),
            (Decimal("50030"), Decimal("30")),
        ]
        agg.update_order_book("BTC-USD-PERP", bids, asks)
        return agg

    def test_estimate_fill_buy(self, aggregator):
        """Test fill estimate for buy order."""
        estimate = aggregator.estimate_fill(
            "BTC-USD-PERP",
            LiquidityType.BID,  # Buying
            Decimal("15"),
        )
        assert estimate is not None
        assert estimate.side == LiquidityType.BID
        assert estimate.size == Decimal("15")
        assert estimate.levels_consumed == 2  # 10 + 5 from second level

    def test_estimate_fill_sell(self, aggregator):
        """Test fill estimate for sell order."""
        estimate = aggregator.estimate_fill(
            "BTC-USD-PERP",
            LiquidityType.ASK,  # Selling
            Decimal("25"),
        )
        assert estimate is not None
        assert estimate.side == LiquidityType.ASK
        assert estimate.levels_consumed == 2

    def test_estimate_fill_avg_price(self, aggregator):
        """Test average fill price calculation."""
        estimate = aggregator.estimate_fill(
            "BTC-USD-PERP",
            LiquidityType.BID,
            Decimal("10"),
        )
        # Fills at single level 50010
        assert estimate.avg_price == Decimal("50010")

    def test_estimate_fill_slippage(self, aggregator):
        """Test slippage calculation."""
        estimate = aggregator.estimate_fill(
            "BTC-USD-PERP",
            LiquidityType.BID,
            Decimal("30"),  # Fills multiple levels
        )
        assert estimate.slippage_bps > 0

    def test_estimate_fill_unfilled(self, aggregator):
        """Test partial fill when insufficient liquidity."""
        estimate = aggregator.estimate_fill(
            "BTC-USD-PERP",
            LiquidityType.BID,
            Decimal("100"),  # More than available
        )
        assert estimate.unfilled_size > 0

    def test_estimate_fill_nonexistent(self, aggregator):
        """Test fill estimate for non-existent market."""
        estimate = aggregator.estimate_fill("UNKNOWN", LiquidityType.BID, Decimal("10"))
        assert estimate is None


class TestLiquidityAggregatorDepth:
    """Tests for depth profile."""

    @pytest.fixture
    def aggregator(self):
        """Create aggregator."""
        agg = LiquidityAggregator()
        bids = [
            (Decimal("50000"), Decimal("10")),
            (Decimal("49900"), Decimal("20")),
        ]
        asks = [
            (Decimal("50100"), Decimal("10")),
            (Decimal("50200"), Decimal("20")),
        ]
        agg.update_order_book("BTC-USD-PERP", bids, asks)
        return agg

    def test_get_depth(self, aggregator):
        """Test getting depth profile."""
        depth = aggregator.get_depth("BTC-USD-PERP", num_levels=5)
        assert depth is not None
        assert len(depth.price_levels) == 11  # -5 to +5

    def test_depth_cumulative_bids(self, aggregator):
        """Test cumulative bid liquidity."""
        depth = aggregator.get_depth("BTC-USD-PERP", num_levels=5)
        # Higher prices should have less cumulative bid liquidity
        assert depth.bid_cumulative[0] >= depth.bid_cumulative[-1]

    def test_depth_cumulative_asks(self, aggregator):
        """Test cumulative ask liquidity."""
        depth = aggregator.get_depth("BTC-USD-PERP", num_levels=5)
        # Higher prices should have more cumulative ask liquidity
        assert depth.ask_cumulative[-1] >= depth.ask_cumulative[0]

    def test_get_depth_nonexistent(self, aggregator):
        """Test getting depth for non-existent market."""
        depth = aggregator.get_depth("UNKNOWN")
        assert depth is None


class TestLiquidityAggregatorPrices:
    """Tests for price queries."""

    @pytest.fixture
    def aggregator(self):
        """Create aggregator."""
        agg = LiquidityAggregator()
        bids = [(Decimal("50000"), Decimal("10"))]
        asks = [(Decimal("50010"), Decimal("10"))]
        agg.update_order_book("BTC-USD-PERP", bids, asks)
        return agg

    def test_get_best_prices(self, aggregator):
        """Test getting best prices."""
        bid, ask = aggregator.get_best_prices("BTC-USD-PERP")
        assert bid == Decimal("50000")
        assert ask == Decimal("50010")

    def test_get_best_prices_nonexistent(self, aggregator):
        """Test getting best prices for non-existent market."""
        bid, ask = aggregator.get_best_prices("UNKNOWN")
        assert bid is None
        assert ask is None

    def test_get_mid_price(self, aggregator):
        """Test getting mid price."""
        mid = aggregator.get_mid_price("BTC-USD-PERP")
        assert mid == Decimal("50005")

    def test_get_mid_price_nonexistent(self, aggregator):
        """Test getting mid price for non-existent market."""
        mid = aggregator.get_mid_price("UNKNOWN")
        assert mid is None

    def test_get_spread(self, aggregator):
        """Test getting spread."""
        spread_abs, spread_bps = aggregator.get_spread("BTC-USD-PERP")
        assert spread_abs == Decimal("10")
        assert spread_bps is not None
        assert spread_bps > 0

    def test_get_spread_nonexistent(self, aggregator):
        """Test getting spread for non-existent market."""
        spread_abs, spread_bps = aggregator.get_spread("UNKNOWN")
        assert spread_abs is None
        assert spread_bps is None

    def test_get_imbalance(self, aggregator):
        """Test getting imbalance."""
        imbalance = aggregator.get_imbalance("BTC-USD-PERP")
        assert imbalance is not None
        # Equal bid/ask
        assert imbalance == pytest.approx(0.5, rel=0.1)

    def test_get_imbalance_nonexistent(self, aggregator):
        """Test getting imbalance for non-existent market."""
        imbalance = aggregator.get_imbalance("UNKNOWN")
        assert imbalance is None


class TestLiquidityAggregatorSources:
    """Tests for multi-source aggregation."""

    @pytest.fixture
    def aggregator(self):
        """Create aggregator."""
        return LiquidityAggregator()

    def test_add_multiple_sources(self, aggregator):
        """Test adding liquidity from multiple sources."""
        # Add order book liquidity
        aggregator.update_order_book(
            "BTC-USD-PERP",
            bids=[(Decimal("50000"), Decimal("10"))],
            asks=[(Decimal("50010"), Decimal("10"))],
            source=LiquiditySource.ORDER_BOOK,
        )

        # Add market maker liquidity
        aggregator.update_order_book(
            "BTC-USD-PERP",
            bids=[(Decimal("50000"), Decimal("5"))],
            asks=[(Decimal("50010"), Decimal("5"))],
            source=LiquiditySource.MARKET_MAKER,
        )

        snapshot = aggregator.get_snapshot("BTC-USD-PERP")
        # Should have aggregated both sources at same price
        assert snapshot.total_bid_liquidity == Decimal("15")

    def test_source_priority(self, aggregator):
        """Test setting source priority."""
        aggregator.set_source_priority(LiquiditySource.DARK_POOL, 20)
        priority = aggregator.get_source_priority(LiquiditySource.DARK_POOL)
        assert priority == 20

    def test_quality_weight(self, aggregator):
        """Test setting quality weight."""
        aggregator.set_quality_weight(LiquidityQuality.HIGH, 1.5)
        weight = aggregator.get_quality_weight(LiquidityQuality.HIGH)
        assert weight == 1.5

    def test_clear_source(self, aggregator):
        """Test clearing specific source."""
        aggregator.update_order_book(
            "BTC-USD-PERP",
            bids=[(Decimal("50000"), Decimal("10"))],
            asks=[(Decimal("50010"), Decimal("10"))],
            source=LiquiditySource.ORDER_BOOK,
        )
        aggregator.update_order_book(
            "BTC-USD-PERP",
            bids=[(Decimal("50000"), Decimal("5"))],
            asks=[(Decimal("50010"), Decimal("5"))],
            source=LiquiditySource.MARKET_MAKER,
        )

        aggregator.clear_source("BTC-USD-PERP", LiquiditySource.ORDER_BOOK)
        snapshot = aggregator.get_snapshot("BTC-USD-PERP")
        # Only market maker liquidity should remain
        assert snapshot.total_bid_liquidity == Decimal("5")


class TestLiquidityAggregatorWeighted:
    """Tests for weighted aggregation mode."""

    def test_weighted_aggregation(self):
        """Test weighted aggregation mode."""
        agg = LiquidityAggregator(aggregation_mode=AggregationMode.WEIGHTED)

        # Add high quality order book
        agg.update_order_book(
            "BTC-USD-PERP",
            bids=[(Decimal("50000"), Decimal("10"))],
            asks=[(Decimal("50010"), Decimal("10"))],
            source=LiquiditySource.ORDER_BOOK,
            quality=LiquidityQuality.HIGH,
        )

        # Add low quality external
        agg.update_order_book(
            "BTC-USD-PERP",
            bids=[(Decimal("50000"), Decimal("10"))],
            asks=[(Decimal("50010"), Decimal("10"))],
            source=LiquiditySource.EXTERNAL,
            quality=LiquidityQuality.LOW,
        )

        snapshot = agg.get_snapshot("BTC-USD-PERP")
        # Weighted liquidity should be less than sum
        assert snapshot.total_bid_liquidity < Decimal("20")


class TestLiquidityAggregatorCallbacks:
    """Tests for callbacks."""

    @pytest.fixture
    def aggregator(self):
        """Create aggregator."""
        return LiquidityAggregator()

    def test_add_callback(self, aggregator):
        """Test adding callback."""
        results = []

        def callback(market, snapshot):
            results.append((market, snapshot))

        aggregator.add_callback(callback)
        aggregator.update_order_book(
            "BTC-USD-PERP",
            bids=[(Decimal("50000"), Decimal("10"))],
            asks=[(Decimal("50010"), Decimal("10"))],
        )
        # Get snapshot to trigger callback
        aggregator.get_snapshot("BTC-USD-PERP")

        assert len(results) == 1
        assert results[0][0] == "BTC-USD-PERP"

    def test_remove_callback(self, aggregator):
        """Test removing callback."""
        results = []

        def callback(market, snapshot):
            results.append(market)

        aggregator.add_callback(callback)
        removed = aggregator.remove_callback(callback)
        assert removed is True

        aggregator.update_order_book(
            "BTC-USD-PERP",
            bids=[(Decimal("50000"), Decimal("10"))],
            asks=[(Decimal("50010"), Decimal("10"))],
        )
        aggregator.get_snapshot("BTC-USD-PERP")

        assert len(results) == 0

    def test_remove_nonexistent_callback(self, aggregator):
        """Test removing non-existent callback."""
        def callback(market, snapshot):
            pass

        removed = aggregator.remove_callback(callback)
        assert removed is False


class TestLiquidityAggregatorUtilities:
    """Tests for utility methods."""

    @pytest.fixture
    def aggregator(self):
        """Create aggregator."""
        agg = LiquidityAggregator()
        agg.update_order_book(
            "BTC-USD-PERP",
            bids=[(Decimal("50000"), Decimal("10"))],
            asks=[(Decimal("50010"), Decimal("10"))],
        )
        agg.update_order_book(
            "ETH-USD-PERP",
            bids=[(Decimal("3000"), Decimal("10"))],
            asks=[(Decimal("3001"), Decimal("10"))],
        )
        return agg

    def test_get_markets(self, aggregator):
        """Test getting list of markets."""
        markets = aggregator.get_markets()
        assert "BTC-USD-PERP" in markets
        assert "ETH-USD-PERP" in markets

    def test_clear_market(self, aggregator):
        """Test clearing a market."""
        aggregator.clear_market("BTC-USD-PERP")
        snapshot = aggregator.get_snapshot("BTC-USD-PERP")
        assert snapshot is None

        # Other market should still exist
        snapshot = aggregator.get_snapshot("ETH-USD-PERP")
        assert snapshot is not None

    def test_clear_all(self, aggregator):
        """Test clearing all markets."""
        aggregator.clear_all()
        markets = aggregator.get_markets()
        assert len(markets) == 0


class TestMultiMarketAggregator:
    """Tests for MultiMarketAggregator class."""

    @pytest.fixture
    def multi_agg(self):
        """Create multi-market aggregator."""
        return MultiMarketAggregator()

    def test_init(self, multi_agg):
        """Test initialization."""
        assert multi_agg.aggregator is not None

    def test_init_with_aggregator(self):
        """Test initialization with existing aggregator."""
        agg = LiquidityAggregator()
        multi = MultiMarketAggregator(agg)
        assert multi.aggregator is agg

    def test_update_all_markets(self, multi_agg):
        """Test updating all markets."""
        data = {
            "BTC-USD-PERP": (
                [(Decimal("50000"), Decimal("10"))],
                [(Decimal("50010"), Decimal("10"))],
            ),
            "ETH-USD-PERP": (
                [(Decimal("3000"), Decimal("10"))],
                [(Decimal("3001"), Decimal("10"))],
            ),
        }

        snapshots = multi_agg.update_all_markets(data)
        assert "BTC-USD-PERP" in snapshots
        assert "ETH-USD-PERP" in snapshots

    def test_get_all_metrics(self, multi_agg):
        """Test getting metrics for all markets."""
        data = {
            "BTC-USD-PERP": (
                [(Decimal("50000"), Decimal("10"))],
                [(Decimal("50010"), Decimal("10"))],
            ),
        }
        multi_agg.update_all_markets(data)

        metrics = multi_agg.get_all_metrics()
        assert "BTC-USD-PERP" in metrics

    def test_get_best_liquidity_market(self, multi_agg):
        """Test finding best liquidity market."""
        data = {
            "BTC-USD-PERP": (
                [(Decimal("50000"), Decimal("100"))],
                [(Decimal("50010"), Decimal("100"))],
            ),
            "ETH-USD-PERP": (
                [(Decimal("3000"), Decimal("10"))],
                [(Decimal("3001"), Decimal("10"))],
            ),
        }
        multi_agg.update_all_markets(data)

        best = multi_agg.get_best_liquidity_market()
        assert best == "BTC-USD-PERP"

    def test_get_best_liquidity_market_by_side(self, multi_agg):
        """Test finding best liquidity by side."""
        data = {
            "BTC-USD-PERP": (
                [(Decimal("50000"), Decimal("100"))],
                [(Decimal("50010"), Decimal("10"))],
            ),
            "ETH-USD-PERP": (
                [(Decimal("3000"), Decimal("10"))],
                [(Decimal("3001"), Decimal("100"))],
            ),
        }
        multi_agg.update_all_markets(data)

        best_bid = multi_agg.get_best_liquidity_market(side=LiquidityType.BID)
        assert best_bid == "BTC-USD-PERP"

        best_ask = multi_agg.get_best_liquidity_market(side=LiquidityType.ASK)
        assert best_ask == "ETH-USD-PERP"

    def test_get_tightest_spread_market(self, multi_agg):
        """Test finding market with tightest spread."""
        data = {
            "BTC-USD-PERP": (
                [(Decimal("50000"), Decimal("10"))],
                [(Decimal("50100"), Decimal("10"))],  # Wide spread
            ),
            "ETH-USD-PERP": (
                [(Decimal("3000"), Decimal("10"))],
                [(Decimal("3001"), Decimal("10"))],  # Tight spread
            ),
        }
        multi_agg.update_all_markets(data)

        tightest = multi_agg.get_tightest_spread_market()
        assert tightest == "ETH-USD-PERP"

    def test_get_cross_market_imbalance(self, multi_agg):
        """Test cross-market imbalance."""
        data = {
            "BTC-USD-PERP": (
                [(Decimal("50000"), Decimal("100"))],
                [(Decimal("50010"), Decimal("100"))],
            ),
            "ETH-USD-PERP": (
                [(Decimal("3000"), Decimal("100"))],
                [(Decimal("3001"), Decimal("100"))],
            ),
        }
        multi_agg.update_all_markets(data)

        imbalance = multi_agg.get_cross_market_imbalance()
        assert imbalance == pytest.approx(0.5, rel=0.01)

    def test_cross_market_callback(self, multi_agg):
        """Test cross-market callback."""
        results = []

        def callback(snapshots):
            results.append(snapshots)

        multi_agg.add_cross_market_callback(callback)

        data = {
            "BTC-USD-PERP": (
                [(Decimal("50000"), Decimal("10"))],
                [(Decimal("50010"), Decimal("10"))],
            ),
        }
        multi_agg.update_all_markets(data)

        assert len(results) == 1
        assert "BTC-USD-PERP" in results[0]

    def test_remove_cross_market_callback(self, multi_agg):
        """Test removing cross-market callback."""
        def callback(snapshots):
            pass

        multi_agg.add_cross_market_callback(callback)
        removed = multi_agg.remove_cross_market_callback(callback)
        assert removed is True

    def test_remove_nonexistent_cross_market_callback(self, multi_agg):
        """Test removing non-existent callback."""
        def callback(snapshots):
            pass

        removed = multi_agg.remove_cross_market_callback(callback)
        assert removed is False


class TestGlobalFunctions:
    """Tests for global functions."""

    def test_get_aggregator(self):
        """Test getting global aggregator."""
        reset_aggregators()
        agg = get_aggregator()
        assert agg is not None
        assert isinstance(agg, LiquidityAggregator)

    def test_get_aggregator_singleton(self):
        """Test aggregator is singleton."""
        reset_aggregators()
        agg1 = get_aggregator()
        agg2 = get_aggregator()
        assert agg1 is agg2

    def test_get_multi_aggregator(self):
        """Test getting global multi-market aggregator."""
        reset_aggregators()
        multi = get_multi_aggregator()
        assert multi is not None
        assert isinstance(multi, MultiMarketAggregator)

    def test_get_multi_aggregator_singleton(self):
        """Test multi-aggregator is singleton."""
        reset_aggregators()
        multi1 = get_multi_aggregator()
        multi2 = get_multi_aggregator()
        assert multi1 is multi2

    def test_reset_aggregators(self):
        """Test resetting aggregators."""
        agg1 = get_aggregator()
        reset_aggregators()
        agg2 = get_aggregator()
        assert agg1 is not agg2


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def aggregator(self):
        """Create aggregator."""
        return LiquidityAggregator()

    def test_empty_order_book(self, aggregator):
        """Test empty order book."""
        aggregator.update_order_book("BTC-USD-PERP", [], [])
        snapshot = aggregator.get_snapshot("BTC-USD-PERP")
        assert snapshot is not None
        assert len(snapshot.bids) == 0
        assert len(snapshot.asks) == 0

    def test_single_level(self, aggregator):
        """Test single level order book."""
        aggregator.update_order_book(
            "BTC-USD-PERP",
            bids=[(Decimal("50000"), Decimal("10"))],
            asks=[(Decimal("50010"), Decimal("10"))],
        )
        snapshot = aggregator.get_snapshot("BTC-USD-PERP")
        assert len(snapshot.bids) == 1
        assert len(snapshot.asks) == 1

    def test_large_order_book(self, aggregator):
        """Test large order book."""
        bids = [(Decimal(str(50000 - i)), Decimal("10")) for i in range(100)]
        asks = [(Decimal(str(50010 + i)), Decimal("10")) for i in range(100)]
        aggregator.update_order_book("BTC-USD-PERP", bids, asks)

        snapshot = aggregator.get_snapshot("BTC-USD-PERP")
        # Should be limited to max_levels
        assert len(snapshot.bids) <= aggregator.max_levels
        assert len(snapshot.asks) <= aggregator.max_levels

    def test_zero_liquidity(self, aggregator):
        """Test zero liquidity level."""
        aggregator.update_order_book(
            "BTC-USD-PERP",
            bids=[(Decimal("50000"), Decimal("0"))],
            asks=[(Decimal("50010"), Decimal("0"))],
        )
        snapshot = aggregator.get_snapshot("BTC-USD-PERP")
        assert snapshot.total_bid_liquidity == Decimal("0")

    def test_same_price_different_sources(self, aggregator):
        """Test same price from different sources."""
        aggregator.update_order_book(
            "BTC-USD-PERP",
            bids=[(Decimal("50000"), Decimal("10"))],
            asks=[(Decimal("50010"), Decimal("10"))],
            source=LiquiditySource.ORDER_BOOK,
        )
        aggregator.update_order_book(
            "BTC-USD-PERP",
            bids=[(Decimal("50000"), Decimal("5"))],
            asks=[(Decimal("50010"), Decimal("5"))],
            source=LiquiditySource.MARKET_MAKER,
        )

        snapshot = aggregator.get_snapshot("BTC-USD-PERP")
        # Should aggregate at same price
        assert snapshot.bids[0].total_size == Decimal("15")

    def test_very_small_prices(self, aggregator):
        """Test very small prices."""
        aggregator.update_order_book(
            "SMALL-USD-PERP",
            bids=[(Decimal("0.000001"), Decimal("1000000"))],
            asks=[(Decimal("0.000002"), Decimal("1000000"))],
        )
        snapshot = aggregator.get_snapshot("SMALL-USD-PERP")
        assert snapshot is not None
        assert snapshot.bids[0].price == Decimal("0.000001")

    def test_very_large_prices(self, aggregator):
        """Test very large prices."""
        aggregator.update_order_book(
            "LARGE-USD-PERP",
            bids=[(Decimal("1000000000"), Decimal("0.001"))],
            asks=[(Decimal("1000000001"), Decimal("0.001"))],
        )
        snapshot = aggregator.get_snapshot("LARGE-USD-PERP")
        assert snapshot is not None


class TestAddLiquidity:
    """Tests for add_liquidity method."""

    @pytest.fixture
    def aggregator(self):
        """Create aggregator."""
        return LiquidityAggregator()

    def test_add_bid_liquidity(self, aggregator):
        """Test adding bid liquidity."""
        levels = [
            LiquidityLevel(
                price=Decimal("50000"),
                size=Decimal("10"),
                source=LiquiditySource.ORDER_BOOK,
            )
        ]
        aggregator.add_liquidity("BTC-USD-PERP", levels, LiquidityType.BID)

        snapshot = aggregator.get_snapshot("BTC-USD-PERP")
        # Only bids, no asks
        assert len(snapshot.bids) == 1
        assert len(snapshot.asks) == 0

    def test_add_ask_liquidity(self, aggregator):
        """Test adding ask liquidity."""
        levels = [
            LiquidityLevel(
                price=Decimal("50010"),
                size=Decimal("10"),
                source=LiquiditySource.ORDER_BOOK,
            )
        ]
        aggregator.add_liquidity("BTC-USD-PERP", levels, LiquidityType.ASK)

        snapshot = aggregator.get_snapshot("BTC-USD-PERP")
        assert len(snapshot.bids) == 0
        assert len(snapshot.asks) == 1

    def test_add_both_liquidity(self, aggregator):
        """Test adding liquidity to both sides."""
        levels = [
            LiquidityLevel(
                price=Decimal("50000"),
                size=Decimal("10"),
                source=LiquiditySource.ORDER_BOOK,
            )
        ]
        aggregator.add_liquidity("BTC-USD-PERP", levels, LiquidityType.BOTH)

        snapshot = aggregator.get_snapshot("BTC-USD-PERP")
        # Added to both sides
        assert len(snapshot.bids) == 1
        assert len(snapshot.asks) == 1
