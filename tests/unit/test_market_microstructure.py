"""
Tests for Market Microstructure Analysis Module.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from src.analytics.market_microstructure import (
    LiquidityState,
    MarketMakerActivity,
    FlowToxicity,
    SpreadRegime,
    SpreadMetrics,
    DepthMetrics,
    TradeImbalance,
    QuoteActivity,
    PriceImpact,
    MicrostructureSnapshot,
    QuoteRecord,
    TradeRecord,
    SpreadAnalyzer,
    DepthAnalyzer,
    TradeFlowAnalyzer,
    QuoteAnalyzer,
    PriceImpactEstimator,
    MicrostructureAnalyzer,
)


# =============================================================================
# Enum Tests
# =============================================================================

class TestLiquidityState:
    """Tests for LiquidityState enum."""

    def test_all_states_defined(self):
        """Test all liquidity states are defined."""
        assert LiquidityState.HIGHLY_LIQUID.value == "highly_liquid"
        assert LiquidityState.LIQUID.value == "liquid"
        assert LiquidityState.NORMAL.value == "normal"
        assert LiquidityState.ILLIQUID.value == "illiquid"
        assert LiquidityState.CRITICALLY_ILLIQUID.value == "critically_illiquid"

    def test_state_count(self):
        """Test correct number of states."""
        assert len(LiquidityState) == 5


class TestMarketMakerActivity:
    """Tests for MarketMakerActivity enum."""

    def test_all_activities_defined(self):
        """Test all activity levels are defined."""
        assert MarketMakerActivity.VERY_ACTIVE.value == "very_active"
        assert MarketMakerActivity.ACTIVE.value == "active"
        assert MarketMakerActivity.NORMAL.value == "normal"
        assert MarketMakerActivity.REDUCED.value == "reduced"
        assert MarketMakerActivity.ABSENT.value == "absent"


class TestFlowToxicity:
    """Tests for FlowToxicity enum."""

    def test_all_toxicity_levels_defined(self):
        """Test all toxicity levels are defined."""
        assert FlowToxicity.VERY_LOW.value == "very_low"
        assert FlowToxicity.LOW.value == "low"
        assert FlowToxicity.MODERATE.value == "moderate"
        assert FlowToxicity.HIGH.value == "high"
        assert FlowToxicity.VERY_HIGH.value == "very_high"


class TestSpreadRegime:
    """Tests for SpreadRegime enum."""

    def test_all_regimes_defined(self):
        """Test all spread regimes are defined."""
        assert SpreadRegime.TIGHT.value == "tight"
        assert SpreadRegime.NORMAL.value == "normal"
        assert SpreadRegime.WIDE.value == "wide"
        assert SpreadRegime.VERY_WIDE.value == "very_wide"


# =============================================================================
# Data Class Tests
# =============================================================================

class TestSpreadMetrics:
    """Tests for SpreadMetrics dataclass."""

    def test_creation(self):
        """Test SpreadMetrics creation."""
        metrics = SpreadMetrics(
            symbol="BTC-USD-PERP",
            timestamp=datetime.now(),
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            spread_absolute=Decimal("10"),
            spread_bps=Decimal("2"),
            mid_price=Decimal("50005"),
            spread_regime=SpreadRegime.TIGHT
        )
        assert metrics.symbol == "BTC-USD-PERP"
        assert metrics.spread_absolute == Decimal("10")
        assert metrics.is_crossed is False

    def test_crossed_market(self):
        """Test crossed market detection."""
        metrics = SpreadMetrics(
            symbol="BTC-USD-PERP",
            timestamp=datetime.now(),
            bid_price=Decimal("50010"),
            ask_price=Decimal("50000"),
            spread_absolute=Decimal("-10"),
            spread_bps=Decimal("-2"),
            mid_price=Decimal("50005"),
            spread_regime=SpreadRegime.VERY_WIDE,
            is_crossed=True
        )
        assert metrics.is_crossed is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        now = datetime.now()
        metrics = SpreadMetrics(
            symbol="BTC-USD-PERP",
            timestamp=now,
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            spread_absolute=Decimal("10"),
            spread_bps=Decimal("2"),
            mid_price=Decimal("50005"),
            spread_regime=SpreadRegime.TIGHT
        )
        result = metrics.to_dict()
        assert result["symbol"] == "BTC-USD-PERP"
        assert result["spread_regime"] == "tight"
        assert result["timestamp"] == now.isoformat()


class TestDepthMetrics:
    """Tests for DepthMetrics dataclass."""

    def test_creation(self):
        """Test DepthMetrics creation."""
        metrics = DepthMetrics(
            symbol="BTC-USD-PERP",
            timestamp=datetime.now(),
            bid_depth_levels=20,
            ask_depth_levels=20,
            total_bid_volume=Decimal("1000000"),
            total_ask_volume=Decimal("900000"),
            bid_volume_1pct=Decimal("500000"),
            ask_volume_1pct=Decimal("450000"),
            bid_volume_5pct=Decimal("800000"),
            ask_volume_5pct=Decimal("750000"),
            depth_imbalance=Decimal("0.053"),
            liquidity_state=LiquidityState.HIGHLY_LIQUID
        )
        assert metrics.bid_depth_levels == 20
        assert metrics.liquidity_state == LiquidityState.HIGHLY_LIQUID

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = DepthMetrics(
            symbol="BTC-USD-PERP",
            timestamp=datetime.now(),
            bid_depth_levels=20,
            ask_depth_levels=20,
            total_bid_volume=Decimal("1000000"),
            total_ask_volume=Decimal("900000"),
            bid_volume_1pct=Decimal("500000"),
            ask_volume_1pct=Decimal("450000"),
            bid_volume_5pct=Decimal("800000"),
            ask_volume_5pct=Decimal("750000"),
            depth_imbalance=Decimal("0.053"),
            liquidity_state=LiquidityState.LIQUID
        )
        result = metrics.to_dict()
        assert result["liquidity_state"] == "liquid"
        assert result["bid_depth_levels"] == 20


class TestTradeImbalance:
    """Tests for TradeImbalance dataclass."""

    def test_creation(self):
        """Test TradeImbalance creation."""
        imbalance = TradeImbalance(
            symbol="BTC-USD-PERP",
            timestamp=datetime.now(),
            window_seconds=60,
            buy_volume=Decimal("100000"),
            sell_volume=Decimal("80000"),
            buy_trades=50,
            sell_trades=40,
            net_volume=Decimal("20000"),
            imbalance_ratio=Decimal("0.111"),
            vwap_buy=Decimal("50100"),
            vwap_sell=Decimal("49900"),
            flow_toxicity=FlowToxicity.LOW
        )
        assert imbalance.buy_trades == 50
        assert imbalance.flow_toxicity == FlowToxicity.LOW

    def test_to_dict(self):
        """Test conversion to dictionary."""
        imbalance = TradeImbalance(
            symbol="BTC-USD-PERP",
            timestamp=datetime.now(),
            window_seconds=60,
            buy_volume=Decimal("100000"),
            sell_volume=Decimal("80000"),
            buy_trades=50,
            sell_trades=40,
            net_volume=Decimal("20000"),
            imbalance_ratio=Decimal("0.111"),
            vwap_buy=Decimal("50100"),
            vwap_sell=Decimal("49900"),
            flow_toxicity=FlowToxicity.HIGH
        )
        result = imbalance.to_dict()
        assert result["flow_toxicity"] == "high"
        assert result["window_seconds"] == 60


class TestQuoteActivity:
    """Tests for QuoteActivity dataclass."""

    def test_creation(self):
        """Test QuoteActivity creation."""
        activity = QuoteActivity(
            symbol="BTC-USD-PERP",
            timestamp=datetime.now(),
            window_seconds=60,
            quote_updates=500,
            quote_rate_per_second=Decimal("8.33"),
            bid_updates=250,
            ask_updates=250,
            quote_to_trade_ratio=Decimal("5.5"),
            average_quote_lifetime_ms=Decimal("120"),
            mm_activity=MarketMakerActivity.ACTIVE
        )
        assert activity.quote_updates == 500
        assert activity.mm_activity == MarketMakerActivity.ACTIVE

    def test_to_dict(self):
        """Test conversion to dictionary."""
        activity = QuoteActivity(
            symbol="BTC-USD-PERP",
            timestamp=datetime.now(),
            window_seconds=60,
            quote_updates=500,
            quote_rate_per_second=Decimal("8.33"),
            bid_updates=250,
            ask_updates=250,
            quote_to_trade_ratio=Decimal("5.5"),
            average_quote_lifetime_ms=Decimal("120"),
            mm_activity=MarketMakerActivity.VERY_ACTIVE
        )
        result = activity.to_dict()
        assert result["mm_activity"] == "very_active"


class TestPriceImpact:
    """Tests for PriceImpact dataclass."""

    def test_creation(self):
        """Test PriceImpact creation."""
        impact = PriceImpact(
            symbol="BTC-USD-PERP",
            timestamp=datetime.now(),
            order_size=Decimal("10"),
            side="buy",
            estimated_impact_bps=Decimal("5"),
            estimated_slippage=Decimal("25"),
            effective_spread_bps=Decimal("7"),
            market_impact_cost=Decimal("350")
        )
        assert impact.side == "buy"
        assert impact.estimated_impact_bps == Decimal("5")

    def test_to_dict(self):
        """Test conversion to dictionary."""
        impact = PriceImpact(
            symbol="BTC-USD-PERP",
            timestamp=datetime.now(),
            order_size=Decimal("10"),
            side="sell",
            estimated_impact_bps=Decimal("5"),
            estimated_slippage=Decimal("25"),
            effective_spread_bps=Decimal("7"),
            market_impact_cost=Decimal("350")
        )
        result = impact.to_dict()
        assert result["side"] == "sell"


class TestMicrostructureSnapshot:
    """Tests for MicrostructureSnapshot dataclass."""

    def test_creation(self):
        """Test MicrostructureSnapshot creation."""
        now = datetime.now()
        spread = SpreadMetrics(
            symbol="BTC-USD-PERP",
            timestamp=now,
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            spread_absolute=Decimal("10"),
            spread_bps=Decimal("2"),
            mid_price=Decimal("50005"),
            spread_regime=SpreadRegime.TIGHT
        )
        depth = DepthMetrics(
            symbol="BTC-USD-PERP",
            timestamp=now,
            bid_depth_levels=20,
            ask_depth_levels=20,
            total_bid_volume=Decimal("1000000"),
            total_ask_volume=Decimal("900000"),
            bid_volume_1pct=Decimal("500000"),
            ask_volume_1pct=Decimal("450000"),
            bid_volume_5pct=Decimal("800000"),
            ask_volume_5pct=Decimal("750000"),
            depth_imbalance=Decimal("0.053"),
            liquidity_state=LiquidityState.HIGHLY_LIQUID
        )
        snapshot = MicrostructureSnapshot(
            symbol="BTC-USD-PERP",
            timestamp=now,
            spread_metrics=spread,
            depth_metrics=depth,
            trade_imbalance=None,
            quote_activity=None,
            price_impact_buy=None,
            price_impact_sell=None,
            overall_quality_score=Decimal("85")
        )
        assert snapshot.overall_quality_score == Decimal("85")

    def test_to_dict(self):
        """Test conversion to dictionary."""
        now = datetime.now()
        spread = SpreadMetrics(
            symbol="BTC-USD-PERP",
            timestamp=now,
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            spread_absolute=Decimal("10"),
            spread_bps=Decimal("2"),
            mid_price=Decimal("50005"),
            spread_regime=SpreadRegime.TIGHT
        )
        depth = DepthMetrics(
            symbol="BTC-USD-PERP",
            timestamp=now,
            bid_depth_levels=20,
            ask_depth_levels=20,
            total_bid_volume=Decimal("1000000"),
            total_ask_volume=Decimal("900000"),
            bid_volume_1pct=Decimal("500000"),
            ask_volume_1pct=Decimal("450000"),
            bid_volume_5pct=Decimal("800000"),
            ask_volume_5pct=Decimal("750000"),
            depth_imbalance=Decimal("0.053"),
            liquidity_state=LiquidityState.LIQUID
        )
        snapshot = MicrostructureSnapshot(
            symbol="BTC-USD-PERP",
            timestamp=now,
            spread_metrics=spread,
            depth_metrics=depth,
            trade_imbalance=None,
            quote_activity=None,
            price_impact_buy=None,
            price_impact_sell=None,
            overall_quality_score=Decimal("75")
        )
        result = snapshot.to_dict()
        assert result["overall_quality_score"] == "75"
        assert result["trade_imbalance"] is None


class TestQuoteRecord:
    """Tests for QuoteRecord dataclass."""

    def test_creation(self):
        """Test QuoteRecord creation."""
        record = QuoteRecord(
            timestamp=datetime.now(),
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            bid_size=Decimal("5"),
            ask_size=Decimal("4")
        )
        assert record.bid_price == Decimal("50000")


class TestTradeRecord:
    """Tests for TradeRecord dataclass."""

    def test_creation(self):
        """Test TradeRecord creation."""
        record = TradeRecord(
            timestamp=datetime.now(),
            price=Decimal("50000"),
            size=Decimal("1.5"),
            side="buy"
        )
        assert record.side == "buy"


# =============================================================================
# Spread Analyzer Tests
# =============================================================================

class TestSpreadAnalyzer:
    """Tests for SpreadAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return SpreadAnalyzer(
            tight_threshold_bps=Decimal("5"),
            wide_threshold_bps=Decimal("20"),
            very_wide_threshold_bps=Decimal("50")
        )

    def test_analyze_tight_spread(self, analyzer):
        """Test tight spread detection."""
        metrics = analyzer.analyze_spread(
            symbol="BTC-USD-PERP",
            bid_price=Decimal("50000"),
            ask_price=Decimal("50002")  # 0.4 bps
        )
        assert metrics.spread_regime == SpreadRegime.TIGHT
        assert metrics.spread_bps < Decimal("5")

    def test_analyze_normal_spread(self, analyzer):
        """Test normal spread detection."""
        metrics = analyzer.analyze_spread(
            symbol="BTC-USD-PERP",
            bid_price=Decimal("50000"),
            ask_price=Decimal("50050")  # 10 bps
        )
        assert metrics.spread_regime == SpreadRegime.NORMAL

    def test_analyze_wide_spread(self, analyzer):
        """Test wide spread detection."""
        metrics = analyzer.analyze_spread(
            symbol="BTC-USD-PERP",
            bid_price=Decimal("50000"),
            ask_price=Decimal("50150")  # 30 bps
        )
        assert metrics.spread_regime == SpreadRegime.WIDE

    def test_analyze_very_wide_spread(self, analyzer):
        """Test very wide spread detection."""
        metrics = analyzer.analyze_spread(
            symbol="BTC-USD-PERP",
            bid_price=Decimal("50000"),
            ask_price=Decimal("50300")  # 60 bps
        )
        assert metrics.spread_regime == SpreadRegime.VERY_WIDE

    def test_crossed_market_detection(self, analyzer):
        """Test crossed market detection."""
        metrics = analyzer.analyze_spread(
            symbol="BTC-USD-PERP",
            bid_price=Decimal("50010"),
            ask_price=Decimal("50000")
        )
        assert metrics.is_crossed is True
        assert metrics.spread_regime == SpreadRegime.VERY_WIDE

    def test_mid_price_calculation(self, analyzer):
        """Test mid price calculation."""
        metrics = analyzer.analyze_spread(
            symbol="BTC-USD-PERP",
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010")
        )
        assert metrics.mid_price == Decimal("50005")

    def test_spread_history_storage(self, analyzer):
        """Test spread history is stored."""
        for i in range(5):
            analyzer.analyze_spread(
                symbol="BTC-USD-PERP",
                bid_price=Decimal("50000") + i,
                ask_price=Decimal("50010") + i
            )
        assert len(analyzer._spread_history["BTC-USD-PERP"]) == 5

    def test_get_average_spread(self, analyzer):
        """Test average spread calculation."""
        now = datetime.now()
        for i in range(10):
            analyzer.analyze_spread(
                symbol="BTC-USD-PERP",
                bid_price=Decimal("50000"),
                ask_price=Decimal("50010"),
                timestamp=now - timedelta(seconds=i)
            )
        avg = analyzer.get_average_spread("BTC-USD-PERP", window_seconds=60)
        assert avg is not None
        assert avg > Decimal("0")

    def test_get_average_spread_no_data(self, analyzer):
        """Test average spread with no data."""
        avg = analyzer.get_average_spread("UNKNOWN", window_seconds=60)
        assert avg is None

    def test_get_spread_volatility(self, analyzer):
        """Test spread volatility calculation."""
        now = datetime.now()
        for i in range(10):
            spread = Decimal("10") + (i % 3)  # Varying spread
            analyzer.analyze_spread(
                symbol="BTC-USD-PERP",
                bid_price=Decimal("50000"),
                ask_price=Decimal("50000") + spread,
                timestamp=now - timedelta(seconds=i)
            )
        vol = analyzer.get_spread_volatility("BTC-USD-PERP", window_seconds=60)
        assert vol is not None

    def test_spread_volatility_no_data(self, analyzer):
        """Test spread volatility with no data."""
        vol = analyzer.get_spread_volatility("UNKNOWN", window_seconds=60)
        assert vol is None

    def test_history_limit(self, analyzer):
        """Test history is limited."""
        analyzer._max_history = 100
        for i in range(150):
            analyzer.analyze_spread(
                symbol="BTC-USD-PERP",
                bid_price=Decimal("50000"),
                ask_price=Decimal("50010")
            )
        assert len(analyzer._spread_history["BTC-USD-PERP"]) == 100


# =============================================================================
# Depth Analyzer Tests
# =============================================================================

class TestDepthAnalyzer:
    """Tests for DepthAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return DepthAnalyzer(
            highly_liquid_threshold=Decimal("1000000"),
            liquid_threshold=Decimal("500000"),
            illiquid_threshold=Decimal("100000"),
            critical_threshold=Decimal("10000")
        )

    @pytest.fixture
    def sample_orderbook(self):
        """Create sample order book."""
        mid = Decimal("50000")
        bids = [
            (mid - Decimal("10") * i, Decimal("10")) for i in range(1, 21)
        ]
        asks = [
            (mid + Decimal("10") * i, Decimal("10")) for i in range(1, 21)
        ]
        return bids, asks

    def test_analyze_depth_highly_liquid(self, analyzer):
        """Test highly liquid market detection."""
        # Create very liquid book
        bids = [(Decimal("50000") - i * 10, Decimal("1000")) for i in range(20)]
        asks = [(Decimal("50010") + i * 10, Decimal("1000")) for i in range(20)]

        metrics = analyzer.analyze_depth(
            symbol="BTC-USD-PERP",
            bids=bids,
            asks=asks,
            mid_price=Decimal("50005")
        )
        assert metrics.liquidity_state == LiquidityState.HIGHLY_LIQUID

    def test_analyze_depth_normal(self, analyzer):
        """Test normal liquidity detection."""
        # Volume should be between 100k and 500k for NORMAL state
        # 0.2 size * 10 levels * ~50k price = ~100k per side = 200k total
        bids = [(Decimal("50000") - i * 10, Decimal("0.2")) for i in range(10)]
        asks = [(Decimal("50010") + i * 10, Decimal("0.2")) for i in range(10)]

        metrics = analyzer.analyze_depth(
            symbol="BTC-USD-PERP",
            bids=bids,
            asks=asks,
            mid_price=Decimal("50005")
        )
        assert metrics.liquidity_state == LiquidityState.NORMAL

    def test_analyze_depth_illiquid(self, analyzer):
        """Test illiquid market detection."""
        # Volume should be between 10k and 100k for ILLIQUID state
        # 0.02 size * 10 levels * ~50k price = ~10k per side = 20k total
        bids = [(Decimal("50000") - i * 10, Decimal("0.02")) for i in range(10)]
        asks = [(Decimal("50010") + i * 10, Decimal("0.02")) for i in range(10)]

        metrics = analyzer.analyze_depth(
            symbol="BTC-USD-PERP",
            bids=bids,
            asks=asks,
            mid_price=Decimal("50005")
        )
        assert metrics.liquidity_state == LiquidityState.ILLIQUID

    def test_analyze_depth_critically_illiquid(self, analyzer):
        """Test critically illiquid detection."""
        bids = [(Decimal("50000"), Decimal("0.01"))]
        asks = [(Decimal("50010"), Decimal("0.01"))]

        metrics = analyzer.analyze_depth(
            symbol="BTC-USD-PERP",
            bids=bids,
            asks=asks,
            mid_price=Decimal("50005")
        )
        assert metrics.liquidity_state == LiquidityState.CRITICALLY_ILLIQUID

    def test_depth_imbalance_calculation(self, analyzer):
        """Test depth imbalance calculation."""
        bids = [(Decimal("50000"), Decimal("100"))]  # 5M notional
        asks = [(Decimal("50010"), Decimal("50"))]   # 2.5M notional

        metrics = analyzer.analyze_depth(
            symbol="BTC-USD-PERP",
            bids=bids,
            asks=asks,
            mid_price=Decimal("50005")
        )
        # More bids than asks = positive imbalance
        assert metrics.depth_imbalance > Decimal("0")

    def test_volume_by_price_range(self, analyzer, sample_orderbook):
        """Test volume calculation by price range."""
        bids, asks = sample_orderbook
        metrics = analyzer.analyze_depth(
            symbol="BTC-USD-PERP",
            bids=bids,
            asks=asks,
            mid_price=Decimal("50000")
        )
        # Volume within 1% should be less than 5%
        assert metrics.bid_volume_1pct <= metrics.bid_volume_5pct
        assert metrics.ask_volume_1pct <= metrics.ask_volume_5pct

    def test_depth_levels_count(self, analyzer, sample_orderbook):
        """Test depth levels counting."""
        bids, asks = sample_orderbook
        metrics = analyzer.analyze_depth(
            symbol="BTC-USD-PERP",
            bids=bids,
            asks=asks
        )
        assert metrics.bid_depth_levels == 20
        assert metrics.ask_depth_levels == 20

    def test_empty_orderbook(self, analyzer):
        """Test with empty order book."""
        metrics = analyzer.analyze_depth(
            symbol="BTC-USD-PERP",
            bids=[],
            asks=[]
        )
        assert metrics.bid_depth_levels == 0
        assert metrics.ask_depth_levels == 0
        assert metrics.liquidity_state == LiquidityState.CRITICALLY_ILLIQUID


# =============================================================================
# Trade Flow Analyzer Tests
# =============================================================================

class TestTradeFlowAnalyzer:
    """Tests for TradeFlowAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return TradeFlowAnalyzer(
            high_toxicity_threshold=Decimal("0.7"),
            moderate_toxicity_threshold=Decimal("0.4")
        )

    def test_record_trade(self, analyzer):
        """Test trade recording."""
        analyzer.record_trade(
            symbol="BTC-USD-PERP",
            price=Decimal("50000"),
            size=Decimal("1"),
            side="buy"
        )
        assert len(analyzer._trade_history["BTC-USD-PERP"]) == 1

    def test_analyze_balanced_imbalance(self, analyzer):
        """Test balanced trade flow."""
        now = datetime.now()
        for i in range(10):
            analyzer.record_trade(
                symbol="BTC-USD-PERP",
                price=Decimal("50000"),
                size=Decimal("1"),
                side="buy",
                timestamp=now - timedelta(seconds=i)
            )
            analyzer.record_trade(
                symbol="BTC-USD-PERP",
                price=Decimal("50000"),
                size=Decimal("1"),
                side="sell",
                timestamp=now - timedelta(seconds=i)
            )

        imbalance = analyzer.analyze_imbalance(
            symbol="BTC-USD-PERP",
            window_seconds=60,
            timestamp=now
        )
        assert imbalance is not None
        assert abs(imbalance.imbalance_ratio) < Decimal("0.1")
        assert imbalance.flow_toxicity == FlowToxicity.VERY_LOW

    def test_analyze_buy_dominated(self, analyzer):
        """Test buy-dominated flow."""
        now = datetime.now()
        for i in range(10):
            analyzer.record_trade(
                symbol="BTC-USD-PERP",
                price=Decimal("50000"),
                size=Decimal("2"),
                side="buy",
                timestamp=now - timedelta(seconds=i)
            )
            analyzer.record_trade(
                symbol="BTC-USD-PERP",
                price=Decimal("50000"),
                size=Decimal("0.5"),
                side="sell",
                timestamp=now - timedelta(seconds=i)
            )

        imbalance = analyzer.analyze_imbalance(
            symbol="BTC-USD-PERP",
            window_seconds=60,
            timestamp=now
        )
        assert imbalance is not None
        assert imbalance.imbalance_ratio > Decimal("0")
        assert imbalance.buy_volume > imbalance.sell_volume

    def test_analyze_very_high_toxicity(self, analyzer):
        """Test very high toxicity detection."""
        now = datetime.now()
        for i in range(10):
            analyzer.record_trade(
                symbol="BTC-USD-PERP",
                price=Decimal("50000"),
                size=Decimal("10"),
                side="buy",
                timestamp=now - timedelta(seconds=i)
            )

        imbalance = analyzer.analyze_imbalance(
            symbol="BTC-USD-PERP",
            window_seconds=60,
            timestamp=now
        )
        assert imbalance is not None
        assert imbalance.flow_toxicity == FlowToxicity.VERY_HIGH

    def test_vwap_calculation(self, analyzer):
        """Test VWAP calculation."""
        now = datetime.now()
        analyzer.record_trade(
            symbol="BTC-USD-PERP",
            price=Decimal("50000"),
            size=Decimal("1"),
            side="buy",
            timestamp=now
        )
        analyzer.record_trade(
            symbol="BTC-USD-PERP",
            price=Decimal("50100"),
            size=Decimal("1"),
            side="buy",
            timestamp=now
        )
        analyzer.record_trade(
            symbol="BTC-USD-PERP",
            price=Decimal("49900"),
            size=Decimal("1"),
            side="sell",
            timestamp=now
        )

        imbalance = analyzer.analyze_imbalance(
            symbol="BTC-USD-PERP",
            window_seconds=60,
            timestamp=now
        )
        assert imbalance is not None
        assert imbalance.vwap_buy == Decimal("50050")  # Average of 50000 and 50100
        assert imbalance.vwap_sell == Decimal("49900")

    def test_no_data_returns_none(self, analyzer):
        """Test no data returns None."""
        imbalance = analyzer.analyze_imbalance(
            symbol="UNKNOWN",
            window_seconds=60
        )
        assert imbalance is None

    def test_vpin_calculation(self, analyzer):
        """Test VPIN calculation."""
        now = datetime.now()
        for i in range(100):
            side = "buy" if i % 2 == 0 else "sell"
            analyzer.record_trade(
                symbol="BTC-USD-PERP",
                price=Decimal("50000"),
                size=Decimal("1"),
                side=side,
                timestamp=now - timedelta(seconds=i)
            )

        vpin = analyzer.get_vpin(
            symbol="BTC-USD-PERP",
            bucket_size=Decimal("100000"),
            num_buckets=10
        )
        assert vpin is not None
        assert Decimal("0") <= vpin <= Decimal("1")

    def test_vpin_no_data(self, analyzer):
        """Test VPIN with no data."""
        vpin = analyzer.get_vpin(symbol="UNKNOWN")
        assert vpin is None

    def test_history_limit(self, analyzer):
        """Test history limit."""
        analyzer._max_history = 100
        for i in range(150):
            analyzer.record_trade(
                symbol="BTC-USD-PERP",
                price=Decimal("50000"),
                size=Decimal("1"),
                side="buy"
            )
        assert len(analyzer._trade_history["BTC-USD-PERP"]) == 100


# =============================================================================
# Quote Analyzer Tests
# =============================================================================

class TestQuoteAnalyzer:
    """Tests for QuoteAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return QuoteAnalyzer(
            very_active_threshold=100,
            active_threshold=50,
            reduced_threshold=10,
            absent_threshold=1
        )

    def test_record_quote(self, analyzer):
        """Test quote recording."""
        analyzer.record_quote(
            symbol="BTC-USD-PERP",
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            bid_size=Decimal("5"),
            ask_size=Decimal("4")
        )
        assert len(analyzer._quote_history["BTC-USD-PERP"]) == 1

    def test_analyze_very_active_mm(self, analyzer):
        """Test very active market maker detection."""
        now = datetime.now()
        for i in range(6000):  # 100 per second over 60 seconds
            analyzer.record_quote(
                symbol="BTC-USD-PERP",
                bid_price=Decimal("50000"),
                ask_price=Decimal("50010"),
                bid_size=Decimal("5"),
                ask_size=Decimal("4"),
                timestamp=now - timedelta(milliseconds=i * 10)
            )

        activity = analyzer.analyze_activity(
            symbol="BTC-USD-PERP",
            window_seconds=60,
            timestamp=now
        )
        assert activity is not None
        assert activity.mm_activity == MarketMakerActivity.VERY_ACTIVE

    def test_analyze_active_mm(self, analyzer):
        """Test active market maker detection."""
        now = datetime.now()
        for i in range(3000):  # 50 per second
            analyzer.record_quote(
                symbol="BTC-USD-PERP",
                bid_price=Decimal("50000"),
                ask_price=Decimal("50010"),
                bid_size=Decimal("5"),
                ask_size=Decimal("4"),
                timestamp=now - timedelta(milliseconds=i * 20)
            )

        activity = analyzer.analyze_activity(
            symbol="BTC-USD-PERP",
            window_seconds=60,
            timestamp=now
        )
        assert activity is not None
        assert activity.mm_activity in [MarketMakerActivity.ACTIVE, MarketMakerActivity.VERY_ACTIVE]

    def test_analyze_reduced_mm(self, analyzer):
        """Test reduced market maker activity detection."""
        now = datetime.now()
        for i in range(60):  # 1 per second
            analyzer.record_quote(
                symbol="BTC-USD-PERP",
                bid_price=Decimal("50000"),
                ask_price=Decimal("50010"),
                bid_size=Decimal("5"),
                ask_size=Decimal("4"),
                timestamp=now - timedelta(seconds=i)
            )

        activity = analyzer.analyze_activity(
            symbol="BTC-USD-PERP",
            window_seconds=60,
            timestamp=now
        )
        assert activity is not None
        assert activity.mm_activity == MarketMakerActivity.REDUCED

    def test_quote_to_trade_ratio(self, analyzer):
        """Test quote to trade ratio calculation."""
        now = datetime.now()
        for i in range(100):
            analyzer.record_quote(
                symbol="BTC-USD-PERP",
                bid_price=Decimal("50000"),
                ask_price=Decimal("50010"),
                bid_size=Decimal("5"),
                ask_size=Decimal("4"),
                timestamp=now - timedelta(seconds=i % 60)
            )

        activity = analyzer.analyze_activity(
            symbol="BTC-USD-PERP",
            window_seconds=60,
            trade_count=10,
            timestamp=now
        )
        assert activity is not None
        assert activity.quote_to_trade_ratio == Decimal("10")

    def test_bid_ask_update_counting(self, analyzer):
        """Test bid/ask update counting."""
        now = datetime.now()
        for i in range(10):
            analyzer.record_quote(
                symbol="BTC-USD-PERP",
                bid_price=Decimal("50000") + i,  # Changing bid
                ask_price=Decimal("50010"),       # Fixed ask
                bid_size=Decimal("5"),
                ask_size=Decimal("4"),
                timestamp=now - timedelta(seconds=10 - i)
            )

        activity = analyzer.analyze_activity(
            symbol="BTC-USD-PERP",
            window_seconds=60,
            timestamp=now
        )
        assert activity is not None
        assert activity.bid_updates == 9  # First doesn't count
        assert activity.ask_updates == 0

    def test_no_data_returns_none(self, analyzer):
        """Test no data returns None."""
        activity = analyzer.analyze_activity(
            symbol="UNKNOWN",
            window_seconds=60
        )
        assert activity is None


# =============================================================================
# Price Impact Estimator Tests
# =============================================================================

class TestPriceImpactEstimator:
    """Tests for PriceImpactEstimator."""

    @pytest.fixture
    def estimator(self):
        """Create estimator instance."""
        return PriceImpactEstimator(
            impact_coefficient=Decimal("0.1"),
            permanent_impact_ratio=Decimal("0.5")
        )

    @pytest.fixture
    def sample_asks(self):
        """Create sample asks for buy impact."""
        return [
            (Decimal("50010"), Decimal("10")),
            (Decimal("50020"), Decimal("10")),
            (Decimal("50030"), Decimal("10")),
            (Decimal("50040"), Decimal("10")),
            (Decimal("50050"), Decimal("10")),
        ]

    @pytest.fixture
    def sample_bids(self):
        """Create sample bids for sell impact."""
        return [
            (Decimal("50000"), Decimal("10")),
            (Decimal("49990"), Decimal("10")),
            (Decimal("49980"), Decimal("10")),
            (Decimal("49970"), Decimal("10")),
            (Decimal("49960"), Decimal("10")),
        ]

    def test_small_order_impact(self, estimator, sample_asks):
        """Test impact of small order."""
        impact = estimator.estimate_impact(
            symbol="BTC-USD-PERP",
            order_size=Decimal("5"),
            side="buy",
            orderbook=sample_asks,
            mid_price=Decimal("50005")
        )
        assert impact.estimated_slippage >= Decimal("0")
        assert impact.side == "buy"

    def test_large_order_impact(self, estimator, sample_asks):
        """Test impact of large order."""
        impact = estimator.estimate_impact(
            symbol="BTC-USD-PERP",
            order_size=Decimal("100"),
            side="buy",
            orderbook=sample_asks,
            mid_price=Decimal("50005")
        )
        # Large order should have higher impact
        assert impact.estimated_slippage > Decimal("0")

    def test_order_larger_than_book(self, estimator, sample_asks):
        """Test order larger than available book."""
        impact = estimator.estimate_impact(
            symbol="BTC-USD-PERP",
            order_size=Decimal("100"),  # More than book total of 50
            side="buy",
            orderbook=sample_asks,
            mid_price=Decimal("50005")
        )
        assert impact.market_impact_cost > Decimal("0")

    def test_sell_impact(self, estimator, sample_bids):
        """Test sell order impact."""
        impact = estimator.estimate_impact(
            symbol="BTC-USD-PERP",
            order_size=Decimal("5"),
            side="sell",
            orderbook=sample_bids,
            mid_price=Decimal("50005")
        )
        assert impact.side == "sell"

    def test_with_daily_volume(self, estimator, sample_asks):
        """Test impact with daily volume provided."""
        impact = estimator.estimate_impact(
            symbol="BTC-USD-PERP",
            order_size=Decimal("10"),
            side="buy",
            orderbook=sample_asks,
            mid_price=Decimal("50005"),
            daily_volume=Decimal("100000000")  # 100M daily volume
        )
        # With high volume, impact should be relatively low
        assert impact.estimated_impact_bps >= Decimal("0")

    def test_empty_orderbook(self, estimator):
        """Test with empty order book."""
        impact = estimator.estimate_impact(
            symbol="BTC-USD-PERP",
            order_size=Decimal("10"),
            side="buy",
            orderbook=[],
            mid_price=Decimal("50005")
        )
        # Should have significant impact with no liquidity
        assert impact.market_impact_cost > Decimal("0")


# =============================================================================
# Microstructure Analyzer Tests
# =============================================================================

class TestMicrostructureAnalyzer:
    """Tests for MicrostructureAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return MicrostructureAnalyzer()

    @pytest.fixture
    def sample_orderbook(self):
        """Create sample order book."""
        mid = Decimal("50000")
        bids = [
            (mid - Decimal("10") * i, Decimal("100")) for i in range(1, 21)
        ]
        asks = [
            (mid + Decimal("10") * i, Decimal("100")) for i in range(1, 21)
        ]
        return bids, asks

    def test_basic_analysis(self, analyzer, sample_orderbook):
        """Test basic microstructure analysis."""
        bids, asks = sample_orderbook
        snapshot = analyzer.analyze(
            symbol="BTC-USD-PERP",
            bids=bids,
            asks=asks
        )
        assert snapshot.symbol == "BTC-USD-PERP"
        assert snapshot.spread_metrics is not None
        assert snapshot.depth_metrics is not None
        assert Decimal("0") <= snapshot.overall_quality_score <= Decimal("100")

    def test_analysis_with_order_size(self, analyzer, sample_orderbook):
        """Test analysis with reference order size."""
        bids, asks = sample_orderbook
        snapshot = analyzer.analyze(
            symbol="BTC-USD-PERP",
            bids=bids,
            asks=asks,
            reference_order_size=Decimal("10")
        )
        assert snapshot.price_impact_buy is not None
        assert snapshot.price_impact_sell is not None

    def test_analysis_with_daily_volume(self, analyzer, sample_orderbook):
        """Test analysis with daily volume."""
        bids, asks = sample_orderbook
        snapshot = analyzer.analyze(
            symbol="BTC-USD-PERP",
            bids=bids,
            asks=asks,
            reference_order_size=Decimal("10"),
            daily_volume=Decimal("100000000")
        )
        assert snapshot is not None

    def test_record_trade(self, analyzer):
        """Test trade recording."""
        analyzer.record_trade(
            symbol="BTC-USD-PERP",
            price=Decimal("50000"),
            size=Decimal("1"),
            side="buy"
        )
        assert len(analyzer.trade_flow._trade_history["BTC-USD-PERP"]) == 1

    def test_record_quote(self, analyzer):
        """Test quote recording."""
        analyzer.record_quote(
            symbol="BTC-USD-PERP",
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            bid_size=Decimal("5"),
            ask_size=Decimal("4")
        )
        assert len(analyzer.quote._quote_history["BTC-USD-PERP"]) == 1

    def test_analysis_with_trade_history(self, analyzer, sample_orderbook):
        """Test analysis with trade history."""
        now = datetime.now()
        for i in range(20):
            analyzer.record_trade(
                symbol="BTC-USD-PERP",
                price=Decimal("50000"),
                size=Decimal("1"),
                side="buy" if i % 2 == 0 else "sell",
                timestamp=now - timedelta(seconds=i)
            )

        bids, asks = sample_orderbook
        snapshot = analyzer.analyze(
            symbol="BTC-USD-PERP",
            bids=bids,
            asks=asks,
            timestamp=now
        )
        assert snapshot.trade_imbalance is not None

    def test_analysis_with_quote_history(self, analyzer, sample_orderbook):
        """Test analysis with quote history."""
        now = datetime.now()
        for i in range(100):
            analyzer.record_quote(
                symbol="BTC-USD-PERP",
                bid_price=Decimal("50000") + (i % 5),
                ask_price=Decimal("50010") + (i % 5),
                bid_size=Decimal("5"),
                ask_size=Decimal("4"),
                timestamp=now - timedelta(milliseconds=i * 600)
            )

        # Also add some trades for quote-to-trade ratio
        for i in range(10):
            analyzer.record_trade(
                symbol="BTC-USD-PERP",
                price=Decimal("50000"),
                size=Decimal("1"),
                side="buy",
                timestamp=now - timedelta(seconds=i)
            )

        bids, asks = sample_orderbook
        snapshot = analyzer.analyze(
            symbol="BTC-USD-PERP",
            bids=bids,
            asks=asks,
            timestamp=now
        )
        assert snapshot.quote_activity is not None

    def test_quality_score_calculation(self, analyzer):
        """Test quality score calculation."""
        # Very good market
        bids = [(Decimal("50000") - i * 10, Decimal("1000")) for i in range(20)]
        asks = [(Decimal("50002") + i * 10, Decimal("1000")) for i in range(20)]

        snapshot = analyzer.analyze(
            symbol="BTC-USD-PERP",
            bids=bids,
            asks=asks
        )
        assert snapshot.overall_quality_score >= Decimal("50")

    def test_quality_history(self, analyzer, sample_orderbook):
        """Test quality history retrieval."""
        bids, asks = sample_orderbook
        for i in range(5):
            analyzer.analyze(
                symbol="BTC-USD-PERP",
                bids=bids,
                asks=asks
            )

        history = analyzer.get_quality_history("BTC-USD-PERP", limit=10)
        assert len(history) == 5

    def test_quality_history_no_data(self, analyzer):
        """Test quality history with no data."""
        history = analyzer.get_quality_history("UNKNOWN", limit=10)
        assert len(history) == 0

    def test_snapshot_history_limit(self, analyzer, sample_orderbook):
        """Test snapshot history limit."""
        analyzer._max_history = 100
        bids, asks = sample_orderbook
        for i in range(150):
            analyzer.analyze(
                symbol="BTC-USD-PERP",
                bids=bids,
                asks=asks
            )
        assert len(analyzer._snapshot_history["BTC-USD-PERP"]) == 100

    def test_crossed_market_quality_score(self, analyzer):
        """Test quality score with crossed market."""
        bids = [(Decimal("50010"), Decimal("100"))]  # Crossed
        asks = [(Decimal("50000"), Decimal("100"))]

        snapshot = analyzer.analyze(
            symbol="BTC-USD-PERP",
            bids=bids,
            asks=asks
        )
        # Crossed market should have lower quality
        assert snapshot.spread_metrics.is_crossed is True

    def test_empty_orderbook_analysis(self, analyzer):
        """Test analysis with empty order book."""
        snapshot = analyzer.analyze(
            symbol="BTC-USD-PERP",
            bids=[],
            asks=[]
        )
        assert snapshot.depth_metrics.liquidity_state == LiquidityState.CRITICALLY_ILLIQUID

    def test_multiple_symbols(self, analyzer):
        """Test analysis of multiple symbols."""
        for symbol in ["BTC-USD-PERP", "ETH-USD-PERP"]:
            bids = [(Decimal("1000") - i * 10, Decimal("10")) for i in range(10)]
            asks = [(Decimal("1010") + i * 10, Decimal("10")) for i in range(10)]
            analyzer.analyze(symbol=symbol, bids=bids, asks=asks)

        assert "BTC-USD-PERP" in analyzer._snapshot_history
        assert "ETH-USD-PERP" in analyzer._snapshot_history


# =============================================================================
# Integration Tests
# =============================================================================

class TestMicrostructureIntegration:
    """Integration tests for microstructure analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with custom settings."""
        return MicrostructureAnalyzer(
            spread_analyzer=SpreadAnalyzer(
                tight_threshold_bps=Decimal("3"),
                wide_threshold_bps=Decimal("15"),
                very_wide_threshold_bps=Decimal("40")
            ),
            depth_analyzer=DepthAnalyzer(
                highly_liquid_threshold=Decimal("2000000"),
                liquid_threshold=Decimal("1000000"),
                illiquid_threshold=Decimal("200000"),
                critical_threshold=Decimal("20000")
            ),
            trade_flow_analyzer=TradeFlowAnalyzer(
                high_toxicity_threshold=Decimal("0.8"),
                moderate_toxicity_threshold=Decimal("0.5")
            ),
            quote_analyzer=QuoteAnalyzer(
                very_active_threshold=200,
                active_threshold=100,
                reduced_threshold=20,
                absent_threshold=5
            )
        )

    def test_full_workflow(self, analyzer):
        """Test full analysis workflow."""
        symbol = "BTC-USD-PERP"
        now = datetime.now()

        # Record some quotes
        for i in range(50):
            analyzer.record_quote(
                symbol=symbol,
                bid_price=Decimal("50000") + (i % 3),
                ask_price=Decimal("50010") + (i % 3),
                bid_size=Decimal("5"),
                ask_size=Decimal("4"),
                timestamp=now - timedelta(seconds=60 - i)
            )

        # Record some trades
        for i in range(20):
            analyzer.record_trade(
                symbol=symbol,
                price=Decimal("50005"),
                size=Decimal("0.5"),
                side="buy" if i % 3 != 0 else "sell",
                timestamp=now - timedelta(seconds=60 - i * 3)
            )

        # Perform analysis
        bids = [(Decimal("50000") - i * 10, Decimal("50")) for i in range(20)]
        asks = [(Decimal("50010") + i * 10, Decimal("50")) for i in range(20)]

        snapshot = analyzer.analyze(
            symbol=symbol,
            bids=bids,
            asks=asks,
            reference_order_size=Decimal("5"),
            daily_volume=Decimal("50000000"),
            timestamp=now
        )

        # Verify all components
        assert snapshot.spread_metrics is not None
        assert snapshot.depth_metrics is not None
        assert snapshot.trade_imbalance is not None
        assert snapshot.quote_activity is not None
        assert snapshot.price_impact_buy is not None
        assert snapshot.price_impact_sell is not None
        assert snapshot.overall_quality_score > Decimal("0")

    def test_market_quality_degradation(self, analyzer):
        """Test detection of market quality degradation."""
        symbol = "BTC-USD-PERP"

        # Good market first
        bids_good = [(Decimal("50000") - i * 5, Decimal("100")) for i in range(20)]
        asks_good = [(Decimal("50005") + i * 5, Decimal("100")) for i in range(20)]

        snapshot_good = analyzer.analyze(
            symbol=symbol,
            bids=bids_good,
            asks=asks_good
        )

        # Poor market
        bids_poor = [(Decimal("50000") - i * 50, Decimal("5")) for i in range(5)]
        asks_poor = [(Decimal("50100") + i * 50, Decimal("5")) for i in range(5)]

        snapshot_poor = analyzer.analyze(
            symbol=symbol,
            bids=bids_poor,
            asks=asks_poor
        )

        # Quality should decrease
        assert snapshot_poor.overall_quality_score < snapshot_good.overall_quality_score

    def test_toxicity_detection(self, analyzer):
        """Test flow toxicity detection."""
        symbol = "BTC-USD-PERP"
        now = datetime.now()

        # Heavy one-sided flow
        for i in range(30):
            analyzer.record_trade(
                symbol=symbol,
                price=Decimal("50000"),
                size=Decimal("5"),
                side="buy",
                timestamp=now - timedelta(seconds=i)
            )

        bids = [(Decimal("50000"), Decimal("100"))]
        asks = [(Decimal("50010"), Decimal("100"))]

        snapshot = analyzer.analyze(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=now
        )

        assert snapshot.trade_imbalance is not None
        assert snapshot.trade_imbalance.flow_toxicity in [FlowToxicity.HIGH, FlowToxicity.VERY_HIGH]
