"""Tests for Arbitrage Detector module."""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from src.analytics.arbitrage_detector import (
    ArbitrageDetector,
    ArbitrageOpportunity,
    ArbitrageSummary,
    ArbitrageType,
    BasisTrade,
    FundingArbitrage,
    MarketPrice,
    OpportunityStatus,
    RiskLevel,
    TriangularPath,
    get_arbitrage_detector,
    reset_arbitrage_detector,
)


class TestArbitrageTypeEnum:
    """Tests for ArbitrageType enum."""

    def test_all_types(self):
        """Test all arbitrage types."""
        types = [
            ArbitrageType.CROSS_MARKET,
            ArbitrageType.BASIS_TRADE,
            ArbitrageType.FUNDING_ARB,
            ArbitrageType.TRIANGULAR,
            ArbitrageType.STATISTICAL,
        ]
        assert len(types) == 5

    def test_type_values(self):
        """Test type values."""
        assert ArbitrageType.CROSS_MARKET.value == "cross_market"
        assert ArbitrageType.BASIS_TRADE.value == "basis_trade"


class TestOpportunityStatusEnum:
    """Tests for OpportunityStatus enum."""

    def test_all_statuses(self):
        """Test all statuses."""
        statuses = [
            OpportunityStatus.ACTIVE,
            OpportunityStatus.EXPIRED,
            OpportunityStatus.EXECUTED,
            OpportunityStatus.MISSED,
        ]
        assert len(statuses) == 4


class TestRiskLevelEnum:
    """Tests for RiskLevel enum."""

    def test_all_levels(self):
        """Test all risk levels."""
        levels = [
            RiskLevel.VERY_LOW,
            RiskLevel.LOW,
            RiskLevel.MEDIUM,
            RiskLevel.HIGH,
            RiskLevel.VERY_HIGH,
        ]
        assert len(levels) == 5


class TestMarketPrice:
    """Tests for MarketPrice dataclass."""

    def test_create_price(self):
        """Test creating market price."""
        price = MarketPrice(
            market="BTC-USD-PERP",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            mid=Decimal("50005"),
            spread_bps=2.0,
            timestamp=datetime.now(),
        )
        assert price.market == "BTC-USD-PERP"
        assert price.bid == Decimal("50000")

    def test_to_dict(self):
        """Test converting to dict."""
        price = MarketPrice(
            market="BTC-USD-PERP",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            mid=Decimal("50005"),
            spread_bps=2.0,
            timestamp=datetime.now(),
            volume_24h=Decimal("1000000"),
        )
        d = price.to_dict()
        assert d["market"] == "BTC-USD-PERP"
        assert d["volume_24h"] == 1000000.0


class TestArbitrageOpportunity:
    """Tests for ArbitrageOpportunity dataclass."""

    def test_create_opportunity(self):
        """Test creating opportunity."""
        opp = ArbitrageOpportunity(
            id="ARB-000001",
            arb_type=ArbitrageType.CROSS_MARKET,
            buy_market="MARKET-A",
            sell_market="MARKET-B",
            buy_price=Decimal("50000"),
            sell_price=Decimal("50020"),
            spread=Decimal("20"),
            spread_bps=4.0,
            profit_potential=Decimal("15"),
            max_size=Decimal("10"),
            estimated_profit=Decimal("150"),
            risk_level=RiskLevel.MEDIUM,
            confidence=0.8,
            timestamp=datetime.now(),
            expiry=datetime.now() + timedelta(seconds=60),
        )
        assert opp.id == "ARB-000001"
        assert opp.spread == Decimal("20")

    def test_to_dict(self):
        """Test converting to dict."""
        opp = ArbitrageOpportunity(
            id="ARB-000001",
            arb_type=ArbitrageType.CROSS_MARKET,
            buy_market="MARKET-A",
            sell_market="MARKET-B",
            buy_price=Decimal("50000"),
            sell_price=Decimal("50020"),
            spread=Decimal("20"),
            spread_bps=4.0,
            profit_potential=Decimal("15"),
            max_size=Decimal("10"),
            estimated_profit=Decimal("150"),
            risk_level=RiskLevel.MEDIUM,
            confidence=0.8,
            timestamp=datetime.now(),
            expiry=datetime.now() + timedelta(seconds=60),
        )
        d = opp.to_dict()
        assert d["id"] == "ARB-000001"
        assert d["arb_type"] == "cross_market"


class TestBasisTrade:
    """Tests for BasisTrade dataclass."""

    def test_create_basis_trade(self):
        """Test creating basis trade."""
        trade = BasisTrade(
            spot_market="BTC-USD",
            perp_market="BTC-USD-PERP",
            spot_price=Decimal("50000"),
            perp_price=Decimal("50100"),
            basis=Decimal("100"),
            basis_pct=0.2,
            annualized_yield=10.4,
            funding_rate=Decimal("0.0003"),
            net_yield=12.5,
            recommended_direction="long_spot_short_perp",
            timestamp=datetime.now(),
        )
        assert trade.basis == Decimal("100")
        assert trade.recommended_direction == "long_spot_short_perp"

    def test_to_dict(self):
        """Test converting to dict."""
        trade = BasisTrade(
            spot_market="BTC-USD",
            perp_market="BTC-USD-PERP",
            spot_price=Decimal("50000"),
            perp_price=Decimal("50100"),
            basis=Decimal("100"),
            basis_pct=0.2,
            annualized_yield=10.4,
            funding_rate=Decimal("0.0003"),
            net_yield=12.5,
            recommended_direction="long_spot_short_perp",
            timestamp=datetime.now(),
        )
        d = trade.to_dict()
        assert d["spot_market"] == "BTC-USD"


class TestFundingArbitrage:
    """Tests for FundingArbitrage dataclass."""

    def test_create_funding_arb(self):
        """Test creating funding arbitrage."""
        arb = FundingArbitrage(
            long_market="ETH-USD-PERP",
            short_market="BTC-USD-PERP",
            long_funding=Decimal("0.0001"),
            short_funding=Decimal("0.0005"),
            funding_spread=Decimal("0.0004"),
            annualized_spread=43.8,
            recommended_size=Decimal("1000"),
            estimated_daily_profit=Decimal("1.2"),
            risk_level=RiskLevel.MEDIUM,
            timestamp=datetime.now(),
        )
        assert arb.funding_spread == Decimal("0.0004")

    def test_to_dict(self):
        """Test converting to dict."""
        arb = FundingArbitrage(
            long_market="ETH-USD-PERP",
            short_market="BTC-USD-PERP",
            long_funding=Decimal("0.0001"),
            short_funding=Decimal("0.0005"),
            funding_spread=Decimal("0.0004"),
            annualized_spread=43.8,
            recommended_size=Decimal("1000"),
            estimated_daily_profit=Decimal("1.2"),
            risk_level=RiskLevel.MEDIUM,
            timestamp=datetime.now(),
        )
        d = arb.to_dict()
        assert d["long_market"] == "ETH-USD-PERP"


class TestTriangularPath:
    """Tests for TriangularPath dataclass."""

    def test_create_path(self):
        """Test creating triangular path."""
        path = TriangularPath(
            market_a="BTC-USD",
            market_b="ETH-BTC",
            market_c="ETH-USD",
            path=[("BTC-USD", "buy"), ("ETH-BTC", "sell"), ("ETH-USD", "sell")],
            profit_pct=0.05,
            execution_cost_pct=0.03,
            net_profit_pct=0.02,
            timestamp=datetime.now(),
        )
        assert len(path.path) == 3

    def test_to_dict(self):
        """Test converting to dict."""
        path = TriangularPath(
            market_a="BTC-USD",
            market_b="ETH-BTC",
            market_c="ETH-USD",
            path=[],
            profit_pct=0.05,
            execution_cost_pct=0.03,
            net_profit_pct=0.02,
            timestamp=datetime.now(),
        )
        d = path.to_dict()
        assert d["market_a"] == "BTC-USD"


class TestArbitrageDetector:
    """Tests for ArbitrageDetector class."""

    @pytest.fixture
    def detector(self):
        """Create detector."""
        return ArbitrageDetector()

    def test_init_defaults(self):
        """Test default initialization."""
        detector = ArbitrageDetector()
        assert detector.min_spread_bps == 5.0
        assert detector.fee_rate_bps == 5.0

    def test_init_custom(self):
        """Test custom initialization."""
        detector = ArbitrageDetector(
            min_spread_bps=10.0,
            fee_rate_bps=3.0,
        )
        assert detector.min_spread_bps == 10.0
        assert detector.fee_rate_bps == 3.0

    def test_update_price(self, detector):
        """Test updating price."""
        price = MarketPrice(
            market="BTC-USD-PERP",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            mid=Decimal("50005"),
            spread_bps=2.0,
            timestamp=datetime.now(),
        )
        detector.update_price(price)

        retrieved = detector.get_price("BTC-USD-PERP")
        assert retrieved is not None
        assert retrieved.bid == Decimal("50000")

    def test_update_prices(self, detector):
        """Test updating prices via helper."""
        detector.update_prices(
            "BTC-USD-PERP",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            volume_24h=Decimal("1000000"),
        )

        price = detector.get_price("BTC-USD-PERP")
        assert price is not None
        assert price.mid == Decimal("50005")

    def test_update_funding_rate(self, detector):
        """Test updating funding rate."""
        detector.update_funding_rate("BTC-USD-PERP", Decimal("0.0003"))
        # No getter, but used internally

    def test_get_price_missing(self, detector):
        """Test getting missing price."""
        price = detector.get_price("UNKNOWN")
        assert price is None


class TestCrossMarketArbitrage:
    """Tests for cross-market arbitrage detection."""

    @pytest.fixture
    def detector(self):
        """Create detector with prices."""
        detector = ArbitrageDetector(min_spread_bps=5.0, fee_rate_bps=2.0)

        # Market A: Lower prices
        detector.update_prices(
            "MARKET-A",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            volume_24h=Decimal("10000000"),
        )

        # Market B: Higher prices (arbitrage opportunity)
        # Spread after fees needs to be > 5 bps
        # B.bid - A.ask = 50080 - 50010 = 70, which is ~14 bps
        detector.update_prices(
            "MARKET-B",
            bid=Decimal("50080"),
            ask=Decimal("50090"),
            volume_24h=Decimal("10000000"),
        )

        return detector

    def test_detect_cross_market_opportunity(self, detector):
        """Test detecting cross-market opportunity."""
        opp = detector.detect_cross_market("MARKET-A", "MARKET-B")
        assert opp is not None
        assert opp.arb_type == ArbitrageType.CROSS_MARKET
        assert opp.buy_market == "MARKET-A"
        assert opp.sell_market == "MARKET-B"

    def test_no_opportunity_when_no_spread(self):
        """Test no opportunity when prices equal."""
        detector = ArbitrageDetector()
        detector.update_prices("A", Decimal("50000"), Decimal("50010"))
        detector.update_prices("B", Decimal("50000"), Decimal("50010"))

        opp = detector.detect_cross_market("A", "B")
        assert opp is None

    def test_no_opportunity_missing_market(self, detector):
        """Test no opportunity when market missing."""
        opp = detector.detect_cross_market("MARKET-A", "UNKNOWN")
        assert opp is None

    def test_opportunity_spread_calculation(self, detector):
        """Test spread is calculated correctly."""
        opp = detector.detect_cross_market("MARKET-A", "MARKET-B")
        assert opp is not None
        # Spread = B.bid - A.ask = 50080 - 50010 = 70
        assert opp.spread == Decimal("70")

    def test_opportunity_risk_level(self, detector):
        """Test risk level is assessed."""
        opp = detector.detect_cross_market("MARKET-A", "MARKET-B")
        assert opp is not None
        assert opp.risk_level in RiskLevel


class TestBasisTradeDetection:
    """Tests for basis trade detection."""

    @pytest.fixture
    def detector(self):
        """Create detector."""
        detector = ArbitrageDetector()

        # Spot market
        detector.update_prices(
            "BTC-USD",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
        )

        # Perp market (premium)
        detector.update_prices(
            "BTC-USD-PERP",
            bid=Decimal("50100"),
            ask=Decimal("50110"),
        )

        detector.update_funding_rate("BTC-USD-PERP", Decimal("0.0003"))

        return detector

    def test_detect_basis_trade(self, detector):
        """Test detecting basis trade."""
        trade = detector.detect_basis_trade("BTC-USD", "BTC-USD-PERP")
        assert trade is not None
        assert trade.basis > 0  # Perp at premium

    def test_basis_trade_direction(self, detector):
        """Test recommended direction."""
        trade = detector.detect_basis_trade("BTC-USD", "BTC-USD-PERP")
        assert trade is not None
        # Positive basis = short perp, long spot
        assert trade.recommended_direction == "long_spot_short_perp"

    def test_basis_trade_no_data(self, detector):
        """Test basis trade with missing data."""
        trade = detector.detect_basis_trade("UNKNOWN", "BTC-USD-PERP")
        assert trade is None

    def test_basis_pct_calculation(self, detector):
        """Test basis percentage calculation."""
        trade = detector.detect_basis_trade("BTC-USD", "BTC-USD-PERP")
        assert trade is not None
        # Basis = 50105 - 50005 = 100, ~0.2%
        assert trade.basis_pct == pytest.approx(0.2, rel=0.1)


class TestFundingArbitrageDetection:
    """Tests for funding arbitrage detection."""

    @pytest.fixture
    def detector(self):
        """Create detector with funding rates."""
        detector = ArbitrageDetector()

        detector.update_funding_rate("BTC-USD-PERP", Decimal("0.0005"))  # High
        detector.update_funding_rate("ETH-USD-PERP", Decimal("0.0001"))  # Low
        detector.update_funding_rate("SOL-USD-PERP", Decimal("0.0003"))  # Medium

        return detector

    def test_detect_funding_arbitrage(self, detector):
        """Test detecting funding arbitrage."""
        arb = detector.detect_funding_arbitrage()
        assert arb is not None
        assert arb.long_market == "ETH-USD-PERP"  # Lowest funding
        assert arb.short_market == "BTC-USD-PERP"  # Highest funding

    def test_funding_spread(self, detector):
        """Test funding spread calculation."""
        arb = detector.detect_funding_arbitrage()
        assert arb is not None
        # Spread = 0.0005 - 0.0001 = 0.0004
        assert arb.funding_spread == Decimal("0.0004")

    def test_funding_arb_specific_markets(self, detector):
        """Test with specific markets."""
        arb = detector.detect_funding_arbitrage(["ETH-USD-PERP", "SOL-USD-PERP"])
        assert arb is not None
        assert arb.long_market == "ETH-USD-PERP"
        assert arb.short_market == "SOL-USD-PERP"

    def test_no_funding_arb_single_market(self, detector):
        """Test no arb with single market."""
        arb = detector.detect_funding_arbitrage(["BTC-USD-PERP"])
        assert arb is None


class TestDetectAll:
    """Tests for detect_all method."""

    @pytest.fixture
    def detector(self):
        """Create detector with multiple markets."""
        detector = ArbitrageDetector(min_spread_bps=5.0, fee_rate_bps=2.0)

        detector.update_prices("A", Decimal("50000"), Decimal("50010"), Decimal("10000000"))
        detector.update_prices("B", Decimal("50080"), Decimal("50090"), Decimal("10000000"))
        detector.update_prices("C", Decimal("50000"), Decimal("50010"), Decimal("10000000"))

        return detector

    def test_detect_all(self, detector):
        """Test detecting all opportunities."""
        opps = detector.detect_all()
        assert len(opps) >= 1  # At least A-B opportunity


class TestOpportunityManagement:
    """Tests for opportunity management."""

    @pytest.fixture
    def detector(self):
        """Create detector with opportunity."""
        detector = ArbitrageDetector(min_spread_bps=5.0, fee_rate_bps=2.0)

        detector.update_prices("A", Decimal("50000"), Decimal("50010"), Decimal("10000000"))
        detector.update_prices("B", Decimal("50080"), Decimal("50090"), Decimal("10000000"))

        return detector

    def test_get_active_opportunities(self, detector):
        """Test getting active opportunities."""
        detector.detect_cross_market("A", "B")
        active = detector.get_active_opportunities()
        assert len(active) >= 1

    def test_get_opportunity_by_id(self, detector):
        """Test getting specific opportunity."""
        opp = detector.detect_cross_market("A", "B")
        retrieved = detector.get_opportunity(opp.id)
        assert retrieved is not None
        assert retrieved.id == opp.id

    def test_mark_executed(self, detector):
        """Test marking opportunity as executed."""
        opp = detector.detect_cross_market("A", "B")
        success = detector.mark_executed(opp.id)
        assert success is True

        # Should be removed from active
        active = detector.get_active_opportunities()
        assert all(o.id != opp.id for o in active)

    def test_mark_executed_nonexistent(self, detector):
        """Test marking non-existent opportunity."""
        success = detector.mark_executed("NONEXISTENT")
        assert success is False


class TestSummary:
    """Tests for summary generation."""

    @pytest.fixture
    def detector(self):
        """Create detector with opportunities."""
        detector = ArbitrageDetector(min_spread_bps=5.0, fee_rate_bps=2.0)

        detector.update_prices("A", Decimal("50000"), Decimal("50010"), Decimal("10000000"))
        detector.update_prices("B", Decimal("50080"), Decimal("50090"), Decimal("10000000"))
        detector.detect_cross_market("A", "B")

        return detector

    def test_get_summary(self, detector):
        """Test getting summary."""
        summary = detector.get_summary()
        assert isinstance(summary, ArbitrageSummary)
        assert summary.total_opportunities >= 1

    def test_summary_best_opportunity(self, detector):
        """Test summary includes best opportunity."""
        summary = detector.get_summary()
        assert summary.best_opportunity is not None

    def test_summary_to_dict(self, detector):
        """Test summary to_dict."""
        summary = detector.get_summary()
        d = summary.to_dict()
        assert "total_opportunities" in d


class TestBestOpportunity:
    """Tests for best opportunity retrieval."""

    @pytest.fixture
    def detector(self):
        """Create detector with opportunities."""
        detector = ArbitrageDetector(min_spread_bps=5.0, fee_rate_bps=2.0)

        # Create multiple opportunities
        detector.update_prices("A", Decimal("50000"), Decimal("50010"), Decimal("10000000"))
        detector.update_prices("B", Decimal("50080"), Decimal("50090"), Decimal("10000000"))
        detector.update_prices("C", Decimal("50150"), Decimal("50160"), Decimal("10000000"))

        detector.detect_cross_market("A", "B")
        detector.detect_cross_market("A", "C")

        return detector

    def test_get_best_opportunity(self, detector):
        """Test getting best opportunity."""
        best = detector.get_best_opportunity()
        assert best is not None
        # A-C should have higher profit (larger spread: 50150-50010=140 vs 50080-50010=70)
        assert best.sell_market == "C"

    def test_get_best_by_type(self, detector):
        """Test getting best by type."""
        best = detector.get_best_opportunity(arb_type=ArbitrageType.CROSS_MARKET)
        assert best is not None
        assert best.arb_type == ArbitrageType.CROSS_MARKET

    def test_get_best_no_opportunities(self):
        """Test getting best with no opportunities."""
        detector = ArbitrageDetector()
        best = detector.get_best_opportunity()
        assert best is None


class TestCallbacks:
    """Tests for callbacks."""

    @pytest.fixture
    def detector(self):
        """Create detector."""
        detector = ArbitrageDetector(min_spread_bps=5.0, fee_rate_bps=2.0)
        detector.update_prices("A", Decimal("50000"), Decimal("50010"), Decimal("10000000"))
        detector.update_prices("B", Decimal("50080"), Decimal("50090"), Decimal("10000000"))
        return detector

    def test_add_callback(self, detector):
        """Test adding callback."""
        results = []

        def callback(opp):
            results.append(opp)

        detector.add_callback(callback)
        detector.detect_cross_market("A", "B")

        assert len(results) == 1

    def test_remove_callback(self, detector):
        """Test removing callback."""
        def callback(opp):
            pass

        detector.add_callback(callback)
        removed = detector.remove_callback(callback)
        assert removed is True

    def test_remove_nonexistent_callback(self, detector):
        """Test removing non-existent callback."""
        def callback(opp):
            pass

        removed = detector.remove_callback(callback)
        assert removed is False


class TestHistory:
    """Tests for history."""

    @pytest.fixture
    def detector(self):
        """Create detector with executed opportunity."""
        detector = ArbitrageDetector(min_spread_bps=5.0, fee_rate_bps=2.0)
        detector.update_prices("A", Decimal("50000"), Decimal("50010"), Decimal("10000000"))
        detector.update_prices("B", Decimal("50080"), Decimal("50090"), Decimal("10000000"))

        opp = detector.detect_cross_market("A", "B")
        detector.mark_executed(opp.id)

        return detector

    def test_get_history(self, detector):
        """Test getting history."""
        history = detector.get_history()
        assert len(history) >= 1

    def test_get_history_by_status(self, detector):
        """Test getting history by status."""
        history = detector.get_history(status=OpportunityStatus.EXECUTED)
        assert len(history) >= 1
        assert all(o.status == OpportunityStatus.EXECUTED for o in history)

    def test_get_history_limited(self, detector):
        """Test getting limited history."""
        history = detector.get_history(limit=1)
        assert len(history) <= 1


class TestUtilities:
    """Tests for utility methods."""

    @pytest.fixture
    def detector(self):
        """Create detector with data."""
        detector = ArbitrageDetector()
        detector.update_prices("A", Decimal("50000"), Decimal("50010"))
        detector.update_prices("B", Decimal("50000"), Decimal("50010"))
        return detector

    def test_get_markets(self, detector):
        """Test getting markets list."""
        markets = detector.get_markets()
        assert "A" in markets
        assert "B" in markets

    def test_clear_market(self, detector):
        """Test clearing market."""
        detector.clear_market("A")
        price = detector.get_price("A")
        assert price is None

    def test_clear_all(self, detector):
        """Test clearing all."""
        detector.clear_all()
        markets = detector.get_markets()
        assert len(markets) == 0


class TestGlobalFunctions:
    """Tests for global functions."""

    def test_get_arbitrage_detector(self):
        """Test getting global detector."""
        reset_arbitrage_detector()
        detector = get_arbitrage_detector()
        assert detector is not None

    def test_get_arbitrage_detector_singleton(self):
        """Test detector is singleton."""
        reset_arbitrage_detector()
        d1 = get_arbitrage_detector()
        d2 = get_arbitrage_detector()
        assert d1 is d2

    def test_reset_arbitrage_detector(self):
        """Test resetting detector."""
        d1 = get_arbitrage_detector()
        reset_arbitrage_detector()
        d2 = get_arbitrage_detector()
        assert d1 is not d2


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def detector(self):
        """Create detector."""
        return ArbitrageDetector()

    def test_zero_prices(self, detector):
        """Test with zero prices."""
        detector.update_prices("A", Decimal("0"), Decimal("0"))
        opp = detector.detect_cross_market("A", "A")
        assert opp is None

    def test_very_small_spread(self, detector):
        """Test with very small spread."""
        detector.update_prices("A", Decimal("50000"), Decimal("50001"))
        detector.update_prices("B", Decimal("50001"), Decimal("50002"))

        opp = detector.detect_cross_market("A", "B")
        # Should be filtered by min_spread_bps
        assert opp is None

    def test_negative_spread(self, detector):
        """Test with B lower than A."""
        detector.update_prices("A", Decimal("50010"), Decimal("50020"), Decimal("10000000"))
        detector.update_prices("B", Decimal("50000"), Decimal("50010"), Decimal("10000000"))

        # min_spread_bps=5.0 default, this spread is too small
        opp = detector.detect_cross_market("A", "B")
        # No profitable arbitrage
        assert opp is None

    def test_stale_price(self, detector):
        """Test with stale price data."""
        old_price = MarketPrice(
            market="OLD",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            mid=Decimal("50005"),
            spread_bps=2.0,
            timestamp=datetime.now() - timedelta(minutes=10),
        )
        detector.update_price(old_price)

        detector.update_prices("NEW", Decimal("50030"), Decimal("50035"), Decimal("10000000"))

        opp = detector.detect_cross_market("OLD", "NEW")
        if opp:
            # Should have higher risk due to staleness
            assert opp.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH, RiskLevel.MEDIUM]

    def test_max_opportunities_limit(self):
        """Test opportunity limit."""
        detector = ArbitrageDetector(
            min_spread_bps=1.0,
            fee_rate_bps=0.5,
            max_opportunities=2,
        )

        # Create many opportunities
        for i in range(5):
            detector.update_prices(
                f"M{i}",
                Decimal(str(50000 + i * 100)),
                Decimal(str(50005 + i * 100)),
                Decimal("10000000"),
            )

        detector.detect_all()
        active = detector.get_active_opportunities()
        assert len(active) <= 2

    def test_opportunity_expiry(self):
        """Test opportunity expiry."""
        detector = ArbitrageDetector(
            min_spread_bps=5.0,
            fee_rate_bps=2.0,
            opportunity_ttl_seconds=1,  # 1 second TTL
        )

        detector.update_prices("A", Decimal("50000"), Decimal("50010"), Decimal("10000000"))
        detector.update_prices("B", Decimal("50080"), Decimal("50090"), Decimal("10000000"))

        opp = detector.detect_cross_market("A", "B")
        assert opp is not None

        # Wait for expiry
        import time
        time.sleep(1.5)

        active = detector.get_active_opportunities()
        # Should be expired
        assert all(o.id != opp.id for o in active)
