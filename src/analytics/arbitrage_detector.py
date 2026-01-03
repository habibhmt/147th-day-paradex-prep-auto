"""
Arbitrage Detector Module.

Detects arbitrage opportunities across markets,
funding rate arbitrage, and basis trade opportunities.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable


class ArbitrageType(Enum):
    """Type of arbitrage opportunity."""

    CROSS_MARKET = "cross_market"  # Same asset, different markets
    BASIS_TRADE = "basis_trade"  # Spot vs perpetual
    FUNDING_ARB = "funding_arb"  # Funding rate arbitrage
    TRIANGULAR = "triangular"  # Three-asset triangular arbitrage
    STATISTICAL = "statistical"  # Mean reversion based


class OpportunityStatus(Enum):
    """Status of an opportunity."""

    ACTIVE = "active"
    EXPIRED = "expired"
    EXECUTED = "executed"
    MISSED = "missed"


class RiskLevel(Enum):
    """Risk level of arbitrage."""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class MarketPrice:
    """Price information for a market."""

    market: str
    bid: Decimal
    ask: Decimal
    mid: Decimal
    spread_bps: float
    timestamp: datetime
    volume_24h: Decimal = Decimal("0")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market": self.market,
            "bid": float(self.bid),
            "ask": float(self.ask),
            "mid": float(self.mid),
            "spread_bps": self.spread_bps,
            "timestamp": self.timestamp.isoformat(),
            "volume_24h": float(self.volume_24h),
        }


@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity."""

    id: str
    arb_type: ArbitrageType
    buy_market: str
    sell_market: str
    buy_price: Decimal
    sell_price: Decimal
    spread: Decimal
    spread_bps: float
    profit_potential: Decimal  # Expected profit per unit
    max_size: Decimal
    estimated_profit: Decimal
    risk_level: RiskLevel
    confidence: float
    timestamp: datetime
    expiry: datetime
    status: OpportunityStatus = OpportunityStatus.ACTIVE
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "arb_type": self.arb_type.value,
            "buy_market": self.buy_market,
            "sell_market": self.sell_market,
            "buy_price": float(self.buy_price),
            "sell_price": float(self.sell_price),
            "spread": float(self.spread),
            "spread_bps": self.spread_bps,
            "profit_potential": float(self.profit_potential),
            "max_size": float(self.max_size),
            "estimated_profit": float(self.estimated_profit),
            "risk_level": self.risk_level.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "expiry": self.expiry.isoformat(),
            "status": self.status.value,
            "notes": self.notes,
        }


@dataclass
class BasisTrade:
    """Spot-perpetual basis trade opportunity."""

    spot_market: str
    perp_market: str
    spot_price: Decimal
    perp_price: Decimal
    basis: Decimal  # Perp - Spot
    basis_pct: float
    annualized_yield: float
    funding_rate: Decimal
    net_yield: float  # Basis yield + funding
    recommended_direction: str  # 'long_spot_short_perp' or 'short_spot_long_perp'
    timestamp: datetime
    expiry_days: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "spot_market": self.spot_market,
            "perp_market": self.perp_market,
            "spot_price": float(self.spot_price),
            "perp_price": float(self.perp_price),
            "basis": float(self.basis),
            "basis_pct": self.basis_pct,
            "annualized_yield": self.annualized_yield,
            "funding_rate": float(self.funding_rate),
            "net_yield": self.net_yield,
            "recommended_direction": self.recommended_direction,
            "timestamp": self.timestamp.isoformat(),
            "expiry_days": self.expiry_days,
        }


@dataclass
class FundingArbitrage:
    """Funding rate arbitrage opportunity."""

    long_market: str  # Market to go long (lower funding)
    short_market: str  # Market to go short (higher funding)
    long_funding: Decimal
    short_funding: Decimal
    funding_spread: Decimal
    annualized_spread: float
    recommended_size: Decimal
    estimated_daily_profit: Decimal
    risk_level: RiskLevel
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "long_market": self.long_market,
            "short_market": self.short_market,
            "long_funding": float(self.long_funding),
            "short_funding": float(self.short_funding),
            "funding_spread": float(self.funding_spread),
            "annualized_spread": self.annualized_spread,
            "recommended_size": float(self.recommended_size),
            "estimated_daily_profit": float(self.estimated_daily_profit),
            "risk_level": self.risk_level.value,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TriangularPath:
    """Triangular arbitrage path."""

    market_a: str  # A/B
    market_b: str  # B/C
    market_c: str  # A/C
    path: list[tuple[str, str]]  # [(market, direction), ...]
    profit_pct: float
    execution_cost_pct: float
    net_profit_pct: float
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market_a": self.market_a,
            "market_b": self.market_b,
            "market_c": self.market_c,
            "path": self.path,
            "profit_pct": self.profit_pct,
            "execution_cost_pct": self.execution_cost_pct,
            "net_profit_pct": self.net_profit_pct,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ArbitrageSummary:
    """Summary of arbitrage opportunities."""

    timestamp: datetime
    total_opportunities: int
    by_type: dict[ArbitrageType, int]
    best_opportunity: ArbitrageOpportunity | None
    total_potential_profit: Decimal
    avg_spread_bps: float
    markets_analyzed: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_opportunities": self.total_opportunities,
            "by_type": {k.value: v for k, v in self.by_type.items()},
            "best_opportunity": self.best_opportunity.to_dict() if self.best_opportunity else None,
            "total_potential_profit": float(self.total_potential_profit),
            "avg_spread_bps": self.avg_spread_bps,
            "markets_analyzed": self.markets_analyzed,
        }


class ArbitrageDetector:
    """Detects arbitrage opportunities across markets."""

    def __init__(
        self,
        min_spread_bps: float = 5.0,
        min_profit_threshold: Decimal = Decimal("1"),
        fee_rate_bps: float = 5.0,
        opportunity_ttl_seconds: int = 60,
        max_opportunities: int = 100,
    ):
        """
        Initialize detector.

        Args:
            min_spread_bps: Minimum spread to consider
            min_profit_threshold: Minimum profit to report
            fee_rate_bps: Trading fee rate
            opportunity_ttl_seconds: Time before opportunity expires
            max_opportunities: Maximum opportunities to track
        """
        self.min_spread_bps = min_spread_bps
        self.min_profit_threshold = min_profit_threshold
        self.fee_rate_bps = fee_rate_bps
        self.opportunity_ttl_seconds = opportunity_ttl_seconds
        self.max_opportunities = max_opportunities

        # Market prices
        self._prices: dict[str, MarketPrice] = {}

        # Funding rates
        self._funding_rates: dict[str, Decimal] = {}

        # Active opportunities
        self._opportunities: dict[str, ArbitrageOpportunity] = {}

        # History
        self._history: list[ArbitrageOpportunity] = []

        # Callbacks
        self._callbacks: list[Callable[[ArbitrageOpportunity], None]] = []

        # Opportunity counter
        self._opp_counter = 0

    def update_price(self, price: MarketPrice) -> None:
        """
        Update market price.

        Args:
            price: Market price data
        """
        self._prices[price.market] = price

    def update_prices(
        self,
        market: str,
        bid: Decimal,
        ask: Decimal,
        volume_24h: Decimal = Decimal("0"),
    ) -> None:
        """
        Update market prices.

        Args:
            market: Market symbol
            bid: Bid price
            ask: Ask price
            volume_24h: 24h volume
        """
        mid = (bid + ask) / 2
        spread_bps = float((ask - bid) / mid * 10000) if mid > 0 else 0

        price = MarketPrice(
            market=market,
            bid=bid,
            ask=ask,
            mid=mid,
            spread_bps=spread_bps,
            timestamp=datetime.now(),
            volume_24h=volume_24h,
        )
        self.update_price(price)

    def update_funding_rate(self, market: str, rate: Decimal) -> None:
        """
        Update funding rate for a market.

        Args:
            market: Market symbol
            rate: Funding rate (e.g., 0.0003 for 0.03%)
        """
        self._funding_rates[market] = rate

    def get_price(self, market: str) -> MarketPrice | None:
        """Get price for a market."""
        return self._prices.get(market)

    def _generate_opportunity_id(self) -> str:
        """Generate unique opportunity ID."""
        self._opp_counter += 1
        return f"ARB-{self._opp_counter:06d}"

    def detect_cross_market(
        self,
        market_a: str,
        market_b: str,
    ) -> ArbitrageOpportunity | None:
        """
        Detect cross-market arbitrage.

        Args:
            market_a: First market
            market_b: Second market

        Returns:
            Opportunity or None
        """
        price_a = self._prices.get(market_a)
        price_b = self._prices.get(market_b)

        if not price_a or not price_b:
            return None

        # Check if buying on A and selling on B is profitable
        # Buy at ask on A, sell at bid on B
        spread_ab = price_b.bid - price_a.ask
        spread_ab_bps = float(spread_ab / price_a.mid * 10000) if price_a.mid > 0 else 0

        # Check reverse
        spread_ba = price_a.bid - price_b.ask
        spread_ba_bps = float(spread_ba / price_b.mid * 10000) if price_b.mid > 0 else 0

        # Account for fees (both sides)
        total_fees_bps = self.fee_rate_bps * 2

        # Determine best direction
        if spread_ab_bps > spread_ba_bps and spread_ab_bps > total_fees_bps:
            buy_market = market_a
            sell_market = market_b
            buy_price = price_a.ask
            sell_price = price_b.bid
            spread = spread_ab
            spread_bps = spread_ab_bps
        elif spread_ba_bps > total_fees_bps:
            buy_market = market_b
            sell_market = market_a
            buy_price = price_b.ask
            sell_price = price_a.bid
            spread = spread_ba
            spread_bps = spread_ba_bps
        else:
            return None

        # Check minimum spread
        net_spread_bps = spread_bps - total_fees_bps
        if net_spread_bps < self.min_spread_bps:
            return None

        # Calculate potential profit
        profit_per_unit = spread - buy_price * Decimal(str(total_fees_bps / 10000))

        # Estimate max size (limited by smaller volume)
        vol_a = price_a.volume_24h or Decimal("1000000")
        vol_b = price_b.volume_24h or Decimal("1000000")
        max_size = min(vol_a, vol_b) * Decimal("0.01")  # 1% of volume

        estimated_profit = profit_per_unit * max_size

        if estimated_profit < self.min_profit_threshold:
            return None

        # Assess risk
        risk = self._assess_cross_market_risk(price_a, price_b)

        opportunity = ArbitrageOpportunity(
            id=self._generate_opportunity_id(),
            arb_type=ArbitrageType.CROSS_MARKET,
            buy_market=buy_market,
            sell_market=sell_market,
            buy_price=buy_price,
            sell_price=sell_price,
            spread=spread,
            spread_bps=spread_bps,
            profit_potential=profit_per_unit,
            max_size=max_size,
            estimated_profit=estimated_profit,
            risk_level=risk,
            confidence=0.7,
            timestamp=datetime.now(),
            expiry=datetime.now() + timedelta(seconds=self.opportunity_ttl_seconds),
        )

        self._add_opportunity(opportunity)
        return opportunity

    def _assess_cross_market_risk(
        self,
        price_a: MarketPrice,
        price_b: MarketPrice,
    ) -> RiskLevel:
        """Assess risk of cross-market arbitrage."""
        risk_score = 0

        # Spread risk
        if price_a.spread_bps > 10 or price_b.spread_bps > 10:
            risk_score += 2

        # Volume risk
        min_vol = min(
            price_a.volume_24h or Decimal("0"),
            price_b.volume_24h or Decimal("0"),
        )
        if min_vol < Decimal("100000"):
            risk_score += 2
        elif min_vol < Decimal("1000000"):
            risk_score += 1

        # Staleness risk
        age_a = (datetime.now() - price_a.timestamp).total_seconds()
        age_b = (datetime.now() - price_b.timestamp).total_seconds()
        if max(age_a, age_b) > 5:
            risk_score += 2
        elif max(age_a, age_b) > 2:
            risk_score += 1

        if risk_score >= 5:
            return RiskLevel.VERY_HIGH
        elif risk_score >= 4:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MEDIUM
        elif risk_score >= 1:
            return RiskLevel.LOW
        return RiskLevel.VERY_LOW

    def detect_basis_trade(
        self,
        spot_market: str,
        perp_market: str,
    ) -> BasisTrade | None:
        """
        Detect basis trade opportunity.

        Args:
            spot_market: Spot market symbol
            perp_market: Perpetual market symbol

        Returns:
            Basis trade opportunity or None
        """
        spot_price = self._prices.get(spot_market)
        perp_price = self._prices.get(perp_market)

        if not spot_price or not perp_price:
            return None

        # Calculate basis
        basis = perp_price.mid - spot_price.mid
        basis_pct = float(basis / spot_price.mid * 100) if spot_price.mid > 0 else 0

        # Annualize (assuming continuous)
        annualized_yield = basis_pct * 365 / 7  # Assume 1 week convergence

        # Get funding rate
        funding_rate = self._funding_rates.get(perp_market, Decimal("0"))

        # Calculate net yield
        # If basis positive (perp > spot): short perp, long spot
        # Funding adds to yield if we're short and rate is positive
        funding_annual = float(funding_rate) * 3 * 365 * 100  # 3x daily, 365 days

        if basis > 0:
            recommended_direction = "long_spot_short_perp"
            net_yield = annualized_yield + funding_annual
        else:
            recommended_direction = "short_spot_long_perp"
            net_yield = abs(annualized_yield) - funding_annual

        return BasisTrade(
            spot_market=spot_market,
            perp_market=perp_market,
            spot_price=spot_price.mid,
            perp_price=perp_price.mid,
            basis=basis,
            basis_pct=basis_pct,
            annualized_yield=annualized_yield,
            funding_rate=funding_rate,
            net_yield=net_yield,
            recommended_direction=recommended_direction,
            timestamp=datetime.now(),
        )

    def detect_funding_arbitrage(
        self,
        markets: list[str] | None = None,
    ) -> FundingArbitrage | None:
        """
        Detect funding rate arbitrage.

        Args:
            markets: Markets to analyze (all if None)

        Returns:
            Funding arbitrage opportunity or None
        """
        if markets is None:
            markets = list(self._funding_rates.keys())

        if len(markets) < 2:
            return None

        # Find best long (lowest funding) and short (highest funding)
        funding_sorted = sorted(
            [(m, self._funding_rates.get(m, Decimal("0"))) for m in markets],
            key=lambda x: x[1],
        )

        long_market, long_funding = funding_sorted[0]
        short_market, short_funding = funding_sorted[-1]

        # Calculate spread
        funding_spread = short_funding - long_funding

        # Minimum spread threshold
        if funding_spread < Decimal("0.0001"):  # 0.01%
            return None

        # Annualize
        annualized_spread = float(funding_spread) * 3 * 365 * 100

        # Calculate recommended size and profit
        # This depends on available margin, but we estimate
        recommended_size = Decimal("1000")  # Example
        daily_rate = funding_spread * 3  # 3 funding periods per day
        estimated_daily_profit = recommended_size * daily_rate

        # Assess risk
        risk = RiskLevel.MEDIUM
        if annualized_spread > 50:  # >50% APY is suspicious
            risk = RiskLevel.HIGH
        elif annualized_spread < 10:
            risk = RiskLevel.LOW

        return FundingArbitrage(
            long_market=long_market,
            short_market=short_market,
            long_funding=long_funding,
            short_funding=short_funding,
            funding_spread=funding_spread,
            annualized_spread=annualized_spread,
            recommended_size=recommended_size,
            estimated_daily_profit=estimated_daily_profit,
            risk_level=risk,
            timestamp=datetime.now(),
        )

    def detect_all(
        self,
        markets: list[str] | None = None,
    ) -> list[ArbitrageOpportunity]:
        """
        Detect all arbitrage opportunities.

        Args:
            markets: Markets to analyze

        Returns:
            List of opportunities
        """
        if markets is None:
            markets = list(self._prices.keys())

        opportunities = []

        # Cross-market arbitrage
        for i, market_a in enumerate(markets):
            for market_b in markets[i + 1:]:
                opp = self.detect_cross_market(market_a, market_b)
                if opp:
                    opportunities.append(opp)

        return opportunities

    def _add_opportunity(self, opportunity: ArbitrageOpportunity) -> None:
        """Add opportunity and notify callbacks."""
        self._opportunities[opportunity.id] = opportunity

        # Trim if too many
        if len(self._opportunities) > self.max_opportunities:
            oldest = min(self._opportunities.values(), key=lambda x: x.timestamp)
            self._opportunities.pop(oldest.id, None)

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(opportunity)
            except Exception:
                pass

    def get_active_opportunities(self) -> list[ArbitrageOpportunity]:
        """Get active opportunities."""
        now = datetime.now()
        active = []

        for opp in self._opportunities.values():
            if opp.expiry > now and opp.status == OpportunityStatus.ACTIVE:
                active.append(opp)
            elif opp.expiry <= now and opp.status == OpportunityStatus.ACTIVE:
                opp.status = OpportunityStatus.EXPIRED
                self._history.append(opp)

        return active

    def get_opportunity(self, opp_id: str) -> ArbitrageOpportunity | None:
        """Get specific opportunity."""
        return self._opportunities.get(opp_id)

    def mark_executed(self, opp_id: str) -> bool:
        """Mark opportunity as executed."""
        opp = self._opportunities.get(opp_id)
        if opp:
            opp.status = OpportunityStatus.EXECUTED
            self._history.append(opp)
            self._opportunities.pop(opp_id, None)
            return True
        return False

    def get_summary(self) -> ArbitrageSummary:
        """Get summary of opportunities."""
        active = self.get_active_opportunities()

        by_type: dict[ArbitrageType, int] = {}
        for arb_type in ArbitrageType:
            by_type[arb_type] = 0

        total_profit = Decimal("0")
        total_spread = 0.0

        for opp in active:
            by_type[opp.arb_type] = by_type.get(opp.arb_type, 0) + 1
            total_profit += opp.estimated_profit
            total_spread += opp.spread_bps

        best = max(active, key=lambda x: x.estimated_profit) if active else None
        avg_spread = total_spread / len(active) if active else 0.0

        return ArbitrageSummary(
            timestamp=datetime.now(),
            total_opportunities=len(active),
            by_type=by_type,
            best_opportunity=best,
            total_potential_profit=total_profit,
            avg_spread_bps=avg_spread,
            markets_analyzed=len(self._prices),
        )

    def get_best_opportunity(
        self,
        arb_type: ArbitrageType | None = None,
    ) -> ArbitrageOpportunity | None:
        """
        Get best opportunity by estimated profit.

        Args:
            arb_type: Filter by type

        Returns:
            Best opportunity or None
        """
        active = self.get_active_opportunities()

        if arb_type:
            active = [o for o in active if o.arb_type == arb_type]

        if not active:
            return None

        return max(active, key=lambda x: x.estimated_profit)

    def add_callback(self, callback: Callable[[ArbitrageOpportunity], None]) -> None:
        """Add opportunity callback."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[ArbitrageOpportunity], None]) -> bool:
        """Remove callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            return True
        return False

    def get_history(
        self,
        limit: int | None = None,
        status: OpportunityStatus | None = None,
    ) -> list[ArbitrageOpportunity]:
        """
        Get opportunity history.

        Args:
            limit: Maximum results
            status: Filter by status

        Returns:
            List of historical opportunities
        """
        results = self._history

        if status:
            results = [o for o in results if o.status == status]

        if limit:
            results = results[-limit:]

        return results

    def get_markets(self) -> list[str]:
        """Get list of markets with prices."""
        return list(self._prices.keys())

    def clear_market(self, market: str) -> None:
        """Clear price data for a market."""
        self._prices.pop(market, None)
        self._funding_rates.pop(market, None)

    def clear_all(self) -> None:
        """Clear all data."""
        self._prices.clear()
        self._funding_rates.clear()
        self._opportunities.clear()


# Global instance
_detector: ArbitrageDetector | None = None


def get_arbitrage_detector() -> ArbitrageDetector:
    """Get global arbitrage detector."""
    global _detector
    if _detector is None:
        _detector = ArbitrageDetector()
    return _detector


def reset_arbitrage_detector() -> None:
    """Reset global detector."""
    global _detector
    _detector = None
