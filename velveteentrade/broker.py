"""Broker abstraction. Alpaca today; IBKR or any other broker is a subclass away.

The pipeline only ever talks to `BrokerAdapter`. Nothing above this layer knows
which broker is underneath — that is what makes the XTB lesson (a broker killing
its API overnight) survivable.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry: float
    market_value: float


@dataclass
class AccountState:
    equity: float
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)


@dataclass
class OpenOrder:
    id: str
    symbol: str
    qty: float
    side: str  # 'buy' | 'sell'


class BrokerAdapter(ABC):
    @abstractmethod
    def account(self) -> AccountState: ...

    @abstractmethod
    def submit_order(self, symbol: str, qty: float, side: str) -> str:
        """Submit a market order. Returns order id. side: 'buy' | 'sell'."""

    @abstractmethod
    def open_orders(self) -> list[OpenOrder]:
        """Orders submitted but not yet filled (e.g. queued while market closed)."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> None: ...

    @abstractmethod
    def market_open_soon(self) -> bool:
        """True if the market is open or opens within the next session."""


class AlpacaBroker(BrokerAdapter):
    def __init__(self, api_key: str, secret_key: str, paper: bool = True) -> None:
        from alpaca.trading.client import TradingClient

        if not paper:
            log.warning("LIVE trading client requested — make sure this is intentional.")
        self._client = TradingClient(api_key, secret_key, paper=paper)

    def account(self) -> AccountState:
        acct = self._client.get_account()
        positions = {}
        for p in self._client.get_all_positions():
            positions[p.symbol] = Position(
                symbol=p.symbol,
                qty=float(p.qty),
                avg_entry=float(p.avg_entry_price),
                market_value=float(p.market_value),
            )
        return AccountState(equity=float(acct.equity), cash=float(acct.cash), positions=positions)

    def submit_order(self, symbol: str, qty: float, side: str) -> str:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = self._client.submit_order(req)
        log.info("Order submitted: %s %s x%s -> id=%s", side, symbol, qty, order.id)
        return str(order.id)

    def open_orders(self) -> list[OpenOrder]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        orders = self._client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
        return [
            OpenOrder(id=str(o.id), symbol=o.symbol, qty=float(o.qty or 0),
                      side=str(o.side.value if hasattr(o.side, "value") else o.side).lower())
            for o in orders
        ]

    def cancel_order(self, order_id: str) -> None:
        self._client.cancel_order_by_id(order_id)
        log.info("Order cancelled: %s", order_id)

    def market_open_soon(self) -> bool:
        clock = self._client.get_clock()
        return bool(clock.is_open) or clock.next_open is not None


class DryRunBroker(BrokerAdapter):
    """Wraps a real (or mock) account snapshot but never sends orders."""

    def __init__(self, inner: BrokerAdapter) -> None:
        self._inner = inner
        self.intended_orders: list[tuple[str, float, str]] = []

    def account(self) -> AccountState:
        return self._inner.account()

    def submit_order(self, symbol: str, qty: float, side: str) -> str:
        self.intended_orders.append((symbol, qty, side))
        log.info("[DRY RUN] would submit: %s %s x%s", side, symbol, qty)
        return f"dry-{len(self.intended_orders)}"

    def open_orders(self) -> list[OpenOrder]:
        return self._inner.open_orders()

    def cancel_order(self, order_id: str) -> None:
        log.info("[DRY RUN] would cancel: %s", order_id)

    def market_open_soon(self) -> bool:
        return self._inner.market_open_soon()


class MockBroker(BrokerAdapter):
    """In-memory broker for tests and offline development."""

    def __init__(self, equity: float = 100_000.0, cash: float | None = None) -> None:
        self.state = AccountState(equity=equity, cash=cash if cash is not None else equity)
        self.orders: list[tuple[str, float, str]] = []
        self.pending: list[OpenOrder] = []
        self.cancelled: list[str] = []

    def account(self) -> AccountState:
        return self.state

    def submit_order(self, symbol: str, qty: float, side: str) -> str:
        self.orders.append((symbol, qty, side))
        return f"mock-{len(self.orders)}"

    def open_orders(self) -> list[OpenOrder]:
        return list(self.pending)

    def cancel_order(self, order_id: str) -> None:
        self.cancelled.append(order_id)
        self.pending = [o for o in self.pending if o.id != order_id]

    def market_open_soon(self) -> bool:
        return True
