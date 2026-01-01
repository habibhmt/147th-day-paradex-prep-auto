"""Multi-account orchestration and management."""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from paradex_py import ParadexSubkey

from src.network.client_factory import ParadexClientFactory
from src.network.websocket_manager import WebSocketManager
from src.security.keychain import AccountCredentials, KeychainManager

logger = logging.getLogger(__name__)


class AccountRole(Enum):
    """Role assigned to an account in delta-neutral strategy."""

    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"  # Unassigned / dynamic


@dataclass
class AccountState:
    """Current state of a trading account."""

    account_id: str
    alias: str
    l2_address: str
    role: AccountRole = AccountRole.NEUTRAL
    leverage: float = 1.0
    is_active: bool = True

    # Position state
    current_position_size: float = 0.0
    current_position_side: Optional[str] = None  # "LONG" or "SHORT"
    target_position_size: float = 0.0

    # Timing
    last_trade_time: float = 0.0
    last_sync_time: float = 0.0

    # Balance
    available_balance: float = 0.0
    total_equity: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for display."""
        return {
            "account_id": self.account_id,
            "alias": self.alias,
            "role": self.role.value,
            "leverage": self.leverage,
            "is_active": self.is_active,
            "position_size": self.current_position_size,
            "position_side": self.current_position_side,
            "available_balance": self.available_balance,
        }


@dataclass
class AccountManager:
    """Orchestrates multiple trading accounts.

    Manages:
    - Account initialization and lifecycle
    - Role assignment (long/short)
    - Position synchronization
    - Client and WebSocket coordination
    """

    keychain: KeychainManager
    client_factory: ParadexClientFactory
    ws_manager: WebSocketManager

    _accounts: Dict[str, AccountState] = field(default_factory=dict)
    _clients: Dict[str, ParadexSubkey] = field(default_factory=dict)
    _initialized: bool = False

    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._accounts = {}
        self._clients = {}
        self._initialized = False

    async def initialize_accounts(self) -> int:
        """Load and initialize all configured accounts from keychain.

        Returns:
            Number of accounts initialized
        """
        credentials = self.keychain.get_all_credentials()

        if not credentials:
            logger.warning("No accounts found in keychain")
            return 0

        # Initialize concurrently
        tasks = [
            self._initialize_single_account(cred)
            for cred in credentials
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if r is True)
        logger.info(f"Initialized {success_count}/{len(credentials)} accounts")

        self._initialized = True
        return success_count

    async def _initialize_single_account(
        self,
        credentials: AccountCredentials,
    ) -> bool:
        """Initialize a single account.

        Args:
            credentials: Account credentials

        Returns:
            True if successful
        """
        try:
            # Create client
            client = await self.client_factory.create_client(credentials)
            self._clients[credentials.account_id] = client

            # Create account state
            state = AccountState(
                account_id=credentials.account_id,
                alias=credentials.alias,
                l2_address=credentials.l2_address,
            )
            self._accounts[credentials.account_id] = state

            # Sync initial state
            await self._sync_account_state(credentials.account_id)

            # Connect WebSocket
            await self.ws_manager.connect_account(
                credentials.account_id,
                client,
            )

            logger.info(f"Initialized account: {credentials.alias}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize {credentials.alias}: {e}")
            return False

    async def add_account(
        self,
        alias: str,
        l2_address: str,
        l2_private_key: str,
        role: AccountRole = AccountRole.NEUTRAL,
        leverage: float = 1.0,
    ) -> Optional[AccountState]:
        """Add and initialize a new trading account.

        Args:
            alias: Human-readable name
            l2_address: Starknet address
            l2_private_key: Subkey private key
            role: Initial role assignment
            leverage: Default leverage

        Returns:
            AccountState if successful, None otherwise
        """
        from src.security.credentials import CredentialManager

        cred_manager = CredentialManager(self.keychain)

        try:
            # Store credentials
            credentials = cred_manager.add_account(
                alias=alias,
                l2_address=l2_address,
                l2_private_key=l2_private_key,
            )

            # Create client
            client = await self.client_factory.create_client(credentials)
            self._clients[credentials.account_id] = client

            # Create state
            state = AccountState(
                account_id=credentials.account_id,
                alias=alias,
                l2_address=l2_address,
                role=role,
                leverage=leverage,
            )
            self._accounts[credentials.account_id] = state

            # Sync and connect
            await self._sync_account_state(credentials.account_id)
            await self.ws_manager.connect_account(credentials.account_id, client)

            logger.info(f"Added account: {alias}")
            return state

        except Exception as e:
            logger.error(f"Failed to add account {alias}: {e}")
            return None

    async def remove_account(self, alias_or_id: str) -> bool:
        """Remove an account.

        Args:
            alias_or_id: Account alias or ID

        Returns:
            True if removed
        """
        # Find account
        account_id = None
        for aid, state in self._accounts.items():
            if aid == alias_or_id or state.alias == alias_or_id:
                account_id = aid
                break

        if not account_id:
            return False

        # Disconnect
        await self.ws_manager.disconnect_account(account_id)
        self.client_factory.remove_client(account_id)

        # Remove from tracking
        del self._accounts[account_id]
        if account_id in self._clients:
            del self._clients[account_id]

        # Remove from keychain
        self.keychain.delete_account(account_id)

        logger.info(f"Removed account: {alias_or_id}")
        return True

    async def _sync_account_state(self, account_id: str) -> None:
        """Sync account state from exchange.

        Args:
            account_id: Account to sync
        """
        if account_id not in self._clients:
            return

        client = self._clients[account_id]
        state = self._accounts[account_id]

        try:
            # Fetch account summary
            summary = await client.api_client.fetch_account_summary()
            if summary:
                state.available_balance = float(summary.get("free_collateral", 0))
                state.total_equity = float(summary.get("account_value", 0))

            # Fetch positions
            positions = await client.api_client.fetch_positions()
            if positions:
                # Sum up positions (simplified - assumes single market)
                for pos in positions:
                    if float(pos.get("size", 0)) != 0:
                        state.current_position_size = abs(float(pos.get("size", 0)))
                        state.current_position_side = pos.get("side", "").upper()
                        break
                else:
                    state.current_position_size = 0.0
                    state.current_position_side = None

            import time
            state.last_sync_time = time.time()

        except Exception as e:
            logger.error(f"Failed to sync account {account_id}: {e}")

    async def sync_all_positions(self) -> None:
        """Sync positions for all accounts."""
        tasks = [
            self._sync_account_state(account_id)
            for account_id in self._accounts.keys()
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.debug("Synced all account positions")

    def get_account(self, alias_or_id: str) -> Optional[AccountState]:
        """Get account state by alias or ID.

        Args:
            alias_or_id: Account identifier

        Returns:
            AccountState if found
        """
        # Try direct ID lookup
        if alias_or_id in self._accounts:
            return self._accounts[alias_or_id]

        # Try alias lookup
        for state in self._accounts.values():
            if state.alias == alias_or_id:
                return state

        return None

    def get_client(self, account_id: str) -> Optional[ParadexSubkey]:
        """Get client for account.

        Args:
            account_id: Account ID

        Returns:
            ParadexSubkey client if exists
        """
        return self._clients.get(account_id)

    def get_accounts_by_role(self, role: AccountRole) -> List[AccountState]:
        """Get all accounts with specified role.

        Args:
            role: Role to filter by

        Returns:
            List of matching accounts
        """
        return [
            state for state in self._accounts.values()
            if state.role == role and state.is_active
        ]

    def get_active_accounts(self) -> List[AccountState]:
        """Get all active accounts.

        Returns:
            List of active account states
        """
        return [
            state for state in self._accounts.values()
            if state.is_active
        ]

    def get_total_exposure(self) -> Dict[str, float]:
        """Calculate total long/short exposure across accounts.

        Returns:
            Dictionary with 'long', 'short', and 'net' exposure
        """
        long_exposure = 0.0
        short_exposure = 0.0

        for state in self._accounts.values():
            if not state.is_active:
                continue

            if state.current_position_side == "LONG":
                long_exposure += state.current_position_size
            elif state.current_position_side == "SHORT":
                short_exposure += state.current_position_size

        return {
            "long": long_exposure,
            "short": short_exposure,
            "net": long_exposure - short_exposure,
            "gross": long_exposure + short_exposure,
        }

    def assign_role(self, account_id: str, role: AccountRole) -> bool:
        """Assign role to account.

        Args:
            account_id: Account to update
            role: New role

        Returns:
            True if successful
        """
        if account_id not in self._accounts:
            return False

        self._accounts[account_id].role = role
        logger.info(f"Assigned role {role.value} to {self._accounts[account_id].alias}")
        return True

    def assign_roles_balanced(self) -> Dict[str, AccountRole]:
        """Assign roles to achieve 50/50 split.

        Returns:
            Mapping of account_id to assigned role
        """
        active = self.get_active_accounts()
        n = len(active)

        if n < 2:
            logger.warning("Need at least 2 accounts for balanced roles")
            return {}

        n_long = n // 2
        assignments = {}

        for i, state in enumerate(active):
            role = AccountRole.LONG if i < n_long else AccountRole.SHORT
            state.role = role
            assignments[state.account_id] = role

        logger.info(f"Assigned balanced roles: {n_long} long, {n - n_long} short")
        return assignments

    def set_leverage(self, account_id: str, leverage: float) -> bool:
        """Set leverage for account.

        Args:
            account_id: Account to update
            leverage: New leverage value

        Returns:
            True if successful
        """
        if account_id not in self._accounts:
            return False

        self._accounts[account_id].leverage = leverage
        return True

    def set_active(self, account_id: str, active: bool) -> bool:
        """Set account active/inactive.

        Args:
            account_id: Account to update
            active: Active status

        Returns:
            True if successful
        """
        if account_id not in self._accounts:
            return False

        self._accounts[account_id].is_active = active
        return True

    @property
    def account_count(self) -> int:
        """Get total number of accounts."""
        return len(self._accounts)

    @property
    def active_count(self) -> int:
        """Get number of active accounts."""
        return len(self.get_active_accounts())

    @property
    def is_initialized(self) -> bool:
        """Check if manager is initialized."""
        return self._initialized

    def list_accounts(self) -> List[dict]:
        """Get list of all accounts as dictionaries.

        Returns:
            List of account info dicts
        """
        return [state.to_dict() for state in self._accounts.values()]

    async def shutdown(self) -> None:
        """Gracefully shutdown all accounts."""
        await self.ws_manager.disconnect_all()
        await self.client_factory.close_all()
        self._accounts.clear()
        self._clients.clear()
        self._initialized = False
        logger.info("Account manager shut down")
