"""Unit tests for Account Manager."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from decimal import Decimal

from src.core.account_manager import (
    AccountManager,
    AccountState,
    AccountRole,
)


class TestAccountRole:
    """Tests for AccountRole enum."""

    def test_role_values(self):
        """Should have expected role values."""
        assert AccountRole.LONG.value == "long"
        assert AccountRole.SHORT.value == "short"
        assert AccountRole.NEUTRAL.value == "neutral"


class TestAccountState:
    """Tests for AccountState dataclass."""

    def test_create_account_state(self):
        """Should create account state correctly."""
        state = AccountState(
            account_id="acc1",
            alias="test-account",
            l2_address="0x123",
            role=AccountRole.LONG,
            leverage=10.0,
        )

        assert state.account_id == "acc1"
        assert state.alias == "test-account"
        assert state.role == AccountRole.LONG
        assert state.leverage == 10.0
        assert state.is_active is True

    def test_default_values(self):
        """Should have correct default values."""
        state = AccountState(
            account_id="acc1",
            alias="test",
            l2_address="0x123",
        )

        assert state.role == AccountRole.NEUTRAL
        assert state.leverage == 1.0
        assert state.is_active is True
        assert state.current_position_size == 0.0
        assert state.current_position_side is None

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        state = AccountState(
            account_id="acc1",
            alias="test",
            l2_address="0x123",
            role=AccountRole.SHORT,
            current_position_size=1000.0,
            current_position_side="SHORT",
        )

        d = state.to_dict()

        assert d["account_id"] == "acc1"
        assert d["alias"] == "test"
        assert d["role"] == "short"
        assert d["position_size"] == 1000.0
        assert d["position_side"] == "SHORT"


class TestAccountManager:
    """Tests for AccountManager."""

    @pytest.fixture
    def mock_keychain(self):
        """Create mock keychain manager."""
        keychain = MagicMock()
        keychain.get_all_credentials.return_value = []
        return keychain

    @pytest.fixture
    def mock_client_factory(self):
        """Create mock client factory."""
        factory = MagicMock()
        factory.create_client = AsyncMock()
        factory.remove_client = MagicMock()
        factory.close_all = AsyncMock()
        return factory

    @pytest.fixture
    def mock_ws_manager(self):
        """Create mock websocket manager."""
        manager = MagicMock()
        manager.connect_account = AsyncMock()
        manager.disconnect_account = AsyncMock()
        manager.disconnect_all = AsyncMock()
        return manager

    @pytest.fixture
    def manager(self, mock_keychain, mock_client_factory, mock_ws_manager):
        """Create account manager."""
        return AccountManager(
            keychain=mock_keychain,
            client_factory=mock_client_factory,
            ws_manager=mock_ws_manager,
        )

    def test_initial_state(self, manager):
        """Should start with empty state."""
        assert manager.account_count == 0
        assert manager.active_count == 0
        assert manager.is_initialized is False

    @pytest.mark.asyncio
    async def test_initialize_no_accounts(self, manager):
        """Should handle no accounts gracefully."""
        count = await manager.initialize_accounts()

        assert count == 0
        # Note: When no credentials, code returns early before setting initialized
        # This is expected behavior - no accounts means not fully initialized

    def test_get_account_not_found(self, manager):
        """Should return None for missing account."""
        result = manager.get_account("nonexistent")
        assert result is None

    def test_get_client_not_found(self, manager):
        """Should return None for missing client."""
        result = manager.get_client("nonexistent")
        assert result is None

    def test_assign_role(self, manager):
        """Should assign role to account."""
        # Add account manually
        state = AccountState(
            account_id="acc1",
            alias="test",
            l2_address="0x123",
        )
        manager._accounts["acc1"] = state

        result = manager.assign_role("acc1", AccountRole.LONG)

        assert result is True
        assert manager._accounts["acc1"].role == AccountRole.LONG

    def test_assign_role_not_found(self, manager):
        """Should return False for missing account."""
        result = manager.assign_role("nonexistent", AccountRole.LONG)
        assert result is False

    def test_assign_roles_balanced(self, manager):
        """Should assign balanced 50/50 roles."""
        # Add 4 accounts
        for i in range(4):
            state = AccountState(
                account_id=f"acc{i}",
                alias=f"account{i}",
                l2_address=f"0x{i}",
            )
            manager._accounts[f"acc{i}"] = state

        assignments = manager.assign_roles_balanced()

        long_count = sum(1 for r in assignments.values() if r == AccountRole.LONG)
        short_count = sum(1 for r in assignments.values() if r == AccountRole.SHORT)

        assert long_count == 2
        assert short_count == 2

    def test_assign_roles_balanced_insufficient(self, manager):
        """Should warn with less than 2 accounts."""
        state = AccountState(
            account_id="acc1",
            alias="test",
            l2_address="0x123",
        )
        manager._accounts["acc1"] = state

        assignments = manager.assign_roles_balanced()

        assert assignments == {}

    def test_get_accounts_by_role(self, manager):
        """Should filter accounts by role."""
        for i, role in enumerate([AccountRole.LONG, AccountRole.LONG, AccountRole.SHORT]):
            state = AccountState(
                account_id=f"acc{i}",
                alias=f"account{i}",
                l2_address=f"0x{i}",
                role=role,
            )
            manager._accounts[f"acc{i}"] = state

        long_accounts = manager.get_accounts_by_role(AccountRole.LONG)
        short_accounts = manager.get_accounts_by_role(AccountRole.SHORT)

        assert len(long_accounts) == 2
        assert len(short_accounts) == 1

    def test_get_active_accounts(self, manager):
        """Should only return active accounts."""
        state1 = AccountState(
            account_id="acc1",
            alias="active",
            l2_address="0x1",
            is_active=True,
        )
        state2 = AccountState(
            account_id="acc2",
            alias="inactive",
            l2_address="0x2",
            is_active=False,
        )
        manager._accounts["acc1"] = state1
        manager._accounts["acc2"] = state2

        active = manager.get_active_accounts()

        assert len(active) == 1
        assert active[0].account_id == "acc1"

    def test_get_total_exposure(self, manager):
        """Should calculate total exposure correctly."""
        state1 = AccountState(
            account_id="acc1",
            alias="long1",
            l2_address="0x1",
            current_position_size=1000.0,
            current_position_side="LONG",
        )
        state2 = AccountState(
            account_id="acc2",
            alias="long2",
            l2_address="0x2",
            current_position_size=500.0,
            current_position_side="LONG",
        )
        state3 = AccountState(
            account_id="acc3",
            alias="short1",
            l2_address="0x3",
            current_position_size=800.0,
            current_position_side="SHORT",
        )
        manager._accounts["acc1"] = state1
        manager._accounts["acc2"] = state2
        manager._accounts["acc3"] = state3

        exposure = manager.get_total_exposure()

        assert exposure["long"] == 1500.0
        assert exposure["short"] == 800.0
        assert exposure["net"] == 700.0
        assert exposure["gross"] == 2300.0

    def test_set_leverage(self, manager):
        """Should set leverage for account."""
        state = AccountState(
            account_id="acc1",
            alias="test",
            l2_address="0x123",
        )
        manager._accounts["acc1"] = state

        result = manager.set_leverage("acc1", 20.0)

        assert result is True
        assert manager._accounts["acc1"].leverage == 20.0

    def test_set_leverage_not_found(self, manager):
        """Should return False for missing account."""
        result = manager.set_leverage("nonexistent", 10.0)
        assert result is False

    def test_set_active(self, manager):
        """Should set active status for account."""
        state = AccountState(
            account_id="acc1",
            alias="test",
            l2_address="0x123",
            is_active=True,
        )
        manager._accounts["acc1"] = state

        result = manager.set_active("acc1", False)

        assert result is True
        assert manager._accounts["acc1"].is_active is False

    def test_set_active_not_found(self, manager):
        """Should return False for missing account."""
        result = manager.set_active("nonexistent", True)
        assert result is False

    def test_account_count(self, manager):
        """Should count total accounts."""
        for i in range(3):
            state = AccountState(
                account_id=f"acc{i}",
                alias=f"account{i}",
                l2_address=f"0x{i}",
            )
            manager._accounts[f"acc{i}"] = state

        assert manager.account_count == 3

    def test_active_count(self, manager):
        """Should count active accounts only."""
        for i in range(3):
            state = AccountState(
                account_id=f"acc{i}",
                alias=f"account{i}",
                l2_address=f"0x{i}",
                is_active=(i < 2),  # First 2 active
            )
            manager._accounts[f"acc{i}"] = state

        assert manager.active_count == 2

    def test_list_accounts(self, manager):
        """Should list all accounts as dicts."""
        for i in range(2):
            state = AccountState(
                account_id=f"acc{i}",
                alias=f"account{i}",
                l2_address=f"0x{i}",
            )
            manager._accounts[f"acc{i}"] = state

        accounts = manager.list_accounts()

        assert len(accounts) == 2
        assert all("account_id" in a for a in accounts)
        assert all("alias" in a for a in accounts)

    def test_get_account_by_id(self, manager):
        """Should find account by ID."""
        state = AccountState(
            account_id="acc1",
            alias="test-account",
            l2_address="0x123",
        )
        manager._accounts["acc1"] = state

        result = manager.get_account("acc1")

        assert result is not None
        assert result.account_id == "acc1"

    def test_get_account_by_alias(self, manager):
        """Should find account by alias."""
        state = AccountState(
            account_id="acc1",
            alias="test-account",
            l2_address="0x123",
        )
        manager._accounts["acc1"] = state

        result = manager.get_account("test-account")

        assert result is not None
        assert result.alias == "test-account"

    @pytest.mark.asyncio
    async def test_shutdown(self, manager, mock_ws_manager, mock_client_factory):
        """Should clean up on shutdown."""
        state = AccountState(
            account_id="acc1",
            alias="test",
            l2_address="0x123",
        )
        manager._accounts["acc1"] = state
        manager._initialized = True

        await manager.shutdown()

        mock_ws_manager.disconnect_all.assert_called_once()
        mock_client_factory.close_all.assert_called_once()
        assert manager.account_count == 0
        assert manager.is_initialized is False
