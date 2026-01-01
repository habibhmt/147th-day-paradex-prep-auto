"""Unit tests for KeychainManager."""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.security.keychain import (
    KeychainManager,
    AccountCredentials,
    SERVICE_NAME,
    ACCOUNTS_INDEX_KEY,
)


class TestAccountCredentials:
    """Tests for AccountCredentials dataclass."""

    def test_create_credentials(self):
        """Should create credentials correctly."""
        creds = AccountCredentials(
            account_id="acc1",
            alias="main",
            l2_address="0x123",
            l2_private_key="0xprivate",
        )

        assert creds.account_id == "acc1"
        assert creds.alias == "main"
        assert creds.l2_address == "0x123"
        assert creds.l2_private_key == "0xprivate"

    def test_to_dict_excludes_private_key(self):
        """to_dict should not include private key for security."""
        creds = AccountCredentials(
            account_id="acc1",
            alias="main",
            l2_address="0x123",
            l2_private_key="0xprivate",
        )

        d = creds.to_dict()

        assert "account_id" in d
        assert "alias" in d
        assert "l2_address" in d
        assert "l2_private_key" not in d


class TestKeychainManager:
    """Tests for KeychainManager class."""

    @pytest.fixture
    def mock_keyring(self):
        """Mock keyring module."""
        with patch("src.security.keychain.keyring") as mock:
            # Make verification pass
            mock.set_password = MagicMock()
            mock.delete_password = MagicMock()
            mock.get_password = MagicMock(return_value=None)
            yield mock

    @pytest.fixture
    def keychain(self, mock_keyring):
        """Create KeychainManager with mocked keyring."""
        return KeychainManager()

    def test_init_verifies_keychain(self, mock_keyring):
        """Should verify keychain is available on init."""
        KeychainManager()

        # Verification writes and deletes test key
        mock_keyring.set_password.assert_called()
        mock_keyring.delete_password.assert_called()

    def test_store_account(self, keychain, mock_keyring):
        """Should store account in keychain."""
        creds = AccountCredentials(
            account_id="acc1",
            alias="main",
            l2_address="0x123",
            l2_private_key="0xprivate",
        )

        keychain.store_account(creds)

        # Should store credentials
        mock_keyring.set_password.assert_called()
        call_args = mock_keyring.set_password.call_args_list

        # Find the account storage call (not test or index)
        account_calls = [c for c in call_args if "account.acc1" in str(c)]
        assert len(account_calls) > 0

    def test_get_account_returns_credentials(self, keychain, mock_keyring):
        """Should retrieve stored credentials."""
        stored_data = json.dumps({
            "account_id": "acc1",
            "alias": "main",
            "l2_address": "0x123",
            "l2_private_key": "0xprivate",
        })
        mock_keyring.get_password.return_value = stored_data

        creds = keychain.get_account("acc1")

        assert creds is not None
        assert creds.account_id == "acc1"
        assert creds.alias == "main"
        assert creds.l2_private_key == "0xprivate"

    def test_get_account_returns_none_for_missing(self, keychain, mock_keyring):
        """Should return None for missing account."""
        mock_keyring.get_password.return_value = None

        creds = keychain.get_account("nonexistent")

        assert creds is None

    def test_list_accounts(self, keychain, mock_keyring):
        """Should list all account IDs."""
        mock_keyring.get_password.return_value = json.dumps(["acc1", "acc2"])

        accounts = keychain.list_accounts()

        assert accounts == ["acc1", "acc2"]

    def test_delete_account(self, keychain, mock_keyring):
        """Should delete account from keychain."""
        mock_keyring.get_password.return_value = json.dumps(["acc1", "acc2"])

        result = keychain.delete_account("acc1")

        assert result is True
        mock_keyring.delete_password.assert_called()

    def test_get_account_by_alias(self, keychain, mock_keyring):
        """Should find account by alias."""
        def get_password_side_effect(service, key):
            if key == ACCOUNTS_INDEX_KEY:
                return json.dumps(["acc1", "acc2"])
            elif "acc1" in key:
                return json.dumps({
                    "account_id": "acc1",
                    "alias": "main",
                    "l2_address": "0x123",
                    "l2_private_key": "0xprivate1",
                })
            elif "acc2" in key:
                return json.dumps({
                    "account_id": "acc2",
                    "alias": "secondary",
                    "l2_address": "0x456",
                    "l2_private_key": "0xprivate2",
                })
            return None

        mock_keyring.get_password.side_effect = get_password_side_effect

        creds = keychain.get_account_by_alias("secondary")

        assert creds is not None
        assert creds.account_id == "acc2"
        assert creds.alias == "secondary"

    def test_rotate_subkey(self, keychain, mock_keyring):
        """Should rotate subkey for account."""
        def get_password_side_effect(service, key):
            if key == ACCOUNTS_INDEX_KEY:
                return json.dumps(["acc1"])
            elif "acc1" in key:
                return json.dumps({
                    "account_id": "acc1",
                    "alias": "main",
                    "l2_address": "0x123",
                    "l2_private_key": "0xold",
                })
            return None

        mock_keyring.get_password.side_effect = get_password_side_effect

        result = keychain.rotate_subkey("acc1", "0xnew")

        assert result is True

    def test_rotate_subkey_returns_false_for_missing(self, keychain, mock_keyring):
        """Should return False when account doesn't exist."""
        mock_keyring.get_password.return_value = None

        result = keychain.rotate_subkey("nonexistent", "0xnew")

        assert result is False

    def test_get_all_credentials(self, keychain, mock_keyring):
        """Should get all credentials."""
        def get_password_side_effect(service, key):
            if key == ACCOUNTS_INDEX_KEY:
                return json.dumps(["acc1", "acc2"])
            elif "acc1" in key:
                return json.dumps({
                    "account_id": "acc1",
                    "alias": "main",
                    "l2_address": "0x123",
                    "l2_private_key": "0xprivate1",
                })
            elif "acc2" in key:
                return json.dumps({
                    "account_id": "acc2",
                    "alias": "secondary",
                    "l2_address": "0x456",
                    "l2_private_key": "0xprivate2",
                })
            return None

        mock_keyring.get_password.side_effect = get_password_side_effect

        credentials = keychain.get_all_credentials()

        assert len(credentials) == 2
        assert credentials[0].account_id == "acc1"
        assert credentials[1].account_id == "acc2"
