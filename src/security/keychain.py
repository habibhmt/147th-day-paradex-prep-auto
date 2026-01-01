"""macOS Keychain integration for secure credential storage."""

import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional

import keyring
from keyring.errors import KeyringError

logger = logging.getLogger(__name__)

SERVICE_NAME = "paradex-delta-bot"
ACCOUNTS_INDEX_KEY = "__accounts_index__"


@dataclass
class AccountCredentials:
    """Account credentials stored in keychain."""

    account_id: str
    alias: str
    l2_address: str
    l2_private_key: str  # Subkey only - no withdrawal permissions

    def to_dict(self) -> dict:
        """Convert to dictionary (excluding private key for logging)."""
        return {
            "account_id": self.account_id,
            "alias": self.alias,
            "l2_address": self.l2_address,
        }


class KeychainManager:
    """Manages secure credential storage in macOS Keychain.

    Uses Subkeys only - these have trading permissions but cannot:
    - Withdraw funds
    - Transfer funds
    - Modify account settings

    This provides security even if credentials are compromised.
    """

    def __init__(self, service_name: str = SERVICE_NAME):
        """Initialize keychain manager.

        Args:
            service_name: Service name for keychain entries
        """
        self.service_name = service_name
        self._verify_keychain_available()

    def _verify_keychain_available(self) -> None:
        """Verify keychain backend is available."""
        try:
            # Test write and delete
            test_key = f"{self.service_name}.__test__"
            keyring.set_password(self.service_name, test_key, "test")
            keyring.delete_password(self.service_name, test_key)
            logger.debug("Keychain backend verified")
        except KeyringError as e:
            raise RuntimeError(f"Keychain not available: {e}") from e

    def _get_account_key(self, account_id: str) -> str:
        """Get keychain key for account."""
        return f"account.{account_id}"

    def _get_accounts_index(self) -> list[str]:
        """Get list of all account IDs."""
        try:
            data = keyring.get_password(self.service_name, ACCOUNTS_INDEX_KEY)
            if data:
                return json.loads(data)
        except (KeyringError, json.JSONDecodeError):
            pass
        return []

    def _update_accounts_index(self, account_ids: list[str]) -> None:
        """Update accounts index in keychain."""
        keyring.set_password(
            self.service_name,
            ACCOUNTS_INDEX_KEY,
            json.dumps(account_ids),
        )

    def store_account(self, creds: AccountCredentials) -> None:
        """Store account credentials in keychain.

        Args:
            creds: Account credentials to store
        """
        key = self._get_account_key(creds.account_id)

        # Store as JSON (private key included)
        data = json.dumps(asdict(creds))
        keyring.set_password(self.service_name, key, data)

        # Update index
        accounts = self._get_accounts_index()
        if creds.account_id not in accounts:
            accounts.append(creds.account_id)
            self._update_accounts_index(accounts)

        logger.info(f"Stored credentials for account: {creds.alias} ({creds.account_id})")

    def get_account(self, account_id: str) -> Optional[AccountCredentials]:
        """Retrieve account credentials from keychain.

        Args:
            account_id: Account ID to retrieve

        Returns:
            AccountCredentials if found, None otherwise
        """
        key = self._get_account_key(account_id)

        try:
            data = keyring.get_password(self.service_name, key)
            if data:
                creds_dict = json.loads(data)
                return AccountCredentials(**creds_dict)
        except (KeyringError, json.JSONDecodeError) as e:
            logger.error(f"Failed to retrieve account {account_id}: {e}")

        return None

    def get_account_by_alias(self, alias: str) -> Optional[AccountCredentials]:
        """Retrieve account by alias.

        Args:
            alias: Account alias

        Returns:
            AccountCredentials if found, None otherwise
        """
        for account_id in self._get_accounts_index():
            creds = self.get_account(account_id)
            if creds and creds.alias == alias:
                return creds
        return None

    def list_accounts(self) -> list[str]:
        """List all stored account IDs.

        Returns:
            List of account IDs
        """
        return self._get_accounts_index()

    def list_accounts_info(self) -> list[dict]:
        """List all accounts with their info (no private keys).

        Returns:
            List of account info dictionaries
        """
        accounts_info = []
        for account_id in self._get_accounts_index():
            creds = self.get_account(account_id)
            if creds:
                accounts_info.append(creds.to_dict())
        return accounts_info

    def delete_account(self, account_id: str) -> bool:
        """Remove account from keychain.

        Args:
            account_id: Account ID to remove

        Returns:
            True if deleted, False if not found
        """
        key = self._get_account_key(account_id)

        try:
            keyring.delete_password(self.service_name, key)

            # Update index
            accounts = self._get_accounts_index()
            if account_id in accounts:
                accounts.remove(account_id)
                self._update_accounts_index(accounts)

            logger.info(f"Deleted account: {account_id}")
            return True
        except KeyringError:
            return False

    def rotate_subkey(
        self,
        account_id: str,
        new_private_key: str,
    ) -> bool:
        """Rotate subkey for an account.

        Args:
            account_id: Account to rotate key for
            new_private_key: New subkey private key

        Returns:
            True if rotated, False if account not found
        """
        creds = self.get_account(account_id)
        if not creds:
            return False

        # Update credentials
        creds.l2_private_key = new_private_key
        self.store_account(creds)

        logger.info(f"Rotated subkey for account: {creds.alias}")
        return True

    def get_all_credentials(self) -> list[AccountCredentials]:
        """Get all account credentials.

        Returns:
            List of all account credentials
        """
        credentials = []
        for account_id in self._get_accounts_index():
            creds = self.get_account(account_id)
            if creds:
                credentials.append(creds)
        return credentials
