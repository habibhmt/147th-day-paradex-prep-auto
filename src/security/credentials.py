"""High-level credential management."""

import hashlib
import logging
import secrets
from typing import Optional

from src.security.keychain import AccountCredentials, KeychainManager

logger = logging.getLogger(__name__)


class CredentialManager:
    """High-level manager for account credentials.

    Provides additional functionality on top of KeychainManager:
    - Account ID generation
    - Validation
    - Batch operations
    """

    def __init__(self, keychain: Optional[KeychainManager] = None):
        """Initialize credential manager.

        Args:
            keychain: KeychainManager instance (creates new if None)
        """
        self.keychain = keychain or KeychainManager()

    @staticmethod
    def generate_account_id(l2_address: str) -> str:
        """Generate unique account ID from L2 address.

        Args:
            l2_address: Starknet L2 address

        Returns:
            Unique account ID (first 8 chars of hash + random suffix)
        """
        # Hash the address
        addr_hash = hashlib.sha256(l2_address.encode()).hexdigest()[:8]
        # Add random suffix for uniqueness
        suffix = secrets.token_hex(2)
        return f"{addr_hash}-{suffix}"

    @staticmethod
    def validate_l2_address(address: str) -> bool:
        """Validate Starknet L2 address format.

        Args:
            address: Address to validate

        Returns:
            True if valid format
        """
        if not address:
            return False
        if not address.startswith("0x"):
            return False
        # Starknet addresses are 64 hex chars (+ 0x prefix)
        if len(address) != 66:
            return False
        try:
            int(address, 16)
            return True
        except ValueError:
            return False

    @staticmethod
    def validate_private_key(key: str) -> bool:
        """Validate private key format.

        Args:
            key: Private key to validate

        Returns:
            True if valid format
        """
        if not key:
            return False
        if not key.startswith("0x"):
            return False
        try:
            int(key, 16)
            return True
        except ValueError:
            return False

    def add_account(
        self,
        alias: str,
        l2_address: str,
        l2_private_key: str,
        account_id: Optional[str] = None,
    ) -> AccountCredentials:
        """Add a new account.

        Args:
            alias: Human-readable alias
            l2_address: Starknet L2 address
            l2_private_key: Subkey private key
            account_id: Optional custom account ID

        Returns:
            Created AccountCredentials

        Raises:
            ValueError: If validation fails
        """
        # Validate inputs
        if not self.validate_l2_address(l2_address):
            raise ValueError(f"Invalid L2 address format: {l2_address}")

        if not self.validate_private_key(l2_private_key):
            raise ValueError("Invalid private key format")

        # Check alias uniqueness
        existing = self.keychain.get_account_by_alias(alias)
        if existing:
            raise ValueError(f"Alias already exists: {alias}")

        # Generate account ID if not provided
        if not account_id:
            account_id = self.generate_account_id(l2_address)

        # Create credentials
        creds = AccountCredentials(
            account_id=account_id,
            alias=alias,
            l2_address=l2_address,
            l2_private_key=l2_private_key,
        )

        # Store in keychain
        self.keychain.store_account(creds)

        logger.info(f"Added account: {alias} (ID: {account_id})")
        return creds

    def remove_account(self, alias_or_id: str) -> bool:
        """Remove account by alias or ID.

        Args:
            alias_or_id: Account alias or ID

        Returns:
            True if removed
        """
        # Try as alias first
        creds = self.keychain.get_account_by_alias(alias_or_id)
        if creds:
            return self.keychain.delete_account(creds.account_id)

        # Try as account ID
        return self.keychain.delete_account(alias_or_id)

    def get_account(self, alias_or_id: str) -> Optional[AccountCredentials]:
        """Get account by alias or ID.

        Args:
            alias_or_id: Account alias or ID

        Returns:
            AccountCredentials if found
        """
        # Try as alias first
        creds = self.keychain.get_account_by_alias(alias_or_id)
        if creds:
            return creds

        # Try as account ID
        return self.keychain.get_account(alias_or_id)

    def list_accounts(self) -> list[dict]:
        """List all accounts (without private keys).

        Returns:
            List of account info
        """
        return self.keychain.list_accounts_info()

    def get_all_credentials(self) -> list[AccountCredentials]:
        """Get all account credentials.

        Returns:
            List of all credentials
        """
        return self.keychain.get_all_credentials()

    def account_count(self) -> int:
        """Get number of accounts.

        Returns:
            Number of stored accounts
        """
        return len(self.keychain.list_accounts())
