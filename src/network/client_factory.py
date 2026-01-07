"""Factory for creating Paradex API clients."""

import logging
from typing import Dict, Literal, Optional

from paradex_py import Paradex, ParadexSubkey

from src.security.keychain import AccountCredentials

logger = logging.getLogger(__name__)

# Environment type matches paradex-py SDK
ParadexEnv = Literal["prod", "testnet", "nightly"]


class ParadexClientFactory:
    """Factory for creating authenticated Paradex clients.

    Uses ParadexSubkey for L2-only authentication (subkeys).
    Subkeys have trading permissions but cannot withdraw or transfer.
    """

    def __init__(
        self,
        environment: str = "mainnet",
    ):
        """Initialize client factory.

        Args:
            environment: 'mainnet' or 'testnet'
        """
        # Map 'mainnet' to 'prod' as per SDK requirement
        self.environment: ParadexEnv = (
            "prod"
            if environment == "mainnet"
            else "testnet"
        )
        self._clients: Dict[str, ParadexSubkey] = {}
        logger.info(f"Initialized ParadexClientFactory for {environment}")

    async def create_client(
        self,
        credentials: AccountCredentials,
    ) -> ParadexSubkey:
        """Create authenticated Paradex client from credentials.

        Args:
            credentials: Account credentials with subkey

        Returns:
            Authenticated ParadexSubkey client
        """
        account_id = credentials.account_id

        # Check if client already exists
        if account_id in self._clients:
            logger.debug(f"Reusing existing client for {credentials.alias}")
            return self._clients[account_id]

        # Create new client with subkey
        client = ParadexSubkey(
            env=self.environment,
            l2_address=credentials.l2_address,
            l2_private_key=credentials.l2_private_key,
        )

        self._clients[account_id] = client
        logger.info(f"Created client for account: {credentials.alias}")

        return client

    async def create_client_from_keys(
        self,
        account_id: str,
        l2_address: str,
        l2_private_key: str,
    ) -> ParadexSubkey:
        """Create client directly from keys.

        Args:
            account_id: Unique account identifier
            l2_address: Starknet L2 address
            l2_private_key: Subkey private key

        Returns:
            Authenticated ParadexSubkey client
        """
        if account_id in self._clients:
            return self._clients[account_id]

        client = ParadexSubkey(
            env=self.environment,
            l2_address=l2_address,
            l2_private_key=l2_private_key,
        )

        self._clients[account_id] = client
        return client

    def get_client(self, account_id: str) -> Optional[ParadexSubkey]:
        """Get existing client by account ID.

        Args:
            account_id: Account ID to look up

        Returns:
            ParadexSubkey client if exists, None otherwise
        """
        return self._clients.get(account_id)

    def has_client(self, account_id: str) -> bool:
        """Check if client exists for account.

        Args:
            account_id: Account ID to check

        Returns:
            True if client exists
        """
        return account_id in self._clients

    def remove_client(self, account_id: str) -> bool:
        """Remove client from factory.

        Args:
            account_id: Account ID to remove

        Returns:
            True if removed, False if not found
        """
        if account_id in self._clients:
            del self._clients[account_id]
            logger.info(f"Removed client: {account_id}")
            return True
        return False

    async def close_all(self) -> None:
        """Close all client connections."""
        for account_id in list(self._clients.keys()):
            try:
                # ParadexSubkey may have cleanup methods
                client = self._clients[account_id]
                if hasattr(client, "close"):
                    await client.close()
            except Exception as e:
                logger.warning(f"Error closing client {account_id}: {e}")

        self._clients.clear()
        logger.info("Closed all Paradex clients")

    @property
    def client_count(self) -> int:
        """Get number of active clients."""
        return len(self._clients)

    @property
    def client_ids(self) -> list[str]:
        """Get list of active client account IDs."""
        return list(self._clients.keys())


class ParadexL1ClientFactory:
    """Factory for L1+L2 clients (when full access needed).

    Note: Only use this for initial onboarding or operations
    that require main private key. For trading, use ParadexClientFactory
    with subkeys.
    """

    def __init__(self, environment: str = "mainnet"):
        """Initialize L1 client factory."""
        # Map 'mainnet' to 'prod' as per SDK requirement
        self.environment: ParadexEnv = (
            "prod"
            if environment == "mainnet"
            else "testnet"
        )
        self._clients: Dict[str, Paradex] = {}

    async def create_client(
        self,
        account_id: str,
        l1_address: str,
        l1_private_key: str,
    ) -> Paradex:
        """Create L1+L2 client (use with caution).

        WARNING: This client has full access including withdrawals.
        Only use for initial setup or when absolutely necessary.

        Args:
            account_id: Unique identifier
            l1_address: Ethereum L1 address
            l1_private_key: Ethereum private key

        Returns:
            Full Paradex client
        """
        if account_id in self._clients:
            return self._clients[account_id]

        client = Paradex(
            env=self.environment,
            l1_address=l1_address,
            l1_private_key=l1_private_key,
        )

        self._clients[account_id] = client
        logger.warning(f"Created L1 client for {account_id} - use subkey for trading!")

        return client

    async def close_all(self) -> None:
        """Close all L1 clients."""
        self._clients.clear()
