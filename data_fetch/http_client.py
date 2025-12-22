"""
Shared HTTP client for async requests with proper SSL and connection handling.
Abstracts common aiohttp setup to avoid repetition across scripts.
"""
import ssl
import socket
import aiohttp
import certifi
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class HttpClient:
    """
    Async HTTP client with robust SSL configuration.
    
    Usage:
        async with HttpClient() as client:
            async with client.session.get(url) as response:
                data = await response.json()
    
    Or for manual lifecycle management:
        client = HttpClient()
        await client.start()
        # ... use client.session ...
        await client.close()
    """
    
    def __init__(
        self,
        connection_limit: int = 100,
        connection_limit_per_host: Optional[int] = None,
        timeout_total: int = 120,
        timeout_connect: int = 30,
        dns_cache_ttl: int = 300,
        force_close: bool = True,
        use_ipv4_only: bool = False
    ):
        """
        Initialize HTTP client configuration.
        
        Args:
            connection_limit: Max total concurrent connections
            connection_limit_per_host: Max connections per host (defaults to connection_limit)
            timeout_total: Total request timeout in seconds
            timeout_connect: Connection timeout in seconds
            dns_cache_ttl: DNS cache time-to-live in seconds
            force_close: Force close connections after each request
            use_ipv4_only: Use IPv4 only (can help on some networks)
        """
        self.connection_limit = connection_limit
        self.connection_limit_per_host = connection_limit_per_host
        self.timeout_total = timeout_total
        self.timeout_connect = timeout_connect
        self.dns_cache_ttl = dns_cache_ttl
        self.force_close = force_close
        self.use_ipv4_only = use_ipv4_only
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None
    
    @property
    def session(self) -> aiohttp.ClientSession:
        """Get the aiohttp session. Raises if not started."""
        if self._session is None:
            raise RuntimeError(
                "HTTP client not started. Use 'async with HttpClient()' or call 'await client.start()'"
            )
        return self._session
    
    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context with proper certificate verification."""
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        return ssl_context
    
    def _create_connector(self) -> aiohttp.TCPConnector:
        """Create TCP connector with configured settings."""
        ssl_context = self._create_ssl_context()
        resolver = aiohttp.AsyncResolver()
        
        connector_kwargs = {
            "limit": self.connection_limit,
            "ssl": ssl_context,
            "ttl_dns_cache": self.dns_cache_ttl,
            "force_close": self.force_close,
            "resolver": resolver
        }
        
        if self.connection_limit_per_host is not None:
            connector_kwargs["limit_per_host"] = self.connection_limit_per_host
        
        if self.use_ipv4_only:
            connector_kwargs["family"] = socket.AF_INET
        
        return aiohttp.TCPConnector(**connector_kwargs)
    
    def _create_timeout(self) -> aiohttp.ClientTimeout:
        """Create client timeout configuration."""
        return aiohttp.ClientTimeout(
            total=self.timeout_total,
            connect=self.timeout_connect
        )
    
    async def start(self) -> "HttpClient":
        """
        Start the HTTP client session.
        Returns self for method chaining.
        """
        if self._session is not None:
            logger.warning("HTTP client already started")
            return self
        
        self._connector = self._create_connector()
        timeout = self._create_timeout()
        
        self._session = aiohttp.ClientSession(
            connector=self._connector,
            timeout=timeout,
            trust_env=True
        )
        
        logger.debug(
            "HTTP client started (limit=%s, timeout=%ss)",
            self.connection_limit, self.timeout_total
        )
        return self
    
    async def close(self) -> None:
        """Close the HTTP client session."""
        if self._session is not None:
            await self._session.close()
            self._session = None
            self._connector = None
            logger.debug("HTTP client closed")
    
    async def __aenter__(self) -> "HttpClient":
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


# Convenience function for quick session creation
async def create_session(
    connection_limit: int = 100,
    timeout_total: int = 120,
    timeout_connect: int = 30
) -> aiohttp.ClientSession:
    """
    Create and return a configured aiohttp ClientSession.
    
    NOTE: Caller is responsible for closing the session!
    Prefer using HttpClient as a context manager instead.
    
    Usage:
        session = await create_session()
        try:
            async with session.get(url) as response:
                data = await response.json()
        finally:
            await session.close()
    """
    client = HttpClient(
        connection_limit=connection_limit,
        timeout_total=timeout_total,
        timeout_connect=timeout_connect
    )
    await client.start()
    return client.session

