from __future__ import annotations

import threading
import typing

import pytest

from urllib3 import HTTPResponse, PoolManager
from urllib3.connectionpool import HTTPSConnectionPool
from urllib3.exceptions import ClosedPoolError, ProtocolError

if typing.TYPE_CHECKING:
    from collections.abc import Iterator


class PausableHTTPSConnection:
    """A fake HTTPS connection for testing client identity rotation.

    Records the client identity generation (``cert_file``) it was created
    with, every request it serves, and how many times it was closed. Requests
    can be paused via class-level events so tests can interleave certificate
    rotation with in-flight requests.
    """

    created: typing.ClassVar[list[PausableHTTPSConnection]]
    request_entered: typing.ClassVar[threading.Event | None]
    request_gate: typing.ClassVar[threading.Event | None]
    broken_cert_files: typing.ClassVar[frozenset[str]]

    def __init__(
        self,
        host: str,
        port: int | None = None,
        timeout: typing.Any = None,
        **kw: typing.Any,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.cert_file = kw.get("cert_file")
        self.key_file = kw.get("key_file")
        self.close_count = 0
        self.is_closed = True
        self.is_verified = True
        self.proxy_is_verified = True
        self.proxy = None
        self.has_connected_to_proxy = False
        self.requests: list[tuple[str, str]] = []
        type(self).created.append(self)

    @property
    def is_connected(self) -> bool:
        return not self.is_closed

    def connect(self) -> None:
        if self.cert_file in type(self).broken_cert_files:
            raise OSError(f"cannot load client certificate: {self.cert_file}")
        self.is_closed = False

    def request(self, method: str, url: str, **kw: typing.Any) -> None:
        self.requests.append((method, url))
        entered = type(self).request_entered
        gate = type(self).request_gate
        if entered is not None:
            entered.set()
        if gate is not None:
            gate.wait(timeout=10)

    def getresponse(self) -> HTTPResponse:
        return HTTPResponse(body=b"ok", status=200)

    def close(self) -> None:
        self.close_count += 1
        self.is_closed = True


@pytest.fixture()
def conn_cls() -> type[PausableHTTPSConnection]:
    """A fresh fake connection class per test, so recordings don't leak."""

    class FakeHTTPSConnection(PausableHTTPSConnection):
        created: typing.ClassVar[list[PausableHTTPSConnection]] = []
        request_entered: typing.ClassVar[threading.Event | None] = None
        request_gate: typing.ClassVar[threading.Event | None] = None
        broken_cert_files: typing.ClassVar[frozenset[str]] = frozenset()

    return FakeHTTPSConnection


@pytest.fixture()
def manager(
    conn_cls: type[PausableHTTPSConnection],
) -> Iterator[PoolManager]:
    """A PoolManager serving HTTPS through the fake connection class."""

    class FakeHTTPSConnectionPool(HTTPSConnectionPool):
        ConnectionCls = conn_cls  # type: ignore[assignment]

    pool_manager = PoolManager(cert_file="client-v1.pem", key_file="client-v1.key")
    pool_manager.pool_classes_by_scheme = {"https": FakeHTTPSConnectionPool}  # type: ignore[dict-item]
    with pool_manager:
        yield pool_manager


def _request_in_thread(
    pool_manager: PoolManager, url: str
) -> tuple[threading.Thread, dict[str, typing.Any]]:
    result: dict[str, typing.Any] = {}

    def target() -> None:
        try:
            result["response"] = pool_manager.request("GET", url)
        except Exception as e:  # noqa: BLE001
            result["error"] = e

    thread = threading.Thread(target=target)
    thread.start()
    return thread, result


class TestClientIdentityRotation:
    def test_checkout_rotate_return(
        self,
        manager: PoolManager,
        conn_cls: type[PausableHTTPSConnection],
    ) -> None:
        """Borrow a connection, rotate while the request is in flight, then
        return it: the request completes but the connection is closed instead
        of being re-pooled."""
        old_pool = manager.connection_from_url("https://service.example/")
        conn_cls.request_entered = threading.Event()
        conn_cls.request_gate = threading.Event()

        thread, result = _request_in_thread(manager, "https://service.example/")
        assert conn_cls.request_entered.wait(timeout=5)

        # The in-flight request is bound to identity generation 1.
        (conn_v1,) = conn_cls.created
        assert conn_v1.cert_file == "client-v1.pem"
        assert conn_v1.requests == [("GET", "/")]

        retired = manager.retire_client_identity(
            cert_file="client-v1.pem", host="service.example"
        )
        assert retired == 1
        assert len(manager.pools) == 0

        # The retired pool no longer lends connections, even to callers still
        # holding a reference to it.
        with pytest.raises(ClosedPoolError):
            old_pool._get_conn()

        # The in-flight request is allowed to complete normally...
        conn_cls.request_gate.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert result["response"].status == 200

        # ...but its connection is closed on return, not re-pooled.
        assert conn_v1.close_count == 1
        assert conn_v1.requests == [("GET", "/")]

        # Rotate the configuration: new requests must use the new identity.
        manager.connection_pool_kw["cert_file"] = "client-v2.pem"
        manager.connection_pool_kw["key_file"] = "client-v2.key"

        response = manager.request("GET", "https://service.example/")
        assert response.status == 200
        (conn_v2,) = [c for c in conn_cls.created if c.cert_file == "client-v2.pem"]
        assert conn_v2.requests == [("GET", "/")]
        assert conn_v2.close_count == 0  # healthy connection, re-pooled

        # The new pool is reused; no further generation-1 connections appear.
        response = manager.request("GET", "https://service.example/")
        assert response.status == 200
        assert len(conn_cls.created) == 2
        assert conn_v2.requests == [("GET", "/"), ("GET", "/")]
        assert conn_v1.close_count == 1

        # A stale reference to the retired pool still cannot lend connections.
        with pytest.raises(ClosedPoolError):
            old_pool._get_conn()

    def test_rotation_interleaved_with_new_connections(
        self,
        manager: PoolManager,
        conn_cls: type[PausableHTTPSConnection],
    ) -> None:
        """Rotate while an old-identity request is in flight and a
        new-identity connection is being created concurrently."""
        conn_cls.request_entered = threading.Event()
        conn_cls.request_gate = threading.Event()
        gate_v1 = conn_cls.request_gate

        thread, result = _request_in_thread(manager, "https://service.example/")
        assert conn_cls.request_entered.wait(timeout=5)
        (conn_v1,) = conn_cls.created

        # Rotate and immediately switch to the new client certificate.
        assert (
            manager.retire_client_identity(
                cert_file="client-v1.pem", host="service.example"
            )
            == 1
        )
        manager.connection_pool_kw["cert_file"] = "client-v2.pem"
        manager.connection_pool_kw["key_file"] = "client-v2.key"

        # While the generation-1 request is still paused, a new request
        # creates a fresh pool and a generation-2 connection.
        conn_cls.request_gate = None
        response = manager.request("GET", "https://service.example/")
        assert response.status == 200
        (conn_v2,) = [c for c in conn_cls.created if c.cert_file == "client-v2.pem"]
        assert conn_v2.requests == [("GET", "/")]

        # Now let the old request finish; its connection is retired on return.
        gate_v1.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert result["response"].status == 200
        assert conn_v1.requests == [("GET", "/")]
        assert conn_v1.close_count == 1
        assert conn_v2.close_count == 0

    def test_broken_new_config_does_not_fall_back(
        self,
        manager: PoolManager,
        conn_cls: type[PausableHTTPSConnection],
    ) -> None:
        """A failing new client configuration must surface an error instead
        of silently reusing the revoked identity."""
        response = manager.request("GET", "https://service.example/")
        assert response.status == 200
        (conn_v1,) = conn_cls.created
        assert conn_v1.close_count == 0  # pooled after the request

        old_pool = manager.connection_from_url("https://service.example/")
        assert (
            manager.retire_client_identity(
                cert_file="client-v1.pem", host="service.example"
            )
            == 1
        )
        # The idle generation-1 connection is closed immediately.
        assert conn_v1.close_count == 1

        # Switch to a new identity whose certificate cannot be loaded.
        manager.connection_pool_kw["cert_file"] = "client-v2-broken.pem"
        manager.connection_pool_kw["key_file"] = "client-v2-broken.key"
        conn_cls.broken_cert_files = frozenset({"client-v2-broken.pem"})

        with pytest.raises(ProtocolError):
            manager.request("GET", "https://service.example/", retries=False)

        # The revoked identity was not used as a fallback: no generation-1
        # connection served another request or was revived.
        assert conn_v1.requests == [("GET", "/")]
        assert conn_v1.close_count == 1
        (conn_v2,) = [
            c for c in conn_cls.created if c.cert_file == "client-v2-broken.pem"
        ]
        assert conn_v2.requests == []
        assert conn_v2.close_count == 1  # discarded after the failure

        # The manager holds no pool for the revoked identity.
        for pool_key in manager.pools.keys():
            assert pool_key.key_cert_file != "client-v1.pem"
        with pytest.raises(ClosedPoolError):
            old_pool._get_conn()

    def test_other_hosts_and_identities_unaffected(
        self,
        manager: PoolManager,
        conn_cls: type[PausableHTTPSConnection],
    ) -> None:
        """Retirement is scoped: other hosts and other identities keep
        working with their existing pools."""
        url_a = "https://service-a.example/"
        url_b = "https://service-b.example/"

        manager.request("GET", url_a)
        manager.request("GET", url_b)
        # Same host, but a different client identity.
        manager.connection_from_url(url_a, pool_kwargs={"cert_file": "other.pem"})
        other_pool = manager.connection_from_url(
            url_a, pool_kwargs={"cert_file": "other.pem"}
        )
        pool_b = manager.connection_from_url(url_b)
        assert len(manager.pools) == 3

        # Host-scoped retirement only retires the matching pool.
        retired = manager.retire_client_identity(
            cert_file="client-v1.pem", host="service-a.example"
        )
        assert retired == 1
        assert len(manager.pools) == 2
        assert other_pool.pool is not None
        assert pool_b.pool is not None

        # The other identity on the same host still works.
        response = other_pool.request("GET", "/")
        assert response.status == 200
        (conn_other,) = [c for c in conn_cls.created if c.cert_file == "other.pem"]
        assert conn_other.requests == [("GET", "/")]
        assert conn_other.close_count == 0

        # The other host with the same certificate still works.
        response = manager.request("GET", url_b)
        assert response.status == 200
        (conn_b,) = [
            c
            for c in conn_cls.created
            if c.cert_file == "client-v1.pem" and c.host == "service-b.example"
        ]
        assert conn_b.close_count == 0

        # Without a host, the identity is retired across all hosts.
        retired = manager.retire_client_identity(cert_file="client-v1.pem")
        assert retired == 1
        assert pool_b.pool is None
        assert other_pool.pool is not None

    def test_retire_requires_identity_fields(self, manager: PoolManager) -> None:
        with pytest.raises(ValueError, match="requires at least one of"):
            manager.retire_client_identity()

    def test_retire_unknown_identity_is_noop(self, manager: PoolManager) -> None:
        manager.request("GET", "https://service.example/")
        assert manager.retire_client_identity(cert_file="nope.pem") == 0
        assert len(manager.pools) == 1
