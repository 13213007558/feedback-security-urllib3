"""Offline evidence for double-layer TLS through an HTTPS proxy.

The scenario is TLS-in-TLS: an outer TLS handshake authenticates the proxy
with a *proxy* private CA, an HTTP ``CONNECT`` is exchanged inside that
outer layer, and then an inner TLS handshake authenticates the destination
with a separate *origin* private CA.  Nothing in this module opens a real
socket, binds a port, contacts a real proxy, or reads the operating system
trust store:

* certificates are minted on the fly with the ``openssl`` command line
  (two independent private CAs, ECDSA P-256, SHA-256 signatures);
* every byte flows through :func:`socket.socketpair`, so the "server" is a
  thread living in the same process;
* each client :class:`~ssl.SSLContext` is pinned to exactly one private CA
  via :meth:`~ssl.SSLContext.load_verify_locations`, never
  :meth:`~ssl.SSLContext.load_default_certs`.

The failure cases deliberately assert *where* verification happens instead
of collapsing everything into a generic :class:`~urllib3.exceptions.ProxyError`:

* an outer-layer rejection (proxy CA/name) surfaces as ``ProxyError``
  because the proxy was never reached successfully;
* an inner-layer rejection (origin CA/name) surfaces as a plain
  :class:`~urllib3.exceptions.SSLError` because the proxy had already been
  authenticated and the CONNECT tunnel established, so urllib3 no longer
  attributes the failure to the proxy.
"""

from __future__ import annotations

import shutil
import socket
import ssl
import subprocess
import threading
import typing
from pathlib import Path

import pytest

import urllib3
import urllib3.connection as urllib3_connection
from urllib3.exceptions import MaxRetryError, ProxyError, SSLError

pytestmark = pytest.mark.skipif(
    not shutil.which("openssl"), reason="openssl CLI is required to mint test CAs"
)

PROXY_HOST = "proxy.example"
ORIGIN_HOST = "origin.example"
ORIGIN_WRONG_HOST = "origin-wrong.example"
# A leaf issued by the *proxy* CA that nevertheless carries the origin's
# name. It proves a matching name alone cannot cross trust boundaries.
PROXY_CA_ORIGIN_NAME_LEAF = "proxy-ca-poses-as-origin"

_HANDSHAKE_TIMEOUT = 10

if typing.TYPE_CHECKING:
    from urllib3.connection import HTTPSConnection

# A factory takes a ready server thread and returns an HTTPSConnection._new_conn
# replacement that connects both ends of a fresh socketpair.
SocketProvider = typing.Callable[
    [threading.Thread], typing.Callable[["HTTPSConnection"], socket.socket]
]


class TLSMaterial:
    """File paths for the two independent private CAs and their leaves."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def ca(self, authority: str) -> str:
        return str(self.directory / f"{authority}-ca.pem")

    def cert(self, leaf: str) -> str:
        return str(self.directory / f"{leaf}.crt")

    def key(self, leaf: str) -> str:
        return str(self.directory / f"{leaf}.key")


def _run_openssl(arguments: list[str]) -> None:
    subprocess.run(
        ["openssl", *arguments],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _write_leaf_config(path: Path, common_name: str, san: str) -> None:
    path.write_text(
        "[req]\n"
        "distinguished_name=dn\n"
        "req_extensions=v3\n"
        "prompt=no\n"
        "[dn]\n"
        f"CN={common_name}\n"
        "[v3]\n"
        f"subjectAltName=DNS:{san}\n"
        "basicConstraints=CA:FALSE\n"
    )


def _mint_certificate_authority(
    material: TLSMaterial, authority: str, common_name: str
) -> None:
    key_path = material.directory / f"{authority}-ca.key"
    cert_path = material.directory / f"{authority}-ca.pem"
    _run_openssl(
        ["ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", str(key_path)]
    )
    _run_openssl(
        [
            "req",
            "-new",
            "-x509",
            "-sha256",
            "-key",
            str(key_path),
            "-out",
            str(cert_path),
            "-days",
            "2",
            "-subj",
            f"/CN={common_name}",
        ]
    )


def _mint_leaf(
    material: TLSMaterial,
    authority: str,
    leaf: str,
    common_name: str,
    san: str,
) -> None:
    directory = material.directory
    config_path = directory / f"{leaf}.cnf"
    key_path = directory / f"{leaf}.key"
    csr_path = directory / f"{leaf}.csr"
    cert_path = directory / f"{leaf}.crt"
    _write_leaf_config(config_path, common_name, san)
    _run_openssl(
        ["ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", str(key_path)]
    )
    _run_openssl(
        [
            "req",
            "-new",
            "-sha256",
            "-key",
            str(key_path),
            "-out",
            str(csr_path),
            "-config",
            str(config_path),
        ]
    )
    _run_openssl(
        [
            "x509",
            "-req",
            "-sha256",
            "-in",
            str(csr_path),
            "-CA",
            material.ca(authority),
            "-CAkey",
            str(directory / f"{authority}-ca.key"),
            "-CAcreateserial",
            "-days",
            "2",
            "-out",
            str(cert_path),
            "-extensions",
            "v3",
            "-extfile",
            str(config_path),
        ]
    )


@pytest.fixture(scope="module")
def tls_material(tmp_path_factory: pytest.TempPathFactory) -> TLSMaterial:
    """Mint two private CAs and their leaf certificates once per module."""
    material = TLSMaterial(tmp_path_factory.mktemp("tls-in-tls-certs"))
    _mint_certificate_authority(material, "proxy", "proxy private CA")
    _mint_certificate_authority(material, "origin", "origin private CA")
    _mint_leaf(material, "proxy", "proxy-leaf", PROXY_HOST, PROXY_HOST)
    _mint_leaf(material, "origin", "origin-leaf", ORIGIN_HOST, ORIGIN_HOST)
    _mint_leaf(
        material,
        "origin",
        "origin-wrongname",
        ORIGIN_WRONG_HOST,
        ORIGIN_WRONG_HOST,
    )
    # Correct name, wrong trust anchor: signed by the proxy CA rather than
    # the origin CA, even though its SAN is the origin hostname.
    _mint_leaf(
        material,
        "proxy",
        PROXY_CA_ORIGIN_NAME_LEAF,
        ORIGIN_HOST,
        ORIGIN_HOST,
    )
    return material


def _server_context(material: TLSMaterial, leaf: str) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(material.cert(leaf), material.key(leaf))
    return context


def _client_context(material: TLSMaterial, authority: str) -> ssl.SSLContext:
    # PROTOCOL_TLS_CLIENT defaults to check_hostname=True / CERT_REQUIRED.
    # Only the single private CA is trusted; the system trust store is never
    # consulted, so an unrelated or cross-signed leaf cannot be accepted.
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations(material.ca(authority))
    return context


class _InMemoryProxy(threading.Thread):
    """Two-layer TLS server speaking CONNECT over a :func:`socket.socketpair`.

    The outer layer is a regular :class:`ssl.SSLSocket` (the proxy). After
    the CONNECT response, the destination handshake is driven through
    :class:`ssl.MemoryBIO` objects, mirroring urllib3's own
    :class:`~urllib3.util.ssltransport.SSLTransport` on the client side.
    Nesting a second ``SSLSocket`` directly is impossible because the outer
    object owns the file descriptor, which is exactly why the library uses
    the BIO-based wrapper for TLS-in-TLS.
    """

    def __init__(
        self,
        material: TLSMaterial,
        proxy_leaf: str,
        origin_leaf: str,
        *,
        plain_proxy: bool = False,
    ) -> None:
        super().__init__(daemon=True)
        self._raw_socket: socket.socket | None = None
        self._material = material
        self._proxy_leaf = proxy_leaf
        self._origin_leaf = origin_leaf
        self._plain_proxy = plain_proxy
        self.sni: dict[str, str | None] = {}
        self.connect_line: bytes | None = None
        # The furthest stage reached; failure tests use it to show which
        # side aborted the handshake.
        self.stage = "initialized"
        self.error: BaseException | None = None
        self._outer: ssl.SSLSocket | socket.socket | None = None

    def bind(self, raw_socket: socket.socket) -> None:
        self._raw_socket = raw_socket

    def _record_sni(self, layer: str) -> typing.Callable[..., None]:
        def callback(
            ssl_object: ssl.SSLObject | ssl.SSLSocket,
            server_name: str | None,
            ssl_context: ssl.SSLContext,
        ) -> None:
            self.sni[layer] = server_name

        return callback

    def run(self) -> None:
        assert self._raw_socket is not None, "bind() must run before start()"
        try:
            self._outer = self._raw_socket
            if not self._plain_proxy:
                proxy_context = _server_context(self._material, self._proxy_leaf)
                proxy_context.set_servername_callback(self._record_sni("proxy"))
                self._outer = proxy_context.wrap_socket(
                    self._raw_socket, server_side=True
                )
                self.stage = "outer-tls"

            connect_data = b""
            while b"\r\n\r\n" not in connect_data:
                chunk = self._outer.recv(16384)
                if not chunk:
                    return
                connect_data += chunk
            self.connect_line = connect_data.split(b"\r\n", 1)[0]
            self.stage = "connect"
            self._outer.sendall(
                b"HTTP/1.1 200 Connection established\r\n\r\n"
            )

            self._run_inner_handshake(self._outer)
        except BaseException as exc:  # surfaced to the test via self.error
            self.error = exc
        finally:
            self._close_quietly()

    def _close_quietly(self) -> None:
        if self._outer is not None and self._outer is not self._raw_socket:
            try:
                self._outer.close()
            except OSError:
                pass
        raw_socket = self._raw_socket
        if raw_socket is not None and raw_socket.fileno() != -1:
            raw_socket.close()

    def _run_inner_handshake(self, outer: ssl.SSLSocket | socket.socket) -> None:
        origin_context = _server_context(self._material, self._origin_leaf)
        origin_context.set_servername_callback(self._record_sni("origin"))
        incoming = ssl.MemoryBIO()
        outgoing = ssl.MemoryBIO()
        inner = origin_context.wrap_bio(incoming, outgoing, server_side=True)
        while True:
            try:
                inner.do_handshake()
                break
            except ssl.SSLWantReadError:
                if outgoing.pending:
                    outer.sendall(outgoing.read())
                chunk = outer.recv(16384)
                if not chunk:
                    incoming.write_eof()
                    return
                incoming.write(chunk)
        if outgoing.pending:
            outer.sendall(outgoing.read())
        self.stage = "inner-handshake-done"
        self._serve_inner_request(inner, incoming, outgoing, outer)

    def _read_inner(
        self,
        inner: ssl.SSLObject,
        incoming: ssl.MemoryBIO,
        outgoing: ssl.MemoryBIO,
        outer: ssl.SSLSocket | socket.socket,
        size: int,
    ) -> bytes:
        """Read decrypted bytes, only touching the outer socket when the BIO
        genuinely needs more records.

        TLS 1.3 post-handshake messages (session tickets) and the client's
        request can already be buffered in ``incoming``, so a blocking
        ``outer.recv()`` must not be called until ``inner.read()`` signals
        ``SSLWantReadError``; otherwise an unrelated ``b""`` from the outer
        layer would be mistaken for end of stream.
        """
        while True:
            try:
                return inner.read(size)
            except ssl.SSLWantReadError:
                if outgoing.pending:
                    outer.sendall(outgoing.read())
                chunk = outer.recv(16384)
                if not chunk:
                    incoming.write_eof()
                    return b""
                incoming.write(chunk)

    def _serve_inner_request(
        self,
        inner: ssl.SSLObject,
        incoming: ssl.MemoryBIO,
        outgoing: ssl.MemoryBIO,
        outer: ssl.SSLSocket | socket.socket,
    ) -> None:
        request = b""
        while b"\r\n\r\n" not in request:
            chunk = self._read_inner(inner, incoming, outgoing, outer, 16384)
            if not chunk:
                return
            request += chunk
        self.stage = "request"
        inner.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
        # TLS 1.3 may keep the record buffered until a subsequent BIO cycle.
        while True:
            if outgoing.pending:
                outer.sendall(outgoing.read())
            try:
                inner.write(b"")
            except ssl.SSLWantReadError:
                continue
            break


class _InMemoryOrigin(threading.Thread):
    """Single-layer HTTPS origin used by the plain HTTPS control test."""

    def __init__(self, material: TLSMaterial) -> None:
        super().__init__(daemon=True)
        self._raw_socket: socket.socket | None = None
        self._secure_socket: ssl.SSLSocket | None = None
        self._material = material
        self.sni: dict[str, str | None] = {}
        self.error: BaseException | None = None

    def bind(self, raw_socket: socket.socket) -> None:
        self._raw_socket = raw_socket

    def run(self) -> None:
        assert self._raw_socket is not None, "bind() must run before start()"
        try:
            context = _server_context(self._material, "origin-leaf")

            def callback(
                ssl_object: ssl.SSLSocket,
                server_name: str | None,
                ssl_context: ssl.SSLContext,
            ) -> None:
                self.sni["origin"] = server_name

            context.set_servername_callback(callback)
            self._secure_socket = context.wrap_socket(
                self._raw_socket, server_side=True
            )
            request = b""
            while b"\r\n\r\n" not in request:
                chunk = self._secure_socket.recv(16384)
                if not chunk:
                    return
                request += chunk
            self._secure_socket.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"
            )
        except BaseException as exc:
            self.error = exc
        finally:
            if self._secure_socket is not None:
                self._secure_socket.close()
            elif self._raw_socket is not None and self._raw_socket.fileno() != -1:
                self._raw_socket.close()


@pytest.fixture
def socketpair_factory() -> SocketProvider:
    """Patch urllib3 to use fresh in-memory sockets for each connection.

    Returns a helper that wires a server thread to a brand-new
    :func:`socket.socketpair`. Retries must always get a fresh pair because
    reusing a closed descriptor masks the real verification error with
    ``EBADF``. All created client sockets are exposed on ``factory.created``
    for the resource-release assertions.
    """
    created: list[socket.socket] = []

    def factory(
        server_thread: threading.Thread,
    ) -> typing.Callable[["HTTPSConnection"], socket.socket]:
        def provide_socket(self: HTTPSConnection) -> socket.socket:
            client_socket, server_socket = socket.socketpair()
            client_socket.settimeout(_HANDSHAKE_TIMEOUT)
            server_socket.settimeout(_HANDSHAKE_TIMEOUT)
            server_thread.bind(server_socket)  # type: ignore[attr-defined]
            server_thread.start()
            created.append(client_socket)
            return client_socket

        return provide_socket

    factory.created = created  # type: ignore[attr-defined]
    original_new_conn = urllib3_connection.HTTPSConnection._new_conn
    yield factory
    urllib3_connection.HTTPSConnection._new_conn = (
        original_new_conn
    )  # type: ignore[method-assign]


def _proxy_manager(
    material: TLSMaterial,
    *,
    proxy_authority: str,
    origin_authority: str,
    scheme: str = "https",
) -> urllib3.ProxyManager:
    return urllib3.ProxyManager(
        f"{scheme}://{PROXY_HOST}:443",
        proxy_ssl_context=_client_context(material, proxy_authority),
        ssl_context=_client_context(material, origin_authority),
        retries=0,
    )


def _start_proxy(
    material: TLSMaterial,
    factory: SocketProvider,
    *,
    proxy_leaf: str = "proxy-leaf",
    origin_leaf: str = "origin-leaf",
    plain_proxy: bool = False,
) -> _InMemoryProxy:
    server = _InMemoryProxy(
        material,
        proxy_leaf,
        origin_leaf,
        plain_proxy=plain_proxy,
    )
    urllib3_connection.HTTPSConnection._new_conn = (
        factory(server)
    )  # type: ignore[method-assign]
    return server


def _assert_raw_sockets_closed(sockets: list[socket.socket]) -> None:
    assert sockets, "the harness expected at least one in-memory client socket"
    assert all(client_socket.fileno() == -1 for client_socket in sockets)


def _assert_contexts_strict(
    proxy_context: ssl.SSLContext, origin_context: ssl.SSLContext
) -> None:
    for context in (proxy_context, origin_context):
        assert context.check_hostname is True
        assert context.verify_mode == ssl.CERT_REQUIRED
    assert proxy_context is not origin_context


class TestTLSInTLS:
    def test_double_layer_handshake_completes_request(
        self,
        tls_material: TLSMaterial,
        socketpair_factory: SocketProvider,
    ) -> None:
        server = _start_proxy(tls_material, socketpair_factory)
        manager = _proxy_manager(
            tls_material, proxy_authority="proxy", origin_authority="origin"
        )
        try:
            # Keep the connection attached to the response so its verification
            # state and inner wrapper can be inspected before release.
            response = manager.urlopen(
                "GET", f"https://{ORIGIN_HOST}/", preload_content=False
            )
        finally:
            server.join(_HANDSHAKE_TIMEOUT)

        assert response.status == 200
        # Both SNIs travel over their own layer and are recorded separately.
        assert server.sni == {"proxy": PROXY_HOST, "origin": ORIGIN_HOST}
        assert server.connect_line == (
            f"CONNECT {ORIGIN_HOST}:443 HTTP/1.1".encode("ascii")
        )
        assert server.stage == "request"
        assert server.error is None

        # The connection used for the request is retained by the response
        # until the body is drained and the response released.
        connection = response.connection
        assert connection is not None
        try:
            assert connection.proxy_is_verified is True
            assert connection.is_verified is True
            # The inner object is the BIO wrapper, not a second SSLSocket.
            assert type(connection.sock).__name__ == "SSLTransport"
            assert response.read() == b"ok"
        finally:
            response.release_conn()

    def test_wrong_proxy_ca_fails_at_outer_layer(
        self,
        tls_material: TLSMaterial,
        socketpair_factory: SocketProvider,
    ) -> None:
        server = _start_proxy(tls_material, socketpair_factory)
        # The proxy context mistakenly trusts the *origin* CA only.
        manager = _proxy_manager(
            tls_material, proxy_authority="origin", origin_authority="origin"
        )
        with pytest.raises(MaxRetryError) as exc_info:
            manager.urlopen("GET", f"https://{ORIGIN_HOST}/")
        server.join(_HANDSHAKE_TIMEOUT)

        # The client rejects the proxy certificate: the proxy is never
        # authenticated, so urllib3 wraps the failure as ProxyError.
        assert isinstance(exc_info.value.reason, ProxyError)
        inner_error = exc_info.value.reason.original_error
        assert isinstance(inner_error, SSLError)
        assert isinstance(inner_error.args[0], ssl.SSLCertVerificationError)
        assert "unable to get local issuer certificate" in str(inner_error)
        # Only the proxy SNI was ever sent; no CONNECT or inner handshake.
        assert server.sni == {"proxy": PROXY_HOST}
        assert server.connect_line is None
        assert server.stage == "initialized"
        _assert_raw_sockets_closed(
            socketpair_factory.created  # type: ignore[attr-defined]
        )

    def test_wrong_origin_name_fails_at_inner_layer(
        self,
        tls_material: TLSMaterial,
        socketpair_factory: SocketProvider,
    ) -> None:
        # The origin presents a valid origin-CA leaf for the wrong hostname.
        server = _start_proxy(
            tls_material,
            socketpair_factory,
            origin_leaf="origin-wrongname",
        )
        manager = _proxy_manager(
            tls_material, proxy_authority="proxy", origin_authority="origin"
        )
        with pytest.raises(MaxRetryError) as exc_info:
            manager.urlopen("GET", f"https://{ORIGIN_HOST}/")
        server.join(_HANDSHAKE_TIMEOUT)

        # The proxy was already trusted and the tunnel established, so this
        # is an origin-side rejection and must NOT be a ProxyError.
        assert isinstance(exc_info.value.reason, SSLError)
        assert not isinstance(exc_info.value.reason, ProxyError)
        cert_error = exc_info.value.reason.args[0]
        assert isinstance(cert_error, ssl.SSLCertVerificationError)
        assert "Hostname mismatch" in str(cert_error)
        assert ORIGIN_HOST in str(cert_error)
        # Both handshakes were attempted and both SNIs reached the right layer.
        assert server.sni == {"proxy": PROXY_HOST, "origin": ORIGIN_HOST}
        assert server.connect_line is not None
        # The proxy outer layer and CONNECT succeeded; the abort happens while
        # the origin (inner) handshake is being driven, never before CONNECT.
        assert server.stage == "connect"
        # The client itself rejects the name, so its TLS alert tears down the
        # tunnel and the server only observes the resulting broken pipe.
        assert isinstance(server.error, OSError)
        _assert_raw_sockets_closed(
            socketpair_factory.created  # type: ignore[attr-defined]
        )

    def test_wrong_origin_ca_fails_at_inner_layer(
        self,
        tls_material: TLSMaterial,
        socketpair_factory: SocketProvider,
    ) -> None:
        # The origin context mistakenly trusts the proxy CA. The outer proxy
        # handshake still validates (the proxy leaf really is proxy-CA signed),
        # but the origin leaf is rejected at the inner layer.
        server = _start_proxy(tls_material, socketpair_factory)
        manager = _proxy_manager(
            tls_material, proxy_authority="proxy", origin_authority="proxy"
        )
        with pytest.raises(MaxRetryError) as exc_info:
            manager.urlopen("GET", f"https://{ORIGIN_HOST}/")
        server.join(_HANDSHAKE_TIMEOUT)

        # Reached the proxy and CONNECT first, so this is not a ProxyError.
        assert isinstance(exc_info.value.reason, SSLError)
        assert not isinstance(exc_info.value.reason, ProxyError)
        cert_error = exc_info.value.reason.args[0]
        assert isinstance(cert_error, ssl.SSLCertVerificationError)
        assert "unable to get local issuer certificate" in str(cert_error)
        assert server.sni == {"proxy": PROXY_HOST, "origin": ORIGIN_HOST}
        assert server.connect_line is not None
        _assert_raw_sockets_closed(
            socketpair_factory.created  # type: ignore[attr-defined]
        )

    def test_swapped_contexts_fail_at_outer_layer(
        self,
        tls_material: TLSMaterial,
        socketpair_factory: SocketProvider,
    ) -> None:
        server = _start_proxy(tls_material, socketpair_factory)
        # Deliberately swap the two contexts: proxy context trusts the origin
        # CA and vice versa. Neither layer can validate the peer in front of it.
        manager = urllib3.ProxyManager(
            f"https://{PROXY_HOST}:443",
            proxy_ssl_context=_client_context(tls_material, "origin"),
            ssl_context=_client_context(tls_material, "proxy"),
            retries=0,
        )
        with pytest.raises(MaxRetryError) as exc_info:
            manager.urlopen("GET", f"https://{ORIGIN_HOST}/")
        server.join(_HANDSHAKE_TIMEOUT)

        # Failure must happen at the first (proxy) layer.
        assert isinstance(exc_info.value.reason, ProxyError)
        inner_error = exc_info.value.reason.original_error
        assert isinstance(inner_error, SSLError)
        assert "unable to get local issuer certificate" in str(inner_error)
        assert server.connect_line is None
        assert server.stage == "initialized"
        _assert_raw_sockets_closed(
            socketpair_factory.created  # type: ignore[attr-defined]
        )

    def test_proxy_ca_cannot_sign_origin_even_with_matching_name(
        self,
        tls_material: TLSMaterial,
        socketpair_factory: SocketProvider,
    ) -> None:
        # The inner server presents a proxy-CA leaf whose SAN is exactly the
        # origin hostname. The name matches but the trust anchor does not.
        server = _start_proxy(
            tls_material,
            socketpair_factory,
            origin_leaf=PROXY_CA_ORIGIN_NAME_LEAF,
        )
        manager = _proxy_manager(
            tls_material, proxy_authority="proxy", origin_authority="origin"
        )
        with pytest.raises(MaxRetryError) as exc_info:
            manager.urlopen("GET", f"https://{ORIGIN_HOST}/")
        server.join(_HANDSHAKE_TIMEOUT)

        assert isinstance(exc_info.value.reason, SSLError)
        assert not isinstance(exc_info.value.reason, ProxyError)
        cert_error = exc_info.value.reason.args[0]
        assert isinstance(cert_error, ssl.SSLCertVerificationError)
        assert "unable to get local issuer certificate" in str(cert_error)
        assert server.sni.get("origin") == ORIGIN_HOST
        _assert_raw_sockets_closed(
            socketpair_factory.created  # type: ignore[attr-defined]
        )

    def test_contexts_pin_distinct_private_cas_with_verification_on(
        self, tls_material: TLSMaterial
    ) -> None:
        proxy_context = _client_context(tls_material, "proxy")
        origin_context = _client_context(tls_material, "origin")
        _assert_contexts_strict(proxy_context, origin_context)
        # The two CA files are genuinely different certificates.
        proxy_ca = Path(tls_material.ca("proxy")).read_bytes()
        origin_ca = Path(tls_material.ca("origin")).read_bytes()
        assert proxy_ca != origin_ca
        # Each context holds exactly one issuer, i.e. its own private CA.
        proxy_issuers = proxy_context.get_ca_certs()
        origin_issuers = origin_context.get_ca_certs()
        assert len(proxy_issuers) == 1
        assert len(origin_issuers) == 1
        assert (
            proxy_issuers[0]["subject"]
            != origin_issuers[0]["subject"]
        )


class TestTLSInTLSControls:
    def test_plain_https_control_single_layer(
        self,
        tls_material: TLSMaterial,
        socketpair_factory: SocketProvider,
    ) -> None:
        server = _InMemoryOrigin(tls_material)
        urllib3_connection.HTTPSConnection._new_conn = socketpair_factory(
            server
        )  # type: ignore[method-assign]
        pool = urllib3.HTTPSConnectionPool(
            ORIGIN_HOST,
            443,
            ssl_context=_client_context(tls_material, "origin"),
            retries=0,
        )
        try:
            response = pool.urlopen("GET", "/")
        finally:
            server.join(_HANDSHAKE_TIMEOUT)

        assert response.status == 200
        assert response.data == b"ok"
        assert server.sni == {"origin": ORIGIN_HOST}
        assert server.error is None

    def test_http_connect_control_plaintext_proxy(
        self,
        tls_material: TLSMaterial,
        socketpair_factory: SocketProvider,
    ) -> None:
        server = _start_proxy(
            tls_material, socketpair_factory, plain_proxy=True
        )
        manager = _proxy_manager(
            tls_material,
            proxy_authority="proxy",
            origin_authority="origin",
            scheme="http",
        )
        try:
            response = manager.urlopen("GET", f"https://{ORIGIN_HOST}/")
        finally:
            server.join(_HANDSHAKE_TIMEOUT)

        assert response.status == 200
        assert response.data == b"ok"
        # No outer TLS: only the inner origin handshake carries an SNI.
        assert "proxy" not in server.sni
        assert server.sni == {"origin": ORIGIN_HOST}
        assert server.connect_line == (
            f"CONNECT {ORIGIN_HOST}:443 HTTP/1.1".encode("ascii")
        )
        assert server.stage == "request"
        assert server.error is None
