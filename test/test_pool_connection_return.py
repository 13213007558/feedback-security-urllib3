"""Regression tests for connection return safety.

The worst thing a connection pool can hand out is a connection that still
carries bytes from the previous response. The next request would then read a
mixture of the old response tail and its own framing. These tests pin the
rules around returning a connection to the pool:

* a reusable connection must have its HTTP message boundary fully confirmed
  (the whole framed body consumed and, for compressed responses, the decoder
  reached end of stream);
* when the boundary cannot be confirmed (early EOF, a framing/decompression
  failure, or a caller that only reads part of the body) the connection may be
  discarded, but any unconsumed bytes must never reach the next request;
* a complete, well-framed response must still be reusable, so the safe
  behaviour cannot be achieved by simply closing every connection.

The harness never opens a network socket. A fake socket is backed by an
in-memory ``os.pipe`` so the real ``HTTPConnection.is_connected`` polling path
keeps working, and the response body is read through the production
``HTTPResponse`` code via an in-memory file. A separate *byte ledger* on that
file records how many framing bytes remain, instead of trusting the object's
``closed`` flag (which is set even when bytes are still buffered).
"""

from __future__ import annotations

import http.client as httplib
import io
import os
import typing
import zlib

import pytest

from urllib3.connection import HTTPConnection
from urllib3.connectionpool import HTTPConnectionPool
from urllib3.exceptions import DecodeError, ProtocolError
from urllib3.response import HTTPResponse


class MockSock:
    """Stand-in for the socket argument of ``http.client.HTTPResponse``.

    Only ``makefile`` is referenced during construction; the body file is
    replaced explicitly in each test.
    """

    @classmethod
    def makefile(cls, *args: typing.Any, **kwargs: typing.Any) -> None:
        return None


class LedgeredBytesIO(io.BytesIO):
    """In-memory response body with an independent byte ledger.

    ``bytes_remaining`` is derived purely from a counter updated on every read,
    so it stays available (and trustworthy) after the file is closed. It must
    not rely on ``closed``/``isclosed()``, which do not prove that the framed
    body was fully consumed.
    """

    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self._total = len(data)
        self._consumed = 0

    @property
    def bytes_remaining(self) -> int:
        return self._total - self._consumed

    def read(self, n: int = -1) -> bytes:
        chunk = super().read(n)
        self._consumed += len(chunk)
        return chunk

    def read1(self, n: int = -1) -> bytes:
        chunk = super().read1(n)
        self._consumed += len(chunk)
        return chunk

    def readline(self, n: int = -1) -> bytes:
        chunk = super().readline(n)
        self._consumed += len(chunk)
        return chunk


class PipeSocket:
    """Socket-like object backed by an in-memory pipe.

    A real file descriptor lets ``select.poll`` (used by
    ``HTTPConnection.is_connected``) report the connection as alive until
    ``close()`` without any network access, so ``--disable-socket`` tests work.
    """

    def __init__(self) -> None:
        read_fd, write_fd = os.pipe()
        os.set_blocking(read_fd, False)
        self._read_fd = read_fd
        self._write_fd = write_fd
        self.closed = False

    def fileno(self) -> int:
        return self._read_fd

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        os.close(self._read_fd)
        os.close(self._write_fd)

    def settimeout(self, timeout: float | None) -> None:
        pass

    def setblocking(self, block: bool) -> None:
        pass

    def shutdown(self, how: int) -> None:
        pass


class RecordingConnectionPool(HTTPConnectionPool):
    """Pool that records every borrow/return and uses pipe-backed connections."""

    def __init__(self) -> None:
        super().__init__("localhost", maxsize=1)
        self.events: list[tuple[str, bool]] = []

    def _new_conn(self) -> HTTPConnection:
        self.num_connections += 1
        conn = HTTPConnection("localhost")
        # Pretend connect() already happened: a live, non-dropped socket.
        conn.sock = PipeSocket()
        self.events.append(("new", conn.is_closed))
        return conn

    def _put_conn(self, conn: HTTPConnection | None) -> None:
        self.events.append(("put", conn.is_closed if conn is not None else True))
        super()._put_conn(conn)

    def _get_conn(self, timeout: float | None = None) -> HTTPConnection:
        conn = super()._get_conn(timeout)
        self.events.append(("get", conn.is_closed))
        return conn


def gzip_body(payload: bytes) -> bytes:
    compressor = zlib.compressobj(6, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
    return compressor.compress(payload) + compressor.flush()


def build_response(
    pool: RecordingConnectionPool,
    raw: bytes,
    headers: dict[str, str],
    *,
    chunked: bool = False,
) -> tuple[HTTPResponse, LedgeredBytesIO, HTTPConnection]:
    """Wire an in-memory framed response through the production read path."""
    conn = pool._get_conn()
    fp = LedgeredBytesIO(raw)

    httplib_response = httplib.HTTPResponse(MockSock, method="GET")
    httplib_response.fp = fp
    if chunked:
        httplib_response.chunked = True
        httplib_response.chunk_left = None
    else:
        httplib_response.chunked = False
        # Mirror the real parser: bound httplib reads by Content-Length so that
        # bytes beyond the body boundary stay in the ledger.
        content_length = headers.get("content-length")
        httplib_response.length = (
            int(content_length) if content_length is not None else None
        )

    response = HTTPResponse(
        httplib_response,
        preload_content=False,
        headers=headers,
        original_response=httplib_response,
        pool=pool,
        connection=conn,
    )
    return response, fp, conn


class TestConnectionReturnSafety:
    def _assert_no_leftover_reuse(
        self,
        pool: RecordingConnectionPool,
        conn: HTTPConnection,
        fp: LedgeredBytesIO,
    ) -> None:
        # The connection returned to the pool must never be handed out again
        # while framing bytes remain in the ledger.
        borrowed = pool._get_conn()
        reused_live = borrowed is conn and not conn.is_closed
        assert not (reused_live and fp.bytes_remaining > 0), (
            "pool reused a connection that still carries "
            f"{fp.bytes_remaining} unconsumed response byte(s)"
        )
        pool.close()

    @pytest.mark.parametrize(
        "consume",
        [
            pytest.param(lambda r: r.read(), id="read"),
            pytest.param(lambda r: r.readinto(bytearray(64)), id="readinto"),
            pytest.param(lambda r: list(r.stream()), id="stream"),
        ],
    )
    def test_early_eof_discards_connection(self, consume: typing.Any) -> None:
        pool = RecordingConnectionPool()
        # Declares ten bytes, only three are present before the stream ends.
        response, fp, conn = build_response(pool, b"abc", {"content-length": "10"})
        with pytest.raises(ProtocolError):
            consume(response)
        # The error catcher must have closed the connection rather than relying
        # on the caller to notice the broken framing later.
        assert conn.is_closed
        response.release_conn()
        assert fp.bytes_remaining == 0
        self._assert_no_leftover_reuse(pool, conn, fp)

    def test_early_eof_after_partial_read_then_close(self) -> None:
        pool = RecordingConnectionPool()
        response, fp, conn = build_response(pool, b"abc", {"content-length": "10"})
        assert response.read(1) == b"a"
        # A caller giving up early explicitly closes the response; leftover
        # framing bytes must keep this connection out of the reusable set.
        response.close()
        assert conn.is_closed
        assert fp.bytes_remaining == 2
        response.release_conn()
        self._assert_no_leftover_reuse(pool, conn, fp)

    @pytest.mark.parametrize(
        "consume",
        [
            pytest.param(lambda r: r.read(), id="read"),
            pytest.param(lambda r: list(r.stream()), id="stream"),
        ],
    )
    def test_truncated_chunked_body_discards_connection(
        self, consume: typing.Any
    ) -> None:
        pool = RecordingConnectionPool()
        # Second chunk declares five bytes but only two arrive, and the
        # terminating zero-length chunk is missing entirely.
        response, fp, conn = build_response(
            pool,
            b"3\r\nfoo\r\n5\r\nba",
            {"transfer-encoding": "chunked"},
            chunked=True,
        )
        with pytest.raises(ProtocolError):
            consume(response)
        assert conn.is_closed
        response.release_conn()
        self._assert_no_leftover_reuse(pool, conn, fp)

    def test_partial_chunk_read_then_close_discards_connection(self) -> None:
        pool = RecordingConnectionPool()
        response, fp, conn = build_response(
            pool,
            b"3\r\nfoo\r\n3\r\nbar\r\n0\r\n\r\n",
            {"transfer-encoding": "chunked"},
            chunked=True,
        )
        chunks = response.stream()
        assert next(chunks) == b"foo"
        response.close()
        assert conn.is_closed
        assert fp.bytes_remaining > 0
        response.release_conn()
        self._assert_no_leftover_reuse(pool, conn, fp)

    @pytest.mark.parametrize(
        "consume",
        [
            pytest.param(lambda r: r.read(), id="read"),
            pytest.param(lambda r: r.read1(), id="read1"),
            pytest.param(lambda r: r.readinto(bytearray(2**16)), id="readinto"),
            pytest.param(lambda r: list(r.stream()), id="stream"),
        ],
    )
    def test_decompression_failure_with_tail_bytes_discards_connection(
        self, consume: typing.Any
    ) -> None:
        good = gzip_body(b"hello world" * 50)
        # Corrupt the gzip footer so the decoder flush fails, and place a second
        # response's bytes beyond the declared Content-Length boundary.
        framed = good[:-5] + b"NEXTMSG"
        content_length = len(good)
        assert len(framed) > content_length  # bytes exist past the body edge

        pool = RecordingConnectionPool()
        response, fp, conn = build_response(
            pool,
            framed,
            {
                "content-length": str(content_length),
                "content-encoding": "gzip",
            },
        )
        with pytest.raises(DecodeError):
            consume(response)
        # The framing bytes inside the body were read (length_remaining == 0),
        # but the decoder never confirmed end of stream and the next-response
        # tail is still in the ledger. Reusing here would cross responses.
        assert response.length_remaining == 0
        assert fp.bytes_remaining == len(framed) - content_length
        response.release_conn()
        self._assert_no_leftover_reuse(pool, conn, fp)

    def test_partial_content_length_read_then_close_discards(self) -> None:
        pool = RecordingConnectionPool()
        response, fp, conn = build_response(
            pool, b"abcdefghij", {"content-length": "10"}
        )
        assert response.read(3) == b"abc"
        response.close()
        assert conn.is_closed
        assert fp.bytes_remaining == 7
        response.release_conn()
        self._assert_no_leftover_reuse(pool, conn, fp)

    def test_corrupt_gzip_inside_chunked_body_discards_connection(self) -> None:
        # Corrupt a middle byte of the gzip stream (a missing footer alone is
        # tolerated by zlib) and wrap it in a chunked transfer encoding. The
        # decode failure must discard the connection even though the chunked
        # framing itself reached its terminating chunk cleanly.
        good = bytearray(gzip_body(b"hello world" * 50))
        good[12] ^= 0xFF
        corrupted = bytes(good)
        raw = f"{len(corrupted):X}\r\n".encode() + corrupted + b"\r\n0\r\n\r\n"

        pool = RecordingConnectionPool()
        response, fp, conn = build_response(
            pool,
            raw,
            {"transfer-encoding": "chunked", "content-encoding": "gzip"},
            chunked=True,
        )
        with pytest.raises(DecodeError):
            list(response.read_chunked(decode_content=True))
        assert conn.is_closed
        response.release_conn()
        self._assert_no_leftover_reuse(pool, conn, fp)

    def test_partial_readinto_then_close_discards(self) -> None:
        pool = RecordingConnectionPool()
        response, fp, conn = build_response(
            pool, b"abcdefghij", {"content-length": "10"}
        )
        buffer = bytearray(4)
        assert response.readinto(buffer) == 4
        assert bytes(buffer) == b"abcd"
        response.close()
        assert conn.is_closed
        assert fp.bytes_remaining == 6
        response.release_conn()
        self._assert_no_leftover_reuse(pool, conn, fp)

    @pytest.mark.parametrize(
        "consume",
        [
            pytest.param(lambda r: r.read(), id="read"),
            pytest.param(lambda r: list(r.stream()), id="stream"),
            pytest.param(
                lambda r: (lambda buffer: [r.readinto(buffer) for _ in range(3)])(
                    bytearray(4)
                ),
                id="readinto",
            ),
        ],
    )
    def test_complete_content_length_response_is_reused(
        self, consume: typing.Any
    ) -> None:
        pool = RecordingConnectionPool()
        response, fp, conn = build_response(
            pool, b"abcdefghij", {"content-length": "10"}
        )
        consume(response)
        assert fp.bytes_remaining == 0
        assert not conn.is_closed
        response.release_conn()

        borrowed = pool._get_conn()
        assert borrowed is conn
        assert not conn.is_closed
        pool.close()

    def test_complete_chunked_response_is_reused(self) -> None:
        pool = RecordingConnectionPool()
        response, fp, conn = build_response(
            pool,
            b"3\r\nfoo\r\n0\r\n\r\n",
            {"transfer-encoding": "chunked"},
            chunked=True,
        )
        assert list(response.stream()) == [b"foo"]
        assert fp.bytes_remaining == 0
        assert not conn.is_closed
        response.release_conn()

        borrowed = pool._get_conn()
        assert borrowed is conn
        assert not conn.is_closed
        pool.close()

    def test_complete_gzip_response_is_reused(self) -> None:
        payload = b"hello world" * 50
        good = gzip_body(payload)
        pool = RecordingConnectionPool()
        response, fp, conn = build_response(
            pool,
            good,
            {"content-length": str(len(good)), "content-encoding": "gzip"},
        )
        assert response.read(decode_content=True) == payload
        assert fp.bytes_remaining == 0
        assert not conn.is_closed
        response.release_conn()

        borrowed = pool._get_conn()
        assert borrowed is conn
        assert not conn.is_closed
        pool.close()

    def test_preloaded_gzip_response_auto_returns_live_connection(self) -> None:
        # Mirrors urlopen(preload_content=True): the body is read during
        # construction, so the connection must be returned automatically even
        # though the caller never invokes release_conn() explicitly. This
        # guards against "fixing" reuse by simply never pooling decoded bodies.
        payload = b"hello world" * 50
        good = gzip_body(payload)
        pool = RecordingConnectionPool()
        conn = pool._get_conn()
        fp = LedgeredBytesIO(good)
        httplib_response = httplib.HTTPResponse(MockSock, method="GET")
        httplib_response.fp = fp
        httplib_response.chunked = False
        httplib_response.length = len(good)
        response = HTTPResponse(
            httplib_response,
            preload_content=True,
            headers={
                "content-length": str(len(good)),
                "content-encoding": "gzip",
            },
            original_response=httplib_response,
            pool=pool,
            connection=conn,
        )
        assert response.data == payload
        assert response._connection is None  # already auto-released
        assert not conn.is_closed

        borrowed = pool._get_conn()
        assert borrowed is conn
        assert not conn.is_closed
        pool.close()

    def test_recording_pool_tracks_borrow_and_return(self) -> None:
        # Sanity check on the harness itself: a full request/response cycle
        # records a new connection, a borrow, a live return and another borrow.
        pool = RecordingConnectionPool()
        response, fp, conn = build_response(
            pool, b"abcdefghij", {"content-length": "10"}
        )
        response.read()
        response.release_conn()
        pool._get_conn()
        assert ("new", False) in pool.events
        assert ("put", False) in pool.events  # returned while still open
        assert any(kind == "get" for kind, _ in pool.events)
        pool.close()
