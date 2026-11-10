from __future__ import annotations

import datetime
from test import DUMMY_POOL
from unittest import mock

import pytest

from urllib3.exceptions import (
    ConnectTimeoutError,
    DeadlineExceededError,
    InvalidHeader,
    MaxRetryError,
    NewConnectionError,
    ReadTimeoutError,
    ResponseError,
    SSLError,
)
from urllib3.connection import HTTPConnection
from urllib3.connectionpool import HTTPConnectionPool
from urllib3.response import HTTPResponse
from urllib3.util import retry as retry_module
from urllib3.util import timeout as timeout_module
from urllib3.util.retry import RequestHistory, Retry
from urllib3.util.timeout import Timeout


class FakeClock:
    """Controllable replacement for the monotonic clock and sleep.

    Drives both monotonic clocks used by the retry budget and the per-attempt
    timeout, and records every sleep so waits can be asserted on deterministically.
    """

    def __init__(self, start: float = 0.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(seconds)


class TestRetry:
    def test_string(self) -> None:
        """Retry string representation looks the way we expect"""
        retry = Retry()
        assert (
            str(retry)
            == "Retry(total=10, connect=None, read=None, redirect=None, status=None)"
        )
        for _ in range(3):
            retry = retry.increment(method="GET")
        assert (
            str(retry)
            == "Retry(total=7, connect=None, read=None, redirect=None, status=None)"
        )

    def test_retry_both_specified(self) -> None:
        """Total can win if it's lower than the connect value"""
        error = ConnectTimeoutError()
        retry = Retry(connect=3, total=2)
        retry = retry.increment(error=error)
        retry = retry.increment(error=error)
        with pytest.raises(MaxRetryError) as e:
            retry.increment(error=error)
        assert e.value.reason == error

    def test_retry_higher_total_loses(self) -> None:
        """A lower connect timeout than the total is honored"""
        error = ConnectTimeoutError()
        retry = Retry(connect=2, total=3)
        retry = retry.increment(error=error)
        retry = retry.increment(error=error)
        with pytest.raises(MaxRetryError):
            retry.increment(error=error)

    def test_retry_higher_total_loses_vs_read(self) -> None:
        """A lower read timeout than the total is honored"""
        error = ReadTimeoutError(DUMMY_POOL, "/", "read timed out")
        retry = Retry(read=2, total=3)
        retry = retry.increment(method="GET", error=error)
        retry = retry.increment(method="GET", error=error)
        with pytest.raises(MaxRetryError):
            retry.increment(method="GET", error=error)

    def test_retry_total_none(self) -> None:
        """if Total is none, connect error should take precedence"""
        error = ConnectTimeoutError()
        retry = Retry(connect=2, total=None)
        retry = retry.increment(error=error)
        retry = retry.increment(error=error)
        with pytest.raises(MaxRetryError) as e:
            retry.increment(error=error)
        assert e.value.reason == error

        timeout_error = ReadTimeoutError(DUMMY_POOL, "/", "read timed out")
        retry = Retry(connect=2, total=None)
        retry = retry.increment(method="GET", error=timeout_error)
        retry = retry.increment(method="GET", error=timeout_error)
        retry = retry.increment(method="GET", error=timeout_error)
        assert not retry.is_exhausted()

    def test_retry_default(self) -> None:
        """If no value is specified, should retry connects 3 times"""
        retry = Retry()
        assert retry.total == 10
        assert retry.connect is None
        assert retry.read is None
        assert retry.redirect is None
        assert retry.other is None

        error = ConnectTimeoutError()
        retry = Retry(connect=1)
        retry = retry.increment(error=error)
        with pytest.raises(MaxRetryError):
            retry.increment(error=error)

        retry = Retry(connect=1)
        retry = retry.increment(error=error)
        assert not retry.is_exhausted()

        assert Retry(0).raise_on_redirect
        assert not Retry(False).raise_on_redirect

    def test_retry_other(self) -> None:
        """If an unexpected error is raised, should retry other times"""
        other_error = SSLError()
        retry = Retry(connect=1)
        retry = retry.increment(error=other_error)
        retry = retry.increment(error=other_error)
        assert not retry.is_exhausted()

        retry = Retry(other=1)
        retry = retry.increment(error=other_error)
        with pytest.raises(MaxRetryError) as e:
            retry.increment(error=other_error)
        assert e.value.reason == other_error

    def test_retry_read_zero(self) -> None:
        """No second chances on read timeouts, by default"""
        error = ReadTimeoutError(DUMMY_POOL, "/", "read timed out")
        retry = Retry(read=0)
        with pytest.raises(MaxRetryError) as e:
            retry.increment(method="GET", error=error)
        assert e.value.reason == error

    def test_status_counter(self) -> None:
        resp = HTTPResponse(status=400)
        retry = Retry(status=2)
        retry = retry.increment(response=resp)
        retry = retry.increment(response=resp)
        msg = ResponseError.SPECIFIC_ERROR.format(status_code=400)
        with pytest.raises(MaxRetryError, match=msg):
            retry.increment(response=resp)

    def test_backoff(self) -> None:
        """Backoff is computed correctly"""
        max_backoff = Retry.DEFAULT_BACKOFF_MAX

        retry = Retry(total=100, backoff_factor=0.2)
        assert retry.get_backoff_time() == 0  # First request

        retry = retry.increment(method="GET")
        assert retry.get_backoff_time() == 0  # First retry

        retry = retry.increment(method="GET")
        assert retry.backoff_factor == 0.2
        assert retry.total == 98
        assert retry.get_backoff_time() == 0.4  # Start backoff

        retry = retry.increment(method="GET")
        assert retry.get_backoff_time() == 0.8

        retry = retry.increment(method="GET")
        assert retry.get_backoff_time() == 1.6

        for _ in range(10):
            retry = retry.increment(method="GET")

        assert retry.get_backoff_time() == max_backoff

    def test_configurable_backoff_max(self) -> None:
        """Configurable backoff is computed correctly"""
        max_backoff = 1

        retry = Retry(total=100, backoff_factor=0.2, backoff_max=max_backoff)
        assert retry.get_backoff_time() == 0  # First request

        retry = retry.increment(method="GET")
        assert retry.get_backoff_time() == 0  # First retry

        retry = retry.increment(method="GET")
        assert retry.backoff_factor == 0.2
        assert retry.total == 98
        assert retry.get_backoff_time() == 0.4  # Start backoff

        retry = retry.increment(method="GET")
        assert retry.get_backoff_time() == 0.8

        retry = retry.increment(method="GET")
        assert retry.get_backoff_time() == max_backoff

        retry = retry.increment(method="GET")
        assert retry.get_backoff_time() == max_backoff

    @pytest.mark.parametrize(
        "factor, consecutive_errors, expected",
        [
            (0.0, 1025, 0.0),
            (0.5, 1025, 120.0),
            (-0.5, 1025, 0.0),
            (1.0, 1025, 120.0),
            (-1.0, 1025, 0.0),
            (2.0**-1024, 1025, 1.0),
            (2**1024, 2, 120.0),
            (-(2**1024), 2, 0.0),
        ],
    )
    def test_backoff_overflow(
        self, factor: float, consecutive_errors: int, expected: float
    ) -> None:
        retry = Retry(total=None, backoff_factor=factor)
        for _ in range(consecutive_errors):
            retry = retry.increment(method="GET")
        assert retry.get_backoff_time() == expected

    def test_zero_backoff_large_retry_count_with_jitter(self) -> None:
        history = (RequestHistory("GET", "/", None, 503, None),) * 1025
        retry = Retry(backoff_factor=0.0, backoff_jitter=0.4, history=history)
        with mock.patch("random.random", return_value=0.5):
            assert retry.get_backoff_time() == 0.2

    def test_configurable_retry_after_max(self) -> None:
        """Configurable retry after is computed correctly"""
        max_retry_after = Retry.DEFAULT_RETRY_AFTER_MAX

        retry = Retry()
        assert retry.parse_retry_after(str(max_retry_after)) == max_retry_after
        assert retry.parse_retry_after(str(max_retry_after + 1)) == max_retry_after

        retry = Retry(retry_after_max=1)
        assert retry.parse_retry_after(str(1)) == 1
        assert retry.parse_retry_after(str(2)) == 1

    def test_backoff_jitter(self) -> None:
        """Backoff with jitter is computed correctly"""
        max_backoff = 1
        jitter = 0.4
        retry = Retry(
            total=100,
            backoff_factor=0.2,
            backoff_max=max_backoff,
            backoff_jitter=jitter,
        )
        assert retry.get_backoff_time() == 0  # First request

        retry = retry.increment(method="GET")
        assert retry.get_backoff_time() == 0  # First retry

        retry = retry.increment(method="GET")
        assert retry.backoff_factor == 0.2
        assert retry.total == 98
        assert 0.4 <= retry.get_backoff_time() <= 0.8  # Start backoff

        retry = retry.increment(method="GET")
        assert 0.8 <= retry.get_backoff_time() <= max_backoff

        retry = retry.increment(method="GET")
        assert retry.get_backoff_time() == max_backoff

        retry = retry.increment(method="GET")
        assert retry.get_backoff_time() == max_backoff

    def test_zero_backoff(self) -> None:
        retry = Retry()
        assert retry.get_backoff_time() == 0
        retry = retry.increment(method="GET")
        retry = retry.increment(method="GET")
        assert retry.get_backoff_time() == 0

    def test_backoff_reset_after_redirect(self) -> None:
        retry = Retry(total=100, redirect=5, backoff_factor=0.2)
        retry = retry.increment(method="GET")
        retry = retry.increment(method="GET")
        assert retry.get_backoff_time() == 0.4
        redirect_response = HTTPResponse(status=302, headers={"location": "test"})
        retry = retry.increment(method="GET", response=redirect_response)
        assert retry.get_backoff_time() == 0
        retry = retry.increment(method="GET")
        retry = retry.increment(method="GET")
        assert retry.get_backoff_time() == 0.4

    def test_sleep(self) -> None:
        # sleep a very small amount of time so our code coverage is happy
        retry = Retry(backoff_factor=0.0001)
        retry = retry.increment(method="GET")
        retry = retry.increment(method="GET")
        retry.sleep()

    def test_status_forcelist(self) -> None:
        retry = Retry(status_forcelist=range(500, 600))
        assert not retry.is_retry("GET", status_code=200)
        assert not retry.is_retry("GET", status_code=400)
        assert retry.is_retry("GET", status_code=500)

        retry = Retry(total=1, status_forcelist=[418])
        assert not retry.is_retry("GET", status_code=400)
        assert retry.is_retry("GET", status_code=418)

        # String status codes are not matched.
        retry = Retry(total=1, status_forcelist=["418"])  # type: ignore[list-item]
        assert not retry.is_retry("GET", status_code=418)

    def test_allowed_methods_with_status_forcelist(self) -> None:
        # None allowed_methods means to retry on any method.
        retry = Retry(status_forcelist=[500], allowed_methods=None)
        assert retry.is_retry("GET", status_code=500)
        assert retry.is_retry("POST", status_code=500)

        # Criteria of allowed_methods and status_forcelist are ANDed.
        retry = Retry(status_forcelist=[500], allowed_methods=["POST"])
        assert not retry.is_retry("GET", status_code=500)
        assert retry.is_retry("POST", status_code=500)

    def test_false_allowed_methods(self) -> None:
        # False is also accepted to mean retry on any method, since 1.26.x
        retry = Retry(status_forcelist=[500], allowed_methods=False)  # type: ignore[arg-type]
        assert retry.is_retry("POST", status_code=500)

    def test_empty_allowed_methods(self) -> None:
        with pytest.warns(FutureWarning) as records:
            retry = Retry(status_forcelist=[500], allowed_methods=[])
        assert retry.is_retry("GET", status_code=500)
        msg = (
            "Using an empty collection for 'allowed_methods' option to retry "
            "on any verb is deprecated and will skip retries for all verbs "
            "in urllib3 v3.0. Instead use Retry(..., allowed_methods=None)."
        )
        record = records[0]
        assert isinstance(record.message, Warning)
        assert record.message.args[0] == msg

    def test_exhausted(self) -> None:
        assert not Retry(0).is_exhausted()
        assert Retry(-1).is_exhausted()
        assert Retry(1).increment(method="GET").total == 0

    @pytest.mark.parametrize("total", [-1, 0])
    def test_disabled(self, total: int) -> None:
        with pytest.raises(MaxRetryError):
            Retry(total).increment(method="GET")

    def test_error_message(self) -> None:
        retry = Retry(total=0)
        with pytest.raises(MaxRetryError, match="read timed out") as e:
            retry = retry.increment(
                method="GET", error=ReadTimeoutError(DUMMY_POOL, "/", "read timed out")
            )
        assert "Caused by redirect" not in str(e.value)

        retry = Retry(total=1)
        retry = retry.increment("POST", "/")
        with pytest.raises(MaxRetryError, match=ResponseError.GENERIC_ERROR) as e:
            retry = retry.increment("POST", "/")
        assert "Caused by redirect" not in str(e.value)
        assert isinstance(e.value.reason, ResponseError)

        retry = Retry(total=1)
        response = HTTPResponse(status=500)
        msg = ResponseError.SPECIFIC_ERROR.format(status_code=500)
        retry = retry.increment("POST", "/", response=response)
        with pytest.raises(MaxRetryError, match=msg) as e:
            retry = retry.increment("POST", "/", response=response)
        assert "Caused by redirect" not in str(e.value)

        retry = Retry(connect=1)
        retry = retry.increment(error=ConnectTimeoutError("conntimeout"))
        with pytest.raises(MaxRetryError, match="conntimeout") as e:
            retry = retry.increment(error=ConnectTimeoutError("conntimeout"))
        assert "Caused by redirect" not in str(e.value)

    def test_history(self) -> None:
        retry = Retry(total=10, allowed_methods=frozenset(["GET", "POST"]))
        assert retry.history == tuple()
        connection_error = ConnectTimeoutError("conntimeout")
        retry = retry.increment("GET", "/test1", None, connection_error)
        test_history1 = (RequestHistory("GET", "/test1", connection_error, None, None),)
        assert retry.history == test_history1

        read_error = ReadTimeoutError(DUMMY_POOL, "/test2", "read timed out")
        retry = retry.increment("POST", "/test2", None, read_error)
        test_history2 = (
            RequestHistory("GET", "/test1", connection_error, None, None),
            RequestHistory("POST", "/test2", read_error, None, None),
        )
        assert retry.history == test_history2

        response = HTTPResponse(status=500)
        retry = retry.increment("GET", "/test3", response, None)
        test_history3 = (
            RequestHistory("GET", "/test1", connection_error, None, None),
            RequestHistory("POST", "/test2", read_error, None, None),
            RequestHistory("GET", "/test3", None, 500, None),
        )
        assert retry.history == test_history3

    def test_retry_method_not_allowed(self) -> None:
        error = ReadTimeoutError(DUMMY_POOL, "/", "read timed out")
        retry = Retry()
        with pytest.raises(ReadTimeoutError):
            retry.increment(method="POST", error=error)

    def test_retry_default_remove_headers_on_redirect(self) -> None:
        retry = Retry()

        assert retry.remove_headers_on_redirect == {
            "authorization",
            "proxy-authorization",
            "cookie",
        }

    def test_retry_set_remove_headers_on_redirect(self) -> None:
        retry = Retry(remove_headers_on_redirect=["X-API-Secret"])

        assert retry.remove_headers_on_redirect == {"x-api-secret"}

    @pytest.mark.parametrize(
        "value",
        [
            "-1",
            "+1",
            "1.0",
            "\xb2",  # \xb2 = ^2
            "Thu, 01 Jan 10000 00:00:00 GMT",
            "Wed, 21 Oct 99999999999999999999 07:28:00 GMT",
        ],
    )
    def test_parse_retry_after_invalid(self, value: str) -> None:
        retry = Retry()
        with pytest.raises(InvalidHeader):
            retry.parse_retry_after(value)

    @pytest.mark.parametrize(
        "value, expected", [("0", 0), ("1000", 1000), ("\t42 ", 42)]
    )
    def test_parse_retry_after(self, value: str, expected: int) -> None:
        retry = Retry()
        assert retry.parse_retry_after(value) == expected

    @pytest.mark.parametrize("respect_retry_after_header", [True, False])
    def test_respect_retry_after_header_propagated(
        self, respect_retry_after_header: bool
    ) -> None:
        retry = Retry(respect_retry_after_header=respect_retry_after_header)
        new_retry = retry.new()
        assert new_retry.respect_retry_after_header == respect_retry_after_header

    @pytest.mark.parametrize(
        "retry_after_header,respect_retry_after_header,sleep_duration",
        [
            ("3600", True, 3600),
            ("3600", False, None),
            # Will sleep due to header is 1 hour in future
            ("Mon, 3 Jun 2019 12:00:00 UTC", True, 3600),
            # Won't sleep due to not respecting header
            ("Mon, 3 Jun 2019 12:00:00 UTC", False, None),
            # Won't sleep due to current time reached
            ("Mon, 3 Jun 2019 11:00:00 UTC", True, None),
            # Won't sleep due to current time reached + not respecting header
            ("Mon, 3 Jun 2019 11:00:00 UTC", False, None),
            # Handle all the formats in RFC 7231 Section 7.1.1.1
            ("Mon, 03 Jun 2019 11:30:12 GMT", True, 1812),
            ("Monday, 03-Jun-19 11:30:12 GMT", True, 1812),
            # Assume that datetimes without a timezone are in UTC per RFC 7231
            ("Mon Jun  3 11:30:12 2019", True, 1812),
        ],
    )
    @pytest.mark.parametrize(
        "stub_timezone",
        [
            "UTC",
            "Asia/Jerusalem",
            None,
        ],
        indirect=True,
    )
    @pytest.mark.usefixtures("stub_timezone")
    def test_respect_retry_after_header_sleep(
        self,
        retry_after_header: str,
        respect_retry_after_header: bool,
        sleep_duration: int | None,
    ) -> None:
        retry = Retry(respect_retry_after_header=respect_retry_after_header)

        with (
            mock.patch(
                "time.time",
                return_value=datetime.datetime(
                    2019, 6, 3, 11, tzinfo=datetime.timezone.utc
                ).timestamp(),
            ),
            mock.patch("time.sleep") as sleep_mock,
        ):
            # for the default behavior, it must be in RETRY_AFTER_STATUS_CODES
            response = HTTPResponse(
                status=503, headers={"Retry-After": retry_after_header}
            )

            retry.sleep(response)

            # The expected behavior is that we'll only sleep if respecting
            # this header (since we won't have any backoff sleep attempts)
            if respect_retry_after_header and sleep_duration is not None:
                sleep_mock.assert_called_with(sleep_duration)
            else:
                sleep_mock.assert_not_called()

    def test_deadline_defaults_to_none(self) -> None:
        """Without a deadline, behavior is unchanged and no clock is started."""
        retry = Retry()
        assert retry.deadline is None
        assert retry._deadline_start is None
        assert retry.remaining() is None
        assert not retry.is_deadline_exceeded()
        retry.raise_for_deadline()  # must not raise

    @pytest.mark.parametrize("bad_value", [0, -1, -0.5, True, False, "2", 2j])
    def test_deadline_validation(self, bad_value: object) -> None:
        with pytest.raises(ValueError):
            Retry(deadline=bad_value)  # type: ignore[arg-type]

    def test_deadline_uses_monotonic_clock_and_counts_down(self) -> None:
        clock = FakeClock()
        with (
            mock.patch.object(
                retry_module.time, "monotonic", side_effect=clock.monotonic
            ),
            mock.patch.object(retry_module.time, "sleep", side_effect=clock.sleep),
        ):
            retry = Retry(total=10, deadline=2.0)
            assert retry.remaining() == 2.0  # clock starts on first use
            clock.advance(0.75)
            assert retry.remaining() == 1.25
            clock.advance(2.0)
            assert retry.remaining() == 0.0  # never negative
            assert retry.is_deadline_exceeded()

    def test_deadline_new_does_not_reset_start(self) -> None:
        """Retry.new() and increment() must share one continuous budget."""
        clock = FakeClock()
        with mock.patch.object(
            retry_module.time, "monotonic", side_effect=clock.monotonic
        ):
            retry = Retry(total=10, deadline=2.0)
            retry.remaining()  # start the clock
            clock.advance(1.5)

            clone = retry.new()
            assert clone.deadline == 2.0
            assert clone._deadline_start == retry._deadline_start
            assert clone.remaining() == 0.5

            incremented = retry.increment(method="GET")
            assert incremented._deadline_start == retry._deadline_start
            assert incremented.remaining() == 0.5

    def test_deadline_raise_for_deadline_keeps_reason_and_history(self) -> None:
        clock = FakeClock()
        error = ConnectTimeoutError("conntimeout")
        with mock.patch.object(
            retry_module.time, "monotonic", side_effect=clock.monotonic
        ):
            retry = Retry(total=10, deadline=2.0)
            retry.remaining()
            retry = retry.increment(method="GET", url="/a", error=error)
            clock.advance(2.0)

            with pytest.raises(DeadlineExceededError) as exc_info:
                retry.raise_for_deadline(pool=DUMMY_POOL, url="/a", reason=error)

            raised = exc_info.value
            assert isinstance(raised, MaxRetryError)
            assert raised.reason is error
            assert raised.history == retry.history
            assert raised.history[-1].error is error

    def test_deadline_raise_without_reason_uses_timeout_error(self) -> None:
        from urllib3.exceptions import TimeoutError as Urllib3TimeoutError

        clock = FakeClock()
        with mock.patch.object(
            retry_module.time, "monotonic", side_effect=clock.monotonic
        ):
            retry = Retry(deadline=1.0)
            retry.remaining()
            clock.advance(1.0)
            with pytest.raises(DeadlineExceededError) as exc_info:
                retry.raise_for_deadline()
            assert isinstance(exc_info.value.reason, Urllib3TimeoutError)
            assert exc_info.value.history == ()

    def test_deadline_increment_raises_with_full_history(self) -> None:
        clock = FakeClock()
        error = ConnectTimeoutError("boom")
        with mock.patch.object(
            retry_module.time, "monotonic", side_effect=clock.monotonic
        ):
            retry = Retry(connect=10, total=10, deadline=1.0)
            retry.remaining()
            retry = retry.increment(method="GET", url="/a", error=error)
            clock.advance(1.0)

            with pytest.raises(DeadlineExceededError) as exc_info:
                retry.increment(method="GET", url="/a", error=error, _pool=DUMMY_POOL)

            raised = exc_info.value
            assert raised.reason is error
            # Both attempts, including the failing one that exhausted the budget.
            assert len(raised.history) == 2
            assert all(item.error is error for item in raised.history)

    def test_deadline_sleep_is_clipped_to_remaining_budget(self) -> None:
        clock = FakeClock()
        with (
            mock.patch.object(
                retry_module.time, "monotonic", side_effect=clock.monotonic
            ),
            mock.patch.object(retry_module.time, "sleep", side_effect=clock.sleep),
        ):
            retry = Retry(total=10, deadline=2.0)
            retry.remaining()
            clock.advance(1.5)  # 0.5 left

            retry._sleep(3.0)  # wants 3s, must be shortened to 0.5s
            assert clock.sleeps == [0.5]
            assert retry.is_deadline_exceeded()

    def test_deadline_retry_after_sleep_is_clipped(self) -> None:
        clock = FakeClock()
        with (
            mock.patch.object(
                retry_module.time, "monotonic", side_effect=clock.monotonic
            ),
            mock.patch.object(retry_module.time, "sleep", side_effect=clock.sleep),
        ):
            retry = Retry(deadline=2.0)
            retry.remaining()
            clock.advance(1.75)  # 0.25 left
            response = HTTPResponse(
                status=503, headers={"Retry-After": "10"}
            )
            assert retry.sleep_for_retry(response) is True
            assert clock.sleeps == [0.25]
            assert retry.is_deadline_exceeded()

    def test_deadline_does_not_change_sleep_without_budget(self) -> None:
        clock = FakeClock()
        with (
            mock.patch.object(
                retry_module.time, "monotonic", side_effect=clock.monotonic
            ),
            mock.patch.object(retry_module.time, "sleep", side_effect=clock.sleep),
        ):
            retry = Retry()
            retry._sleep(1.5)
            assert clock.sleeps == [1.5]

    def test_deadline_error_is_picklable_with_history(self) -> None:
        import pickle

        history = (RequestHistory("GET", "/x", None, 503, None),)
        error = DeadlineExceededError(
            DUMMY_POOL, "/x", RuntimeError("boom"), history
        )
        restored = pickle.loads(pickle.dumps(error))
        assert isinstance(restored, DeadlineExceededError)
        assert isinstance(restored, MaxRetryError)
        assert restored.url == "/x"
        assert isinstance(restored.reason, RuntimeError)
        assert restored.history == history


class _RecordingConnectionPool(HTTPConnectionPool):
    """A pool whose sends follow a scripted 503 / redirect / failure sequence."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.sends: list[tuple[float, str]] = []
        self.connection_error = NewConnectionError(
            HTTPConnection("localhost"), "connection refused"
        )
        self.script: list[HTTPResponse | Exception] = [
            HTTPResponse(status=503, headers={"Retry-After": "1"}),
            HTTPResponse(status=302, headers={"location": "/elsewhere"}),
            self.connection_error,
            self.connection_error,
            HTTPResponse(status=200),
        ]

    def _make_request(
        self,
        conn: HTTPConnection,
        method: str,
        url: str,
        *args: object,
        timeout: Timeout,
        **kwargs: object,
    ) -> HTTPResponse:
        self.sends.append((self._clock.now, url))  # type: ignore[attr-defined]
        outcome = self.script[len(self.sends) - 1]
        # Every send (connection + request) consumes half a second of budget.
        self._clock.advance(0.5)  # type: ignore[attr-defined]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class TestRetryDeadlineEndToEnd:
    def test_deadline_stops_further_sends_across_full_sequence(self) -> None:
        clock = FakeClock()
        with (
            mock.patch.object(
                retry_module.time, "monotonic", side_effect=clock.monotonic
            ),
            mock.patch.object(
                timeout_module.time, "monotonic", side_effect=clock.monotonic
            ),
            mock.patch.object(retry_module.time, "sleep", side_effect=clock.sleep),
        ):
            pool = _RecordingConnectionPool(host="localhost")
            pool._clock = clock
            retries = Retry(
                total=10,
                connect=10,
                read=10,
                redirect=10,
                status=10,
                deadline=3.0,
                backoff_factor=0.5,
            )

            with pytest.raises(DeadlineExceededError) as exc_info:
                pool.urlopen(
                    "GET", "/a", retries=retries, assert_same_host=False
                )

        # 503 (with a Retry-After wait), a redirect and two connection
        # failures were actually sent; the final 200 was never attempted.
        assert len(pool.sends) == 4
        assert [url for _, url in pool.sends] == [
            "/a",
            "/a",
            "/elsewhere",
            "/elsewhere",
        ]
        # The Retry-After sleep was honored but could not exceed the budget.
        assert clock.sleeps == [1.0]

        raised = exc_info.value
        assert isinstance(raised, MaxRetryError)
        # The last failure is preserved ...
        assert raised.reason is pool.connection_error
        # ... together with the history of every attempt.
        assert len(raised.history) == 4
        assert raised.history[0].status == 503
        assert raised.history[1].status == 302
        assert raised.history[1].redirect_location == "/elsewhere"
        assert raised.history[2].error is pool.connection_error
        assert raised.history[3].error is pool.connection_error

    def test_deadline_pre_send_check_blocks_even_first_send(self) -> None:
        clock = FakeClock()
        with (
            mock.patch.object(
                retry_module.time, "monotonic", side_effect=clock.monotonic
            ),
            mock.patch.object(
                timeout_module.time, "monotonic", side_effect=clock.monotonic
            ),
        ):
            pool = _RecordingConnectionPool(host="localhost")
            pool._clock = clock
            retries = Retry(deadline=1.0)
            retries.remaining()  # start the clock before entering urlopen
            clock.advance(1.0)  # budget already gone

            with pytest.raises(DeadlineExceededError) as exc_info:
                pool.urlopen("GET", "/a", retries=retries)

        assert pool.sends == []
        assert exc_info.value.pool is pool
        assert exc_info.value.url == "/a"

    def test_no_deadline_keeps_historical_retry_behavior(self) -> None:
        clock = FakeClock()
        with (
            mock.patch.object(
                retry_module.time, "monotonic", side_effect=clock.monotonic
            ),
            mock.patch.object(
                timeout_module.time, "monotonic", side_effect=clock.monotonic
            ),
            mock.patch.object(retry_module.time, "sleep", side_effect=clock.sleep),
        ):
            pool = _RecordingConnectionPool(host="localhost")
            pool._clock = clock
            # Plenty of retry counts, no deadline: the whole script runs and
            # the final 200 response is returned.
            retries = Retry(
                total=10, connect=10, read=10, redirect=10, status=10
            )
            response = pool.urlopen(
                "GET", "/a", retries=retries, assert_same_host=False
            )

        assert response.status == 200
        assert len(pool.sends) == 5

    def test_deadline_folds_remaining_budget_into_attempt_timeout(self) -> None:
        clock = FakeClock()
        seen_timeouts: list[Timeout] = []
        with (
            mock.patch.object(
                retry_module.time, "monotonic", side_effect=clock.monotonic
            ),
            mock.patch.object(
                timeout_module.time, "monotonic", side_effect=clock.monotonic
            ),
        ):
            pool = HTTPConnectionPool(host="localhost")
            retries = Retry(deadline=2.0, connect=10, read=10)

            def fake_make_request(
                conn: HTTPConnection,
                method: str,
                url: str,
                *args: object,
                timeout: Timeout,
                **kwargs: object,
            ) -> HTTPResponse:
                seen_timeouts.append(timeout)
                return HTTPResponse(status=200)

            with mock.patch.object(
                pool, "_make_request", side_effect=fake_make_request
            ):
                pool.urlopen(
                    "GET",
                    "/",
                    retries=retries,
                    timeout=Timeout(connect=5.0, read=5.0),
                )

        assert len(seen_timeouts) == 1
        attempt_timeout = seen_timeouts[0]
        # The configured connect limit (5s) is shortened by the budget (2s),
        # never relaxed.
        assert attempt_timeout.connect_timeout == 2.0
        assert attempt_timeout.total == 2.0
