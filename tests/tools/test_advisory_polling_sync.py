# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from arq.connections import ArqRedis as _ArqRedis
from fakeredis import FakeAsyncRedis

from config import AdvisoryPollInterval, AppConfig, RepoConfig, ReposManifest, Settings
from exceptions import SecurityScoutError
from tools.advisory_polling import (
    _GlobalEnqueueBudget,
    _parse_watermark_value,
    _watermark_iso_utc,
    advisory_list_etag_key,
    advisory_list_watermark_key,
    run_repository_advisories_sync,
    run_repository_advisories_sync_from_worker_ctx,
)
from tools.rate_limiter import SlidingWindowRateLimiter
from tools.scm.models import AdvisoryData

_GH = "GHSA-ABCD-ABCD-ABCD"


@contextlib.contextmanager
def _patch_arq_redis_allows_fakeredis() -> Iterator[None]:
    """`run_repository_advisories_sync_from_worker_ctx` uses ``isinstance(redis, ArqRedis)``; fakeredis is a ``Redis``."""
    with patch("tools.advisory_polling.ArqRedis", (_ArqRedis, FakeAsyncRedis)):
        yield


def test_advisory_list_watermark_key() -> None:
    assert advisory_list_watermark_key(repo_slug="acme/p", state="Triage") == "poll:advisory:wm:acme/p:triage"


def test_advisory_list_etag_key() -> None:
    assert advisory_list_etag_key(repo_slug="acme/p", state="Triage") == "etag:advisory:acme/p:triage"


def test_watermark_round_trip() -> None:
    dt = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
    s = _watermark_iso_utc(dt)
    out = _parse_watermark_value(s)
    assert out == dt
    out2 = _parse_watermark_value(s.encode("utf-8"))
    assert out2 == dt


def test_repos_config_rejects_closed_poll_state() -> None:
    with pytest.raises(ValueError, match="closed"):
        RepoConfig(
            name="r",
            github_org="a",
            github_repo="b",
            slack_channel="C123",
            allowed_workflows=[],
            notify_on_severity=[],
            require_approval_for=[],
            advisory_poll_states=["triage", "closed"],
        )


@pytest.mark.asyncio
async def test_global_enqueue_budget_stops_at_cap() -> None:
    b = _GlobalEnqueueBudget(2)

    async def build() -> str | None:
        return "a"

    r1, h1 = await b.try_one_enqueue(build)
    assert r1 == "a"
    assert h1 is False
    r2, h2 = await b.try_one_enqueue(build)
    assert r2 == "a"
    assert h2 is False
    r3, h3 = await b.try_one_enqueue(build)
    assert r3 is None
    assert h3 is True

    assert b.n == 2


def _minimal_empty_poll_app() -> AppConfig:
    r = RepoConfig(
        name="r",
        github_org="a",
        github_repo="b",
        slack_channel="C123",
        allowed_workflows=[],
        notify_on_severity=[],
        require_approval_for=[],
        advisory_poll_states=[],
    )
    return AppConfig(
        settings=Settings(),
        repos=ReposManifest(repos=[r]),
        repos_yaml_sha256="0" * 64,
        repos_yaml_path=Path("/dev/null"),
    )


@pytest.mark.asyncio
async def test_run_sync_noop_when_no_poll_states() -> None:
    with patch("tools.advisory_polling.GitHubSCMProvider") as ghm:
        app = _minimal_empty_poll_app()
        await run_repository_advisories_sync(
            settings=app.settings,
            app_config=app,
            redis=MagicMock(),
            rate_limiter=MagicMock(),
        )
        ghm.assert_not_called()


@pytest.mark.asyncio
async def test_run_sync_errors_when_no_redis() -> None:
    with patch("tools.advisory_polling._LOG") as mlog:
        r = RepoConfig(
            name="r",
            github_org="a",
            github_repo="b",
            slack_channel="C123",
            allowed_workflows=[],
            notify_on_severity=[],
            require_approval_for=[],
            advisory_poll_states=["triage"],
        )
        app = AppConfig(
            settings=Settings(),
            repos=ReposManifest(repos=[r]),
            repos_yaml_sha256="0" * 64,
            repos_yaml_path=Path("/dev/null"),
        )
        await run_repository_advisories_sync(
            settings=app.settings,
            app_config=app,
            redis=None,
            rate_limiter=MagicMock(),
        )
        mlog.error.assert_called()


def _app_with_triage_poll(*, seed_without_enqueue: bool = False) -> AppConfig:
    r = RepoConfig(
        name="r",
        github_org="Acme",
        github_repo="RR",
        slack_channel="C123",
        allowed_workflows=[],
        notify_on_severity=[],
        require_approval_for=[],
        advisory_poll_states=["triage"],
        advisory_poll_seed_without_enqueue=seed_without_enqueue,
    )
    return AppConfig(
        settings=Settings(),
        repos=ReposManifest(repos=[r]),
        repos_yaml_sha256="0" * 64,
        repos_yaml_path=Path("/dev/null"),
    )


class _FakeScm:
    def __init__(
        self,
        pages: list[tuple[AdvisoryData, ...]],
        *,
        emit_list_etag_before_first_page: str | None = None,
        not_modified_first: bool = False,
        not_modified_only_if_conditional: bool = False,
    ) -> None:
        self._pages = pages
        self._emit_list_etag = emit_list_etag_before_first_page
        self._not_modified_first = not_modified_first
        self._not_modified_only_if_conditional = not_modified_only_if_conditional

    async def __aenter__(self) -> _FakeScm:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def iter_list_advisories(
        self,
        repo_slug: str,
        *,
        state: str | None = None,
        severity: str | None = None,
        per_page: int = 30,
        max_pages: int = 20,
        finding_id: str | None = None,
        workflow_run_id: object = None,
        **kwargs: object,
    ) -> object:
        poll_nm = kwargs.get("poll_on_first_page_not_modified")
        poll_etag = kwargs.get("poll_on_first_page_etag")

        async def gen() -> object:
            poll_first = kwargs.get("poll_first_page_if_none_match")
            if self._not_modified_only_if_conditional and poll_first:
                if poll_nm is not None:
                    await poll_nm()
                return
            if self._not_modified_first:
                if poll_nm is not None:
                    await poll_nm()
                return
            for i, p in enumerate(self._pages):
                if i == 0 and self._emit_list_etag is not None and poll_etag is not None:
                    await poll_etag(self._emit_list_etag)
                yield p

        return gen()


@pytest.mark.asyncio
async def test_sync_persists_list_etag_in_redis_with_ttl() -> None:
    r = FakeAsyncRedis()
    rate_limiter = SlidingWindowRateLimiter(r)
    adv = AdvisoryData(
        ghsa_id=_GH,
        source="repository",
        summary="s",
        description="d",
        updated_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    app = _app_with_triage_poll()
    with (
        patch(
            "tools.advisory_polling.GitHubSCMProvider",
            return_value=_FakeScm([(adv,)], emit_list_etag_before_first_page='"e1"'),
        ),
        patch("tools.advisory_polling.try_enqueue_advisory", new_callable=AsyncMock) as tq,
    ):
        tq.return_value = "job-1"
        await run_repository_advisories_sync(
            settings=app.settings,
            app_config=app,
            redis=r,
            rate_limiter=rate_limiter,
        )
    stored = await r.get("etag:advisory:acme/rr:triage")
    assert stored is not None
    assert stored.decode() == '"e1"'
    ttl = await r.ttl("etag:advisory:acme/rr:triage")
    assert ttl > 0


@pytest.mark.asyncio
async def test_sync_not_modified_short_circuits_without_enqueue() -> None:
    r = FakeAsyncRedis()
    rate_limiter = SlidingWindowRateLimiter(r)
    app = _app_with_triage_poll()
    with (
        patch(
            "tools.advisory_polling.GitHubSCMProvider",
            return_value=_FakeScm([], not_modified_first=True),
        ),
        patch("tools.advisory_polling.try_enqueue_advisory", new_callable=AsyncMock) as tq,
        patch("tools.advisory_polling._LOG") as mlog,
    ):
        await run_repository_advisories_sync(
            settings=app.settings,
            app_config=app,
            redis=r,
            rate_limiter=rate_limiter,
        )
    tq.assert_not_awaited()
    mlog.info.assert_any_call(
        "advisory_poll_etag_not_modified",
        metric_name="advisory_poll_etag_hits_total",
        repo="r",
        state="triage",
    )


@pytest.mark.asyncio
async def test_sync_enqueues_and_sets_watermark() -> None:
    r = FakeAsyncRedis()
    rate_limiter = SlidingWindowRateLimiter(r)
    adv = AdvisoryData(
        ghsa_id=_GH,
        source="repository",
        summary="s",
        description="d",
        updated_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    app = _app_with_triage_poll()
    with (
        patch("tools.advisory_polling.GitHubSCMProvider", return_value=_FakeScm(pages=[(adv,)])),
        patch("tools.advisory_polling.try_enqueue_advisory", new_callable=AsyncMock) as tq,
    ):
        tq.return_value = "job-1"
        await run_repository_advisories_sync(
            settings=app.settings,
            app_config=app,
            redis=r,
            rate_limiter=rate_limiter,
        )
    tq.assert_awaited()
    wm = await r.get("poll:advisory:wm:acme/rr:triage")
    assert wm is not None
    assert len(wm) > 0


@pytest.mark.asyncio
async def test_sync_seed_only_sets_watermark() -> None:
    r = FakeAsyncRedis()
    rate_limiter = SlidingWindowRateLimiter(r)
    adv = AdvisoryData(
        ghsa_id=_GH,
        source="repository",
        summary="s",
        description="d",
        updated_at=datetime(2024, 2, 1, tzinfo=UTC),
    )
    app = _app_with_triage_poll(seed_without_enqueue=True)
    with (
        patch("tools.advisory_polling.GitHubSCMProvider", return_value=_FakeScm(pages=[(adv,)])),
        patch("tools.advisory_polling.try_enqueue_advisory", new_callable=AsyncMock) as tq,
    ):
        await run_repository_advisories_sync(
            settings=app.settings,
            app_config=app,
            redis=r,
            rate_limiter=rate_limiter,
        )
    tq.assert_not_awaited()
    wm = await r.get("poll:advisory:wm:acme/rr:triage")
    assert wm is not None


@pytest.mark.asyncio
async def test_sync_seed_deletes_stale_etag_on_304_then_sets_watermark() -> None:
    r = FakeAsyncRedis()
    await r.set("etag:advisory:acme/rr:triage", b'"stale"')
    rate_limiter = SlidingWindowRateLimiter(r)
    adv = AdvisoryData(
        ghsa_id=_GH,
        source="repository",
        summary="s",
        description="d",
        updated_at=datetime(2024, 3, 1, tzinfo=UTC),
    )
    app = _app_with_triage_poll(seed_without_enqueue=True)
    with (
        patch(
            "tools.advisory_polling.GitHubSCMProvider",
            return_value=_FakeScm(
                pages=[(adv,)],
                not_modified_only_if_conditional=True,
                emit_list_etag_before_first_page='"new"',
            ),
        ),
        patch("tools.advisory_polling.try_enqueue_advisory", new_callable=AsyncMock) as tq,
    ):
        await run_repository_advisories_sync(
            settings=app.settings,
            app_config=app,
            redis=r,
            rate_limiter=rate_limiter,
        )
    tq.assert_not_awaited()
    wm = await r.get("poll:advisory:wm:acme/rr:triage")
    assert wm is not None
    assert await r.get("etag:advisory:acme/rr:triage") == b'"new"'


@pytest.mark.asyncio
async def test_sync_stops_before_watermark_without_enqueue() -> None:
    r = FakeAsyncRedis()
    await r.set("poll:advisory:wm:acme/rr:triage", _watermark_iso_utc(datetime(2025, 1, 1, tzinfo=UTC)))
    rate_limiter = SlidingWindowRateLimiter(r)
    adv = AdvisoryData(
        ghsa_id=_GH,
        source="repository",
        summary="s",
        description="d",
        updated_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    app = _app_with_triage_poll()
    with (
        patch("tools.advisory_polling.GitHubSCMProvider", return_value=_FakeScm(pages=[(adv,)])),
        patch("tools.advisory_polling.try_enqueue_advisory", new_callable=AsyncMock) as tq,
    ):
        await run_repository_advisories_sync(
            settings=app.settings,
            app_config=app,
            redis=r,
            rate_limiter=rate_limiter,
        )
    tq.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_sync_marks_tick_partial_on_recoverable_scm_error() -> None:
    """API / transport / Redis errors in one poll state are logged; tick is partial, not a hard fail."""
    r = FakeAsyncRedis()
    rate_limiter = SlidingWindowRateLimiter(r)
    app = _app_with_triage_poll()
    with (
        patch("tools.advisory_polling.GitHubSCMProvider", return_value=_FakeScm(pages=[])),
        patch(
            "tools.advisory_polling._sync_one_poll_state",
            side_effect=SecurityScoutError("upstream"),
        ),
        patch("tools.advisory_polling._LOG") as mlog,
    ):
        await run_repository_advisories_sync(
            settings=app.settings,
            app_config=app,
            redis=r,
            rate_limiter=rate_limiter,
        )
    mlog.exception.assert_called_once()


@pytest.mark.asyncio
async def test_run_sync_skips_non_github_provider() -> None:
    s = Settings()
    s.scm_provider = "gitlab"
    with patch("tools.advisory_polling.GitHubSCMProvider") as ghm:
        await run_repository_advisories_sync(
            settings=s,
            app_config=_minimal_empty_poll_app(),
            redis=FakeAsyncRedis(),
            rate_limiter=SlidingWindowRateLimiter(FakeAsyncRedis()),
        )
        ghm.assert_not_called()


@pytest.mark.asyncio
async def test_run_sync_from_worker_ctx_invokes_sync() -> None:
    r = FakeAsyncRedis()
    rl = SlidingWindowRateLimiter(r)
    ctx = {
        "settings": Settings(),
        "app_config": _minimal_empty_poll_app(),
        "redis": r,
        "rate_limiter": rl,
    }
    with (
        _patch_arq_redis_allows_fakeredis(),
        patch("tools.advisory_polling.GitHubSCMProvider"),
    ):
        await run_repository_advisories_sync_from_worker_ctx(ctx)


@pytest.mark.asyncio
async def test_run_sync_from_worker_ctx_rejects_incomplete_context() -> None:
    with patch("tools.advisory_polling._LOG") as mlog:
        await run_repository_advisories_sync_from_worker_ctx({})
        mlog.error.assert_called()


@pytest.mark.asyncio
async def test_run_sync_from_worker_ctx_rejects_non_arq_redis() -> None:
    with (
        patch("tools.advisory_polling._LOG") as mlog,
        patch("tools.advisory_polling.run_repository_advisories_sync") as rsync,
    ):
        await run_repository_advisories_sync_from_worker_ctx(
            {
                "settings": Settings(),
                "app_config": _minimal_empty_poll_app(),
                "redis": "not-redis",
                "rate_limiter": SlidingWindowRateLimiter(FakeAsyncRedis()),
            }
        )
    mlog.error.assert_called()
    rsync.assert_not_called()


def _settings_with_secrets(*, interval: AdvisoryPollInterval = AdvisoryPollInterval.disabled) -> Settings:
    return Settings(
        github_webhook_secret="a",
        github_pat="b",
        slack_bot_token="c",
        slack_signing_secret="d",
        advisory_poll_interval=interval,
    )


@pytest.mark.asyncio
async def test_run_sync_from_worker_ctx_skips_when_interval_set_but_polling_not_enabled() -> None:
    with (
        _patch_arq_redis_allows_fakeredis(),
        patch("tools.advisory_polling._LOG") as mlog,
        patch("tools.advisory_polling.run_repository_advisories_sync") as rsync,
    ):
        r = FakeAsyncRedis()
        await run_repository_advisories_sync_from_worker_ctx(
            {
                "settings": _settings_with_secrets(interval=AdvisoryPollInterval.hourly),
                "app_config": _minimal_empty_poll_app(),
                "redis": r,
                "rate_limiter": SlidingWindowRateLimiter(r),
            }
        )
    rsync.assert_not_called()
    mlog.warning.assert_called_once()


@pytest.mark.asyncio
async def test_run_sync_from_worker_ctx_invokes_sync_when_interval_set_and_polling_enabled() -> None:
    with (
        _patch_arq_redis_allows_fakeredis(),
        patch("tools.advisory_polling.run_repository_advisories_sync", new_callable=AsyncMock) as rsync,
    ):
        r = FakeAsyncRedis()
        await run_repository_advisories_sync_from_worker_ctx(
            {
                "settings": _settings_with_secrets(interval=AdvisoryPollInterval.hourly),
                "app_config": _minimal_empty_poll_app(),
                "redis": r,
                "rate_limiter": SlidingWindowRateLimiter(r),
                "advisory_polling_enabled": True,
            }
        )
    rsync.assert_awaited_once()
