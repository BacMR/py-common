"""One DSN and one AMQP URL for every py-common integration test -- and a refusal.

These three modules (``test_consumer``, ``test_poller``, ``test_skeleton_smoke``)
declare exchanges, declare queues, publish and consume. Until now each of them
hardcoded ``amqp://bacmr:bacmr_dev@127.0.0.1:5673/`` with no way to override it,
and ``127.0.0.1:5673`` is ``bacmr-rabbitmq`` -- the **production broker**
(the VPS "Ports (host-published)" table). The only reason
the suite had not already declared ``test.consumer.*`` queues and pumped
dead-letters through production is that the Postgres they also hardcoded
(``127.0.0.1:5433``, the dev compose profile) is usually down, so the fixture
errored before the broker was touched. That is an accident, not a safeguard:
``just up`` followed by ``pytest -m integration`` was enough to spend it.

So endpoints come from here, and here refuses production rather than warning:

- **defaults are the isolated ``bacmr-it`` stack** (Postgres 5533, AMQP 5773),
  not the dev profile and certainly not production. A default is how a test suite ends
  up somewhere nobody chose;
- **every production port is denied on every host.** Not just ``127.0.0.1``:
  ``localhost``, ``host.docker.internal``, a container name, an IPv6 literal.
  The simplest rule is the one that cannot be spelled around;
- **an implicit port is denied too**, because ``amqp://host/`` *means* 5672 and
  ``postgresql://host/db`` *means* 5432 -- omitting the port is the shortest
  route to production, not a way to dodge the check;
- **there is no override.** No ``ALLOW_PROD``, no flag, no keyword argument. An
  escape hatch here gets used exactly once, in a hurry, at the worst moment. A
  new stack that needs a broker picks a port outside the table below.

The default is the isolated 5533 rather than the dev profile's 5433 because these
tests mutate ``content.outbox`` / ``content.processed_events`` and must never
share a database with anything that matters. The rule the denylist borrows from
is ``evals/bench/load/targets.py`` (ports denied on every host, no override).

**This module is now the whole repo's test-endpoint resolver, not just
py-common's.** ``pyproject.toml``'s ``[tool.pytest.ini_options] pythonpath`` puts
this directory on ``sys.path`` for every suite, so ``services/knowledge/tests``,
``services/tutor/tests``, ``deploy/tests`` (via ``_endpoints.py``) and
``evals/tests/bench/_dsn.py`` all reach these two resolvers and the denylist
below instead of each keeping a copy. The service suites call ``role_dsn`` for
the one thing the environment cannot decide for them: which Postgres ROLE the
test needs. See that function for why that has to be a property of the test.
"""

from __future__ import annotations

import os
from urllib.parse import parse_qsl, urlsplit, urlunsplit

# ---------------------------------------------------------------------------
# The denylist. Source: the "Ports (host-published)" table and the "Databases --
# one host Postgres (aaPanel, /www/server/pgsql, port 5432)" facts of the VPS
# docs (vps-setup.bacmr.com, re-checked 2026-10-31). Keep the two in sync: a
# production port that is in that table and not in this map is a port these
# tests would happily declare queues on.
# ---------------------------------------------------------------------------

# Brokers. 5673 is the production broker (gateway + knowledge + tutor + scraper
# traffic for the students using bacmr.com); 5672 is the AMQP default port, which
# an implicit-port URL resolves to, and stays denied so no future broker
# published there is ever reached by accident. Declaring a queue or a
# dead-letter exchange on a running broker is a change to a running system.
DENIED_AMQP_PORTS: dict[int, str] = {
    5672: "default RabbitMQ AMQP port -- reserved, never a test target",
    5673: "production RabbitMQ AMQP (bacmr-rabbitmq) -- SERVING PRODUCTION",
}

# The one host Postgres engine. A single port, but four databases behind it:
# bacmr_rag and bacmr_v2 (the service schemas, serving students), bacmr_v2_zitadel
# (identity) and litellm_v2 (the spend ledger). These tests run TRUNCATE.
DENIED_POSTGRES_PORTS: dict[int, str] = {
    5432: "host aaPanel Postgres engine -- bacmr_rag, bacmr_v2, bacmr_v2_zitadel, litellm_v2",
}

# Applied as one list to both kinds of URL. A Postgres DSN on 5673 is nonsense
# and an AMQP URL on 5432 is nonsense, so checking the union costs nothing and
# removes the "which list applies here?" question -- the class of mistake this
# module exists to stop is exactly a URL pointed at the wrong service.
DENIED_PORTS: dict[int, str] = {**DENIED_AMQP_PORTS, **DENIED_POSTGRES_PORTS}

# 5433 (dev compose profile) is deliberately NOT denied: it is a disposable
# stack a developer starts with `just up`, not something anyone is served from.
# It is not the default here either -- see the module docstring.

# Query keys that would move the connection somewhere other than the host and
# port in the URL. asyncpg and libpq both honour `host`/`port`/`hostaddr` as
# connection parameters, so `...@127.0.0.1:5533/bacmr?host=/var/run/postgresql`
# would sail past a port check and land wherever that socket leads.
REDIRECTING_QUERY_KEYS: frozenset[str] = frozenset({"host", "hostaddr", "port"})

# Ports published by the isolated `bacmr-it` project (deploy/compose.itest.yaml,
# `just itest-up`), which is what these tests are allowed to mutate.
DEFAULT_DSN = "postgresql+asyncpg://knowledge_svc:knowledge_dev@127.0.0.1:5533/bacmr"
DEFAULT_AMQP_URL = "amqp://bacmr:bacmr_dev@127.0.0.1:5773/"

# The port a scheme falls back to when the URL omits one. Both land on the
# denylist, which is the point of naming them.
IMPLICIT_PORTS: dict[str, int] = {
    "amqp": 5672,
    "amqps": 5671,
    "postgres": 5432,
    "postgresql": 5432,
}


class ProductionEndpointRefused(ValueError):
    """The URL points at a production broker or the production Postgres engine."""


def _scheme_family(scheme: str) -> str:
    """``postgresql+asyncpg`` -> ``postgresql``; the driver suffix is not a scheme."""
    return scheme.split("+", 1)[0].lower()


def vet_endpoint(url: str) -> str:
    """Return *url* unchanged, or raise :class:`ProductionEndpointRefused`.

    The single entry point on purpose: no test parses a host itself, so there is
    exactly one place where an endpoint becomes usable and one place to audit.
    """
    candidate = url.strip()
    if not candidate:
        raise ProductionEndpointRefused("endpoint is empty; name the isolated stack explicitly")

    try:
        parsed = urlsplit(candidate)
    except ValueError as exc:  # e.g. an unbracketed IPv6 literal
        raise ProductionEndpointRefused(
            f"cannot parse {candidate!r} as a URL ({exc}); a URL the guard cannot read "
            "is a URL it cannot clear"
        ) from exc

    family = _scheme_family(parsed.scheme)
    if not family:
        raise ProductionEndpointRefused(
            f"cannot read a scheme from {candidate!r}; spell the URL in full "
            "(amqp://... or postgresql+asyncpg://...) so it can be checked"
        )

    try:
        port = parsed.port
    except ValueError as exc:  # non-numeric or out-of-range port
        raise ProductionEndpointRefused(f"cannot read a port from {candidate!r}: {exc}") from exc

    host = (parsed.hostname or "").lower()
    if not host:
        raise ProductionEndpointRefused(
            f"cannot read a host from {candidate!r}; name the isolated stack explicitly"
        )

    for key, _value in parse_qsl(parsed.query, keep_blank_values=True):
        if key.lower() in REDIRECTING_QUERY_KEYS:
            raise ProductionEndpointRefused(
                f"refusing {candidate!r}: the {key!r} connection parameter moves the "
                "connection off the host and port in the URL, so the port check below "
                "would prove nothing. Put the real host in the URL."
            )

    if port is None:
        implicit = IMPLICIT_PORTS.get(family)
        if implicit is None:
            raise ProductionEndpointRefused(
                f"refusing {candidate!r}: no port given and no default known for "
                f"scheme {family!r}. Name the isolated stack's port."
            )
        raise ProductionEndpointRefused(
            f"refusing {candidate!r}: no port given, so {family}:// means port "
            f"{implicit} -- {DENIED_PORTS.get(implicit, 'a port nobody chose')}. "
            "Name the isolated stack's port explicitly."
        )

    if port in DENIED_PORTS:
        raise ProductionEndpointRefused(
            f"refusing {host}:{port}: port {port} is {DENIED_PORTS[port]} "
            "(VPS docs, vps-setup.bacmr.com). These tests declare queues, publish "
            "events and TRUNCATE tables -- point them at the isolated bacmr-it stack "
            f"({DEFAULT_AMQP_URL} / {DEFAULT_DSN}). There is no override."
        )
    return candidate


def resolve_dsn() -> str:
    """The SQLAlchemy DSN for the py-common integration fixtures.

    ``TEST_DATABASE_URL`` is the variable the rest of this repo already uses for
    exactly this (``services/knowledge/tests/test_retrieval_integration.py``,
    ``evals/tests/bench/_dsn.py``), and it is expected in SQLAlchemy spelling
    here because these fixtures build an ``AsyncEngine``.
    """
    return vet_endpoint(os.environ.get("TEST_DATABASE_URL") or DEFAULT_DSN)


def resolve_amqp_url() -> str:
    """The broker for the py-common integration fixtures.

    Deliberately **not** ``AMQP_URL``: that name is the services' own runtime
    setting, and several unit tests monkeypatch it to a throwaway value. A test
    broker needs its own variable so no service env can steer this suite.
    """
    return vet_endpoint(os.environ.get("TEST_AMQP_URL") or DEFAULT_AMQP_URL)


def role_dsn(user: str, password: str, database: str = "bacmr") -> str:
    """The resolved DSN's **server**, with the role and database named by the caller.

    One exported ``TEST_DATABASE_URL`` has to drive suites that connect as
    different roles: the production Postgres isolates ``knowledge_svc`` from
    ``tutor_svc`` on purpose (per-schema grants plus ``pg_hba.conf``, and
    ``deploy/tests/test_db_isolation.py`` asserts tutor cannot touch
    ``content``). Before this existed, ``services/tutor/tests`` and
    ``services/knowledge/tests`` both read that one variable and each expected
    its own role in it, so **no single value could make the suite pass** -- the
    knowledge role gave the tutor tests ``permission denied for schema tutor``
    and the tutor role broke the knowledge tests instead.

    So the variable carries the *server* (host and port), which is the part an
    operator chooses, and each test module names the role it needs, which is a
    property of the test rather than of the environment. The logins passed in
    are the isolated stack's throwaway dev ones (``deploy/dev.env``, committed).

    Vetted on the way in and again on the way out, so a substitution cannot
    produce an endpoint the guard has not seen. ``deploy/tests/_endpoints.py``
    re-exports this function rather than keeping a second implementation.
    """
    base = resolve_dsn()
    parts = urlsplit(base)
    if parts.port is None:  # pragma: no cover - resolve_dsn refuses a portless DSN
        raise ProductionEndpointRefused(f"no port in {base!r}")
    host = parts.hostname or ""
    if ":" in host:  # an IPv6 literal has to go back between brackets
        host = f"[{host}]"
    netloc = f"{user}:{password}@{host}:{parts.port}"
    return vet_endpoint(urlunsplit((parts.scheme, netloc, f"/{database}", "", "")))


DSN = resolve_dsn()
AMQP_URL = resolve_amqp_url()
