"""The guard in `_env.py` is the only thing standing between `pytest -m integration`
and the production brokers, so it gets tested like a safety interlock, not like a
helper: every denied port, every host spelling, and an audit that no escape hatch
exists on the public surface.

Unit tests on purpose -- no database, no broker. They run in the default
`pytest -m "not integration"` gate, which is the gate that would be green on the
machine where someone is about to point this suite somewhere it must not go.

Modelled on `evals/tests/bench/test_load.py`, which does the same job for
`evals/bench/load/targets.py`.
"""

import ast
import inspect
import re
from pathlib import Path

import _env
import pytest
from _env import (
    DENIED_AMQP_PORTS,
    DENIED_PORTS,
    DENIED_POSTGRES_PORTS,
    ProductionEndpointRefused,
    resolve_amqp_url,
    resolve_dsn,
    vet_endpoint,
)

TESTS_DIR = Path(_env.__file__).resolve().parent


# --------------------------------------------------------------------------
# 1. Every denied port is refused
# --------------------------------------------------------------------------


@pytest.mark.parametrize("port", sorted(DENIED_PORTS))
def test_every_denied_port_is_refused_as_an_amqp_url(port):
    with pytest.raises(ProductionEndpointRefused) as err:
        vet_endpoint(f"amqp://bacmr:bacmr_dev@127.0.0.1:{port}/")
    assert str(port) in str(err.value)


@pytest.mark.parametrize("port", sorted(DENIED_PORTS))
def test_every_denied_port_is_refused_as_a_postgres_dsn(port):
    """The denylist is one list applied to both kinds of URL: a DSN that has
    landed on a broker port is the same mistake seen from the other side."""
    with pytest.raises(ProductionEndpointRefused) as err:
        vet_endpoint(f"postgresql+asyncpg://knowledge_svc:knowledge_dev@127.0.0.1:{port}/bacmr")
    assert str(port) in str(err.value)


def test_the_denylist_is_the_runbook_table():
    """A regression guard on the list itself: deleting a row would make the
    parametrised tests above pass by giving them nothing to check.

    Source: docs/runbooks/vps-coexistence.md -- the "Ports (host-published)"
    table (5672 legacy broker, 5673 v2 broker) and the "one host Postgres
    (aaPanel, port 5432)" section.
    """
    assert set(DENIED_AMQP_PORTS) == {5672, 5673}
    assert set(DENIED_POSTGRES_PORTS) == {5432}
    assert set(DENIED_PORTS) == {5432, 5672, 5673}


def test_the_denylist_messages_name_what_is_at_stake():
    """Whoever hits this refusal at 23:00 must learn what they nearly touched."""
    assert "PRODUCTION" in DENIED_AMQP_PORTS[5672]
    assert "v2" in DENIED_AMQP_PORTS[5673]
    for db in ("bacmr_rag", "bacmr_v2", "bacmr_v2_zitadel", "litellm_v2"):
        assert db in DENIED_POSTGRES_PORTS[5432]


# --------------------------------------------------------------------------
# 2. The refusal is by port, on every spelling of the host
# --------------------------------------------------------------------------

HOSTS = [
    "127.0.0.1",
    "localhost",
    "host.docker.internal",  # how containers reach the aaPanel engine
    "bacmr-v2-rabbitmq-1",  # a bare container name on a shared docker network
    "rabbitmq",  # the compose service alias
    "[::1]",  # IPv6 literal
    "[2a02:c207:2026:1234::1]",  # IPv6, routable
    "185.185.83.156",  # the VPS itself
    "PROD.EXAMPLE",  # uppercase must not dodge normalisation
]


@pytest.mark.parametrize("host", HOSTS)
@pytest.mark.parametrize("port", sorted(DENIED_PORTS))
def test_refused_on_every_host_with_credentials(host, port):
    with pytest.raises(ProductionEndpointRefused):
        vet_endpoint(f"amqp://bacmr:bacmr_dev@{host}:{port}/")


@pytest.mark.parametrize("host", HOSTS)
@pytest.mark.parametrize("port", sorted(DENIED_PORTS))
def test_refused_on_every_host_without_credentials(host, port):
    """The netloc has no `@`, which is the parse path a hand-rolled split gets
    wrong -- and the path a copy-pasted URL most often takes."""
    with pytest.raises(ProductionEndpointRefused):
        vet_endpoint(f"amqp://{host}:{port}/")


@pytest.mark.parametrize(
    "url",
    [
        "amqp://bacmr:bacmr_dev@127.0.0.1/",  # implicit 5672
        "amqp://localhost/",
        "amqps://127.0.0.1/",  # implicit 5671
        "postgresql+asyncpg://knowledge_svc:knowledge_dev@127.0.0.1/bacmr",  # implicit 5432
        "postgresql://localhost/bacmr_v2",
    ],
)
def test_an_omitted_port_is_refused_because_the_default_is_production(url):
    """`amqp://host/` *means* 5672. Leaving the port out is the shortest route
    to production, so it cannot be the one spelling that escapes the check."""
    with pytest.raises(ProductionEndpointRefused):
        vet_endpoint(url)


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "127.0.0.1:5533",  # no scheme: urlsplit reads this as scheme "127.0.0.1"
        "knowledge_svc:knowledge_dev@127.0.0.1:5533/bacmr",
        "amqp://bacmr@127.0.0.1:not-a-port/",
        "amqp:///",  # no host
        "amqp://[::1:5673/",  # malformed IPv6 literal
    ],
)
def test_anything_unparseable_is_refused_rather_than_guessed(url):
    """Fail closed: a URL the guard cannot read is a URL it cannot clear."""
    with pytest.raises(ProductionEndpointRefused):
        vet_endpoint(url)


@pytest.mark.parametrize("key", ["host", "hostaddr", "port", "HOST", "Port"])
def test_a_redirecting_query_parameter_is_refused(key):
    """libpq and asyncpg both honour `host`/`port` as connection parameters, so
    `...@127.0.0.1:5533/bacmr?host=/var/run/postgresql` would clear a port check
    and then connect somewhere else entirely."""
    with pytest.raises(ProductionEndpointRefused) as err:
        vet_endpoint(f"postgresql+asyncpg://k:p@127.0.0.1:5533/bacmr?{key}=/var/run/postgresql")
    assert key in str(err.value)


# --------------------------------------------------------------------------
# 3. The isolated stack is accepted, and it is the default
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "amqp://bacmr:bacmr_dev@127.0.0.1:5773/",
        "amqp://bacmr:bacmr_dev@localhost:5773/%2F",
        "postgresql+asyncpg://knowledge_svc:knowledge_dev@127.0.0.1:5533/bacmr",
        "postgresql://knowledge_svc:knowledge_dev@[::1]:5533/bacmr",
        "amqp://bacmr:bacmr_dev@127.0.0.1:5433/",  # the dev profile: not production
    ],
)
def test_the_isolated_stack_is_accepted_unchanged(url):
    assert vet_endpoint(url) == url


def test_the_defaults_are_the_itest_ports_not_production(monkeypatch):
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    monkeypatch.delenv("TEST_AMQP_URL", raising=False)
    assert ":5533/" in resolve_dsn()
    assert ":5773/" in resolve_amqp_url()
    assert resolve_dsn() == _env.DEFAULT_DSN
    assert resolve_amqp_url() == _env.DEFAULT_AMQP_URL
    for denied in DENIED_PORTS:
        assert f":{denied}" not in resolve_dsn()
        assert f":{denied}" not in resolve_amqp_url()


def test_the_env_vars_are_honoured(monkeypatch):
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql+asyncpg://u:p@127.0.0.1:5534/bench")
    monkeypatch.setenv("TEST_AMQP_URL", "amqp://u:p@127.0.0.1:5774/")
    assert resolve_dsn() == "postgresql+asyncpg://u:p@127.0.0.1:5534/bench"
    assert resolve_amqp_url() == "amqp://u:p@127.0.0.1:5774/"


@pytest.mark.parametrize("port", sorted(DENIED_PORTS))
def test_an_env_var_cannot_smuggle_production_in(port, monkeypatch):
    """The env var is the override, and it is vetted like everything else."""
    monkeypatch.setenv("TEST_DATABASE_URL", f"postgresql+asyncpg://u:p@db:{port}/bacmr_v2")
    monkeypatch.setenv("TEST_AMQP_URL", f"amqp://u:p@broker:{port}/")
    with pytest.raises(ProductionEndpointRefused):
        resolve_dsn()
    with pytest.raises(ProductionEndpointRefused):
        resolve_amqp_url()


def test_the_amqp_variable_is_not_the_services_own_setting(monkeypatch):
    """`AMQP_URL` is the runtime setting several unit tests monkeypatch to a
    throwaway value (`services/*/tests/test_settings.py` sets it to :5673). If
    this resolver read that name, a service env would steer the test broker."""
    monkeypatch.setenv("AMQP_URL", "amqp://bacmr:x@localhost:5673/")
    monkeypatch.delenv("TEST_AMQP_URL", raising=False)
    assert resolve_amqp_url() == _env.DEFAULT_AMQP_URL


# --------------------------------------------------------------------------
# 3b. One variable, several roles
# --------------------------------------------------------------------------
# The defect `role_dsn` fixes: `services/tutor/tests` and
# `services/knowledge/tests` both read TEST_DATABASE_URL and each expected its own
# role in it, so no single exported value could make the integration suite pass --
# the knowledge role gave the tutor tests "permission denied for schema tutor" and
# the tutor role broke the knowledge tests instead. The variable now carries the
# server; the test names the role.


def test_role_dsn_keeps_the_server_from_the_variable(monkeypatch):
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql+asyncpg://whoever:x@127.0.0.1:5534/bench")
    assert (
        _env.role_dsn("tutor_svc", "tutor_dev")
        == "postgresql+asyncpg://tutor_svc:tutor_dev@127.0.0.1:5534/bacmr"
    )


def test_role_dsn_takes_the_role_from_the_caller_not_from_the_variable(monkeypatch):
    """The whole point: the same exported value yields either role on request."""
    monkeypatch.setenv(
        "TEST_DATABASE_URL", "postgresql+asyncpg://knowledge_svc:knowledge_dev@127.0.0.1:5533/bacmr"
    )
    tutor = _env.role_dsn("tutor_svc", "tutor_dev")
    knowledge = _env.role_dsn("knowledge_svc", "knowledge_dev")
    assert "tutor_svc" in tutor and "knowledge_svc" not in tutor
    assert "knowledge_svc" in knowledge and "tutor_svc" not in knowledge
    assert "127.0.0.1:5533" in tutor and "127.0.0.1:5533" in knowledge


def test_role_dsn_names_the_database_too(monkeypatch):
    """`test_db_isolation.py` asks for `zitadel` as the tutor role, expecting a
    refusal; the database has to be substitutable for that test to exist."""
    monkeypatch.setenv("TEST_DATABASE_URL", _env.DEFAULT_DSN)
    assert _env.role_dsn("tutor_svc", "tutor_dev", "zitadel").endswith(":5533/zitadel")


@pytest.mark.parametrize("port", sorted(DENIED_PORTS))
def test_role_dsn_is_vetted_like_everything_else(port, monkeypatch):
    """Vetted on the way in AND on the way out, so the substitution cannot be the
    step that produces an endpoint the guard has not seen."""
    monkeypatch.setenv("TEST_DATABASE_URL", f"postgresql+asyncpg://u:p@db:{port}/bacmr_v2")
    with pytest.raises(ProductionEndpointRefused):
        _env.role_dsn("knowledge_svc", "knowledge_dev")


def test_role_dsn_refuses_a_portless_variable(monkeypatch):
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql://knowledge_svc:knowledge_dev@db/bacmr")
    with pytest.raises(ProductionEndpointRefused):
        _env.role_dsn("knowledge_svc", "knowledge_dev")


# --------------------------------------------------------------------------
# 4. No override exists
# --------------------------------------------------------------------------

OVERRIDE_RE = re.compile(
    r"i[_-]?know|i[_-]?am[_-]?sure|force|override|bypass|unsafe|no[_-]?check|skip[_-]?check"
    r"|allow[_-]?prod|yes[_-]?really|danger",
    re.IGNORECASE,
)


def test_no_override_on_the_public_surface():
    """Nothing importable from the guard is named like an escape hatch."""
    offenders = [n for n in dir(_env) if not n.startswith("_") and OVERRIDE_RE.search(n)]
    assert offenders == [], f"_env exposes {offenders}"


def test_the_vetter_takes_only_the_url():
    """No second argument to smuggle an exemption through, and no keyword-only
    flag hiding behind a default."""
    assert list(inspect.signature(vet_endpoint).parameters) == ["url"]
    assert list(inspect.signature(resolve_dsn).parameters) == []
    assert list(inspect.signature(resolve_amqp_url).parameters) == []


def _named_things(tree):
    """Identifiers, CLI flag literals and env-var-shaped literals in a module.

    An AST walk rather than a text search, because the prose above is allowed to
    say "there is no override" and a regex over raw source cannot tell that
    sentence from a `--force` flag three lines below it.
    """
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, ast.arg | ast.keyword) and node.arg:
            names.append(node.arg)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.append(node.name)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and (node.value.startswith("-") or re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", node.value))
        ):
            names.append(node.value)
    return names


def test_no_override_in_the_guard_source():
    """Covers the shapes a `dir()` scan misses: a private helper, a keyword
    argument, an environment variable read once and never exported."""
    tree = ast.parse(Path(_env.__file__).read_text(encoding="utf-8"))
    offenders = [n for n in _named_things(tree) if OVERRIDE_RE.search(n)]
    assert offenders == [], f"_env names an escape hatch: {offenders}"


def test_the_guard_reads_no_environment_variable_but_its_own_two():
    """An extra `os.environ` key would be an undocumented second door."""
    tree = ast.parse(Path(_env.__file__).read_text(encoding="utf-8"))
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", node.value)
    }
    assert literals == {"TEST_DATABASE_URL", "TEST_AMQP_URL"}


# --------------------------------------------------------------------------
# 5. The tests this guard exists for actually use it
# --------------------------------------------------------------------------

# Every integration module in this directory, and whether it also talks to a broker.
GUARDED_MODULES = {
    "test_consumer.py": True,
    "test_poller.py": True,
    "test_skeleton_smoke.py": True,
    "test_outbox_emit.py": False,  # Postgres only, but it TRUNCATEs content.outbox
}


@pytest.mark.parametrize("name, uses_broker", sorted(GUARDED_MODULES.items()))
def test_the_integration_modules_resolve_their_endpoints_through_the_guard(name, uses_broker):
    text = (TESTS_DIR / name).read_text(encoding="utf-8")
    assert "import _env" in text, f"{name} must not resolve its own endpoints"
    # `role_dsn` rather than the bare `DSN`: these four write to the `content`
    # schema, so they name `knowledge_svc` instead of taking whichever role the
    # exported TEST_DATABASE_URL happened to carry (the tutor suite needs the
    # other one out of the same variable). Either way the endpoint comes from the
    # guard -- what must not appear is a DSN assembled here.
    assert "_env.DSN" in text or "_env.role_dsn(" in text
    assert ("_env.AMQP_URL" in text) == uses_broker


def test_every_integration_module_here_is_covered_by_the_test_above():
    """A new integration module with its own hardcoded DSN would otherwise be
    invisible to this file -- the dict above has to be kept honest."""
    found = {
        path.name
        for path in TESTS_DIR.glob("test_*.py")
        if path.name != Path(__file__).name
        and "pytestmark = pytest.mark.integration" in path.read_text(encoding="utf-8")
    }
    assert found == set(GUARDED_MODULES)


# --------------------------------------------------------------------------
# 6. And so do the OTHER suites -- this guard is repo-wide now
# --------------------------------------------------------------------------

REPO = TESTS_DIR.parents[2]

# Every directory whose integration tests open a Postgres connection. They reach
# this module through `pyproject.toml`'s `[tool.pytest.ini_options] pythonpath`;
# `deploy/tests` goes through its own `_endpoints.py`, which re-exports it.
OTHER_SUITES = ("services/knowledge/tests", "services/tutor/tests", "deploy/tests")


def _integration_modules_outside_this_directory() -> list[Path]:
    found = [
        path
        for suite in OTHER_SUITES
        for path in sorted((REPO / suite).glob("test_*.py"))
        if "pytest.mark.integration" in path.read_text(encoding="utf-8")
    ]
    assert found, "the globs above stopped matching anything"
    return found


@pytest.mark.parametrize(
    "path", _integration_modules_outside_this_directory(), ids=lambda p: f"{p.parent.name}/{p.name}"
)
def test_no_other_suite_resolves_the_test_database_url_itself(path):
    """One resolver, repo-wide.

    `services/tutor/tests` and `services/knowledge/tests` each used to read
    TEST_DATABASE_URL directly, with a hardcoded fallback naming its own role --
    which is why the integration suite could not pass under any single exported
    value, and why both defaults still pointed at the dev profile's 5433 long
    after the isolated stack moved to 5533. A module that reads the variable
    again is that defect coming back.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    literals = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value == "TEST_DATABASE_URL"
    ]
    assert literals == [], f"{path.name} resolves the endpoint itself"


@pytest.mark.parametrize(
    "path", _integration_modules_outside_this_directory(), ids=lambda p: f"{p.parent.name}/{p.name}"
)
def test_every_other_suites_dsn_comes_from_role_dsn(path):
    """A Postgres DSN in one of those modules must be the guard's, with the role
    named by the test rather than inherited from whatever the variable carried."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    spells_a_dsn = any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith(("postgresql://", "postgresql+"))
        for node in ast.walk(tree)
    )
    assert not spells_a_dsn, f"{path.name} spells out a DSN instead of asking the guard"
    if "postgres" in text.lower() or "make_engine" in text:
        assert "role_dsn(" in text, f"{path.name} must take its DSN from the guard"


def test_no_module_in_this_directory_hardcodes_a_production_endpoint():
    """The defect this guard was written for, as a standing check.

    String literals only, via the AST: a comment is free to say the old constant
    pointed at 127.0.0.1:5673 -- a comment cannot open a connection. `_env.py`
    and this file legitimately hold the ports (one owns the list, the other
    tests it); everything else here must come through the guard.
    """
    owners = {"_env.py", Path(__file__).name}
    offenders: dict[str, list[str]] = {}
    for path in sorted(TESTS_DIR.glob("*.py")):
        if path.name in owners:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        hits = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and any(f":{port}" in node.value for port in DENIED_PORTS)
        ]
        if hits:
            offenders[path.name] = hits
    assert offenders == {}, f"hardcoded production endpoints: {offenders}"
