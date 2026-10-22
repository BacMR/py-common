"""Broker connection naming.

RabbitMQ shows a connection's `connection_name` client property in the
management UI/API; the gateway's /admin/system/flow uses it to attribute
publishers and consumers to a service (BacMR/docs#83). Every service names
each connection it opens, e.g. `knowledge-worker`, `tutor-outbox`.
"""

from typing import Any


def client_properties(connection_name: str | None) -> dict[str, Any]:
    """Keyword arguments for `aio_pika.connect_robust` naming the connection.

    Empty when no name is given, so the library default stays untouched.
    """
    if not connection_name:
        return {}
    return {"client_properties": {"connection_name": connection_name}}
