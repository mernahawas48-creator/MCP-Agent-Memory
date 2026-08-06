from __future__ import annotations

from dataclasses import dataclass
from threading import RLock


@dataclass(frozen=True, slots=True)
class SessionIdentity:
    session_id: str
    employee_id: int
    role: str
    employee_name: str


class SessionRegistry:
    """Process-local authenticated session registry.

    Each authenticated client uses a separate session ID. This prevents one
    client's role or employee identity from overwriting another client's
    session and provides a suitable scope for the future memory system.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionIdentity] = {}
        self._lock = RLock()

    def set(self, identity: SessionIdentity) -> SessionIdentity | None:
        with self._lock:
            previous = self._sessions.get(identity.session_id)
            self._sessions[identity.session_id] = identity
            return previous

    def get(self, session_id: str) -> SessionIdentity | None:
        with self._lock:
            return self._sessions.get(session_id)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def has_role(self, role: str) -> bool:
        with self._lock:
            return any(
                identity.role == role
                for identity in self._sessions.values()
            )


registry = SessionRegistry()


def set_session(
    *,
    session_id: str,
    employee_id: int,
    role: str,
    employee_name: str,
) -> SessionIdentity | None:
    return registry.set(
        SessionIdentity(
            session_id=session_id,
            employee_id=employee_id,
            role=role,
            employee_name=employee_name,
        )
    )


def get_session(session_id: str) -> SessionIdentity | None:
    return registry.get(session_id)


def clear_session(session_id: str) -> None:
    registry.clear(session_id)


def has_role(role: str) -> bool:
    return registry.has_role(role)