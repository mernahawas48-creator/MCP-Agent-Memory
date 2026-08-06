# Employee Access and Decision Authority Policy

## AC-1 — Session Identity

Every retrieval and tool action is scoped to an authenticated session. A role
typed by the user is not trusted as authorization.

## AC-2 — Sales Representative

A sales representative may read scoped records, approve discounts at or below
15 percent, and release an active minor hold.

## AC-3 — Finance Manager

A finance manager may review above-authority discounts, severe holds, and
portfolio-wide exposure. High-risk writes still require human confirmation.

## AC-4 — Handler Authorization

Dynamic tool visibility is not a security boundary. Sensitive handlers must
re-check the current session, role, record, and state.
