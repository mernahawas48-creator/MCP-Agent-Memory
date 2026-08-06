# Credit Hold Classification and Release Policy

## CH-1 — Severity

A minor hold applies when an invoice is 30 to 89 days overdue and no severe
criterion is present.

A severe hold applies when an invoice is at least 90 days overdue or when the
overdue balance exceeds 25 percent of the approved credit limit.

## CH-2 — Minor Release

An authenticated sales representative or finance manager may release an active
minor hold after the server validates the record and current state.

## CH-3 — Severe Release

Only an authenticated finance manager may release an active severe hold.
Explicit human confirmation and an authorization note are required before the
write is committed.

## CH-4 — Customer Status

A customer's credit status may return to good only when no active holds remain
for that customer.
