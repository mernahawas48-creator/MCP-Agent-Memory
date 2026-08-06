# Overdue Invoice Collection Procedure

## IC-1 — Current Delinquency

Invoice age, paid status, and outstanding amount must be retrieved from MySQL.
An invoice from 30 to 89 days overdue may support a minor-hold workflow. An
invoice at least 90 days overdue is a severe-risk signal.

## IC-2 — Follow-Up

The employee records the invoice ID, customer ID, amount, due date, and agreed
next action without promising a decision beyond the employee's authority.

## IC-3 — Escalation

Escalate severe-risk accounts, disputed balances, conflicting records, and
discount requests above delegated authority to finance.
