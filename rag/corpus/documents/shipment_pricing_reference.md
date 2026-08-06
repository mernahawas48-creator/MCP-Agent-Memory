# Shipment Pricing Reference

## SP-1 — Rates

The base rate is the approved starting price. The final rate is the price after
an authorized adjustment. Current values come from MySQL.

## SP-2 — Rate Exceptions

A rate exception links a requested discount to a shipment and requires the
authority rules in RE-1 or RE-2.

## SP-3 — Pricing Sequence

Retrieve the shipment and pending request, verify the role, apply the correct
policy, and write the result through the defensive MCP tool.

## SP-4 — Credit Holds

Pricing approval does not remove a credit hold.
