from __future__ import annotations

from app_instance import app
from db import SwiftrailDatabaseError, db_cursor
from schemas import CustomerInvoicesInput, SearchCustomerInput, ShipmentStatusInput
from tool_support import (
    authorize_session,
    database_failure,
    fail,
    ok,
    unexpected_failure,
    validate_request,
)


@app.tool()
def search_customer(request: SearchCustomerInput) -> dict:
    """Retrieve one customer profile through a scoped, validated read query."""

    validated, error = validate_request(SearchCustomerInput, request)
    if error:
        return error

    try:
        with db_cursor() as (_, cursor):
            _, auth_error = authorize_session(
                cursor,
                session_id=validated.session_id,
                allowed_roles={"sales_rep", "finance_manager"},
            )
            if auth_error:
                return auth_error

            cursor.execute(
                """
                SELECT id, name, credit_limit, balance_due, credit_status
                FROM customers
                WHERE id = %s
                """,
                (validated.customer_id,),
            )
            customer = cursor.fetchone()

        if customer is None:
            return fail(
                "CUSTOMER_NOT_FOUND",
                f"Customer #{validated.customer_id} was not found.",
            )
        return ok(
            "CUSTOMER_RETRIEVED",
            "Customer profile retrieved successfully.",
            {"customer": customer},
        )
    except SwiftrailDatabaseError as db_error:
        return database_failure(db_error)
    except Exception:
        return unexpected_failure("customer lookup")


@app.tool()
def get_shipment_status(request: ShipmentStatusInput) -> dict:
    """Retrieve the current status and financial details of one shipment."""

    validated, error = validate_request(ShipmentStatusInput, request)
    if error:
        return error

    try:
        with db_cursor() as (_, cursor):
            _, auth_error = authorize_session(
                cursor,
                session_id=validated.session_id,
                allowed_roles={"sales_rep", "finance_manager"},
            )
            if auth_error:
                return auth_error

            cursor.execute(
                """
                SELECT
                    s.id,
                    s.customer_id,
                    c.name AS customer_name,
                    s.origin,
                    s.destination,
                    s.railcar_id,
                    s.base_rate,
                    s.final_rate,
                    s.status,
                    s.requested_by,
                    s.created_at
                FROM shipments AS s
                JOIN customers AS c ON c.id = s.customer_id
                WHERE s.id = %s
                """,
                (validated.shipment_id,),
            )
            shipment = cursor.fetchone()

        if shipment is None:
            return fail(
                "SHIPMENT_NOT_FOUND",
                f"Shipment #{validated.shipment_id} was not found.",
            )
        return ok(
            "SHIPMENT_RETRIEVED",
            "Shipment status retrieved successfully.",
            {"shipment": shipment},
        )
    except SwiftrailDatabaseError as db_error:
        return database_failure(db_error)
    except Exception:
        return unexpected_failure("shipment lookup")


@app.tool()
def list_customer_invoices(request: CustomerInvoicesInput) -> dict:
    """List invoices for one validated customer and authenticated session."""

    validated, error = validate_request(CustomerInvoicesInput, request)
    if error:
        return error

    try:
        with db_cursor() as (_, cursor):
            _, auth_error = authorize_session(
                cursor,
                session_id=validated.session_id,
                allowed_roles={"sales_rep", "finance_manager"},
            )
            if auth_error:
                return auth_error

            cursor.execute(
                "SELECT id FROM customers WHERE id = %s",
                (validated.customer_id,),
            )
            if cursor.fetchone() is None:
                return fail(
                    "CUSTOMER_NOT_FOUND",
                    f"Customer #{validated.customer_id} was not found.",
                )

            cursor.execute(
                """
                SELECT id, customer_id, shipment_id, amount, due_date,
                       paid_status, days_overdue
                FROM invoices
                WHERE customer_id = %s
                ORDER BY due_date DESC, id DESC
                """,
                (validated.customer_id,),
            )
            invoices = cursor.fetchall()

        return ok(
            "INVOICES_RETRIEVED",
            f"Retrieved {len(invoices)} invoice(s).",
            {"customer_id": validated.customer_id, "invoices": invoices},
        )
    except SwiftrailDatabaseError as db_error:
        return database_failure(db_error)
    except Exception:
        return unexpected_failure("invoice lookup")
