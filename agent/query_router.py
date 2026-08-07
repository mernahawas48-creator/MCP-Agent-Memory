class QueryRouter:
    """
    Decide where the Swiftrail agent should get information from.
    """


    def route(self, query: str):

        query = query.lower()



        # -------------------------
        # Company knowledge
        # -------------------------

        if (
            "policy" in query
            or "manual" in query
            or "procedure" in query
            or "rule" in query
        ):
            return "rag"



        # -------------------------
        # Previous conversations
        # -------------------------

        if (
            "remember" in query
            or "previous" in query
            or "last time" in query
            or "history" in query
        ):
            return "memory"



        # -------------------------
        # Shipment operations
        # -------------------------

        if (
            "shipment" in query
            or "container" in query
            or "tracking" in query
            or "delivery" in query
            or "package" in query
        ):
            return "shipment_tool"



        # -------------------------
        # Invoice / Payment
        # -------------------------

        if (
            "invoice" in query
            or "bill" in query
            or "payment" in query
            or "amount due" in query
        ):
            return "invoice_tool"



        # -------------------------
        # Credit management
        # -------------------------

        if (
            "credit" in query
            or "limit" in query
            or "hold" in query
            or "blocked" in query
        ):
            return "credit_tool"



        # -------------------------
        # Customer information
        # -------------------------

        if (
            "customer" in query
            or "client" in query
            or "account" in query
        ):
            return "customer_tool"



        # -------------------------
        # Default
        # -------------------------

        return "context"
