if destination == "rag":
            rag_answer = self.rag.answer(last_message, role="sales_rep")
            evidence = {"source": "rag", "data": rag_answer.answer}
            verified = (
                rag_answer.verification.passed
                if rag_answer.verification is not None
                else False
            )
            answer = (
                rag_answer.answer
                if verified
                else "I cannot provide a reliable answer because no "
                     "verified information was found."
            )

        elif destination == "memory":
            customer_id = int(session.customer_id) if session.customer_id else None
            recalled = (
                self.episodic.get_by_customer(customer_id)
                if customer_id is not None
                else []
            )
            verification = self.memory_verifier.verify(last_message, recalled)
            evidence = {
                "source": "episodic_memory",
                "data": recalled,
                "verification": verification.reason,
            }
            verified = verification.passed
            answer = (
                f"Past record: {recalled}"
                if verified
                else "I cannot provide a reliable answer because no "
                     "verified information was found."
            )

        else:
            tool_category = TOOL_DESTINATIONS.get(destination, destination)
            evidence = self.call_mcp_tool(tool_category, last_message)
            verified = evidence.get("data") is not None
            answer = (
                self.generate_response(tool_category, evidence)
                if verified
                else "I cannot provide a reliable answer because no "
                     "verified information was found."
            )

        session.set_verification(verified)

        # -----------------------------
        # 5. Update session scratchpad
        # -----------------------------

        session.add_note(
            {
                "query": last_message,
                "category": destination,
                "verified": verified,
            }
        )

        return {

            "session_id": session_id,

            "category": destination,

            "verified": verified,

            "answer": answer,

            "context": context,

            "evidence": evidence,

            "scratchpad": session.scratchpad

        }

    def call_mcp_tool(self, destination, query):
        """
        Connect with Swiftrail MCP Server.
        """

        if destination == "shipment":

            return {
                "source": "shipment_database",
                "data": "Shipment information retrieved"
            }

        elif destination == "invoice":

            return {
                "source": "invoice_database",
                "data": "Invoice information retrieved"
            }

        elif destination == "customer":

            return {
                "source": "customer_database",
                "data": "Customer information retrieved"
            }

        elif destination == "credit":

            return {
                "source": "finance_database",
                "data": "Credit status retrieved"
            }

        return {
            "source": "unknown",
            "data": None
        }

    def generate_response(self, category, evidence):
        """
        Generate final answer for customer.
        """

        return (
            f"Swiftrail {category} request completed. "
            f"Verified data: {evidence['data']}"
        )

    def end(self, session_id):
        """
        Close customer session.
        """

        self.session_manager.end_session(session_id)
