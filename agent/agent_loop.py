from agent.session_manager import SessionManager
from agent.query_router import QueryRouter
from agent.verifier import Verifier

from context_eval.strategies.sliding_window import SlidingWindow


class AgentLoop:

    def __init__(self):

        # Manage user conversations
        self.session_manager = SessionManager()

        # Keep only useful conversation context
        self.strategy = SlidingWindow(max_messages=10)

        # Decide which logistics area to use
        self.router = QueryRouter()

        # Check if information is trusted
        self.verifier = Verifier()


    def start(self, customer_id=None):
        """
        Create a new customer session.
        """

        session = self.session_manager.create_session(
            customer_id=customer_id
        )

        return session.session_id



    def process(self, session_id, messages):
        """
        Process one customer request.
        """

        # Get current session
        session = self.session_manager.get_session(session_id)


        if session is None:
            raise ValueError("Session not found")


        # -----------------------------
        # 1. Context Management
        # -----------------------------

        context = self.strategy.process(messages)



        # -----------------------------
        # 2. Understand Customer Request
        # -----------------------------

        last_message = messages[-1]["content"]

        destination = self.router.route(last_message)



        # Example routes:
        #
        # shipment
        # customer
        # invoice
        # credit


        session.add_note(
            f"Request routed to: {destination}"
        )



        # -----------------------------
        # 3. Call MCP Server
        # -----------------------------

        evidence = self.call_mcp_tool(
            destination,
            last_message
        )



        # -----------------------------
        # 4. Verify Information
        # -----------------------------

        verified = self.verifier.verify(
            evidence,
            context
        )


        if verified:

            answer = self.generate_response(
                destination,
                evidence
            )

        else:

            answer = (
                "I cannot provide a reliable answer "
                "because no verified information was found."
            )



        # -----------------------------
        # 5. Update Session Memory
        # -----------------------------

        session.add_note(
            {
                "query": last_message,
                "category": destination,
                "verified": verified
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
