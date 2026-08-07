from dataclasses import dataclass, field


@dataclass
class AgentState:

    # Session identifier
    session_id: str


    # Current customer request
    current_goal: str = ""


    # Current processing step
    current_step: str = ""


    # Where the request was routed
    # shipment / invoice / customer / credit
    request_type: str = ""


    # Customer information
    customer_id: str = ""
    customer_name: str = ""


    # Temporary notes during conversation
    scratchpad: list = field(default_factory=list)


    # Context strategy being used
    selected_strategy: str = "sliding_window"


    # Information retrieved from MCP tools
    evidence: list = field(default_factory=list)


    # Verification results
    verification_status: bool = False



    # -------------------------
    # Update current task
    # -------------------------

    def update_goal(self, goal):

        self.current_goal = goal



    def update_step(self, step):

        self.current_step = step



    # -------------------------
    # Routing
    # -------------------------

    def set_request_type(self, request_type):

        self.request_type = request_type



    # -------------------------
    # Customer Data
    # -------------------------

    def set_customer(self, customer_id, customer_name=""):

        self.customer_id = customer_id
        self.customer_name = customer_name



    # -------------------------
    # Scratchpad
    # -------------------------

    def add_note(self, note):

        self.scratchpad.append(note)



    def clear_scratchpad(self):

        self.scratchpad.clear()



    # -------------------------
    # MCP Evidence
    # -------------------------

    def add_evidence(self, data):

        self.evidence.append(data)



    def clear_evidence(self):

        self.evidence.clear()



    # -------------------------
    # Verification
    # -------------------------

    def set_verification(self, status):

        self.verification_status = status
