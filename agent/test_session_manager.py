from agent.session_manager import SessionManager



# Create session manager
manager = SessionManager()



# Create Swiftrail customer session
session = manager.create_session(
    customer_id="CUST001",
    customer_name="ABC Logistics"
)



print("Session ID:")
print(session.session_id)



print("\nCustomer:")
print(session.customer_id)
print(session.customer_name)



# Add some conversation notes
session.add_note(
    "Customer asked about shipment MSKU100001"
)


session.add_note(
    "Request routed to shipment_tool"
)



print("\nScratchpad:")
print(session.scratchpad)



# Show active sessions
print("\nCurrent Sessions:")
print(manager.list_sessions())



# Get session again
loaded_session = manager.get_session(
    session.session_id
)


print("\nLoaded Session:")
print(loaded_session.customer_name)



# Close session
manager.end_session(
    session.session_id
)



print("\nAfter Closing:")
print(manager.list_sessions())

