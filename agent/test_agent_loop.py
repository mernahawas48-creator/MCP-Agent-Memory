from agent.agent_loop import AgentLoop



# Create Swiftrail Agent
agent = AgentLoop()



# Start customer session
session = agent.start(
    customer_id="CUST001"
)



# Customer conversation
messages = [

    {
        "role": "user",
        "content": "Hello"
    },


    {
        "role": "assistant",
        "content": "Hello, welcome to Swiftrail Logistics."
    },


    {
        "role": "user",
        "content": "Where is container MSKU100001?"
    }

]



# Process request
result = agent.process(
    session,
    messages
)



# Display agent result
print("\n--- Swiftrail Agent Result ---")

print("Session:")
print(result["session_id"])


print("\nCategory:")
print(result["category"])


print("\nVerified:")
print(result["verified"])


print("\nAnswer:")
print(result["answer"])


print("\nEvidence:")
print(result["evidence"])



# Close session
agent.end(session)
