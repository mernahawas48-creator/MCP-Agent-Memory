from strategies.recursive_summarization import RecursiveSummarization


messages = [
    {"role": "user", "content": "I need help with my account."},
    {"role": "assistant", "content": "Sure, I can help."},
    {"role": "user", "content": "My account has a payment issue."},
    {"role": "assistant", "content": "Let me check the account."},
    {"role": "user", "content": "The payment was made yesterday."},
    {"role": "assistant", "content": "I found the payment."},
    {"role": "user", "content": "Can you confirm it?"},
    {"role": "assistant", "content": "Yes, it was received."},
]

strategy = RecursiveSummarization(keep_last_messages=3)

result = strategy.apply(messages)

print("Original messages:")
print(messages)

print("\nAfter recursive summarization:")
print(result)

print("\nNumber of messages before:", len(messages))
print("Number of messages after:", len(result))
