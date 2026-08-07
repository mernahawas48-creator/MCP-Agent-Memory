from strategies.zone_based_pruning import ZoneBasedPruning


messages = [
    {"role": "system", "content": "You are a helpful agent."},
    {"role": "user", "content": "Old question 1"},
    {"role": "assistant", "content": "Old answer 1"},
    {"role": "tool", "content": "Old tool result"},
    {"role": "user", "content": "Recent question"},
    {"role": "assistant", "content": "Recent answer"},
]

strategy = ZoneBasedPruning(keep_recent_messages=2)

result = strategy.apply(messages)

print("Original messages:")
print(messages)

print("\nAfter zone-based pruning:")
print(result)

print("\nNumber of messages before:", len(messages))
print("Number of messages after:", len(result))
