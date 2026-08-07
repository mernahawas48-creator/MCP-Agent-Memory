from strategies.sliding_window import SlidingWindow


messages = [
    {"role": "user", "content": "Message 1"},
    {"role": "assistant", "content": "Message 2"},
    {"role": "user", "content": "Message 3"},
    {"role": "assistant", "content": "Message 4"},
    {"role": "user", "content": "Message 5"},
]

strategy = SlidingWindow(max_messages=3)

result = strategy.apply(messages)

print("Original messages:")
print(messages)

print("\nAfter sliding window:")
print(result)
