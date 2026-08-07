from strategies.tool_output_masking import ToolOutputMasking


messages = [
    {"role": "user", "content": "Hello"},
    {"role": "tool", "content": "Huge database result 1"},
    {"role": "user", "content": "Check something else"},
    {"role": "tool", "content": "Huge database result 2"},
    {"role": "assistant", "content": "Okay"},
    {"role": "tool", "content": "Latest database result"},
]

strategy = ToolOutputMasking(keep_last_tool_outputs=1)

result = strategy.apply(messages)

print("Original messages:")
print(messages)

print("\nAfter tool-output masking:")
print(result)
