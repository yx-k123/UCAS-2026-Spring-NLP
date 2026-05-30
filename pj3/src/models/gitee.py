from openai import OpenAI
with open('configs/api.txt', 'r') as f:
	api_key = f.read().strip()

client = OpenAI(
	base_url="https://ai.gitee.com/v1",
	api_key=api_key,
	default_headers={"X-Failover-Enabled":"true"},
)

response = client.chat.completions.create(
	messages=[
		{
			"role": "system",
			"content": "You are a helpful and harmless assistant. You should think step-by-step."
		},
		{
			"role": "user",
			"content": "Can you please let us know more details about yourself."
		}
	],
	model="Qwen3-8B",
	stream=True,
	max_tokens=1024,
	temperature=0,
	top_p=1,
	extra_body={
		"top_k": 50,
	},
	frequency_penalty=1,
)

fullResponse = ""
print("Response:")
# Print streaming response
for chunk in response:
	if len(chunk.choices) == 0:
		continue
	delta = chunk.choices[0].delta
	# If is thinking content, print it in gray
	if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
		fullResponse += delta.reasoning_content
		print(f"\033[90m{delta.reasoning_content}\033[0m", end="", flush=True)
	elif delta.content:
		fullResponse += delta.content
		print(delta.content, end="", flush=True)