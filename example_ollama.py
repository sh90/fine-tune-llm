import ollama

model = "gemma3:1b"

response = ollama.generate(model=model,prompt="Generate 2 line poem")
print(response.response)
