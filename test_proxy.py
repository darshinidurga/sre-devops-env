import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"]
)

resp = client.chat.completions.create(
    model=os.environ["MODEL_NAME"],
    messages=[{"role": "user", "content": "say hello in 5 words"}],
    max_tokens=20
)
print("Proxy working:", resp.choices[0].message.content) 
