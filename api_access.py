from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")

api_key = API_KEY # Replace with your API key
base_url = "https://chat-ai.academiccloud.de/v1"
model = "meta-llama-3.1-8b-instruct" # Choose any available model


client = OpenAI(api_key = api_key, base_url = base_url)

chat_completion = client.chat.completions.create(messages=[{"role":"system","content":"you are a helpful assistant"},
                                                           {"role":"user","content":"hello, how high is the eiffel tower?"}],
                                                           model=model)

print(chat_completion) 