from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

AZURE_API_KEY = os.getenv("AZURE_API_KEY")


endpoint = "https://aim-azure-ai-foundry.cognitiveservices.azure.com/openai/v1/"
deployment_name = "text-embedding-model"
api_key = AZURE_API_KEY

client = OpenAI(
    base_url = endpoint,
    api_key = api_key,
)

response = client.embeddings.create(
    input = "How do I use Python in VS Code?",
    model = deployment_name
)
print(response.data[0].embedding)