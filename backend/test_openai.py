import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

def test_openai_embedding():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    try:
        client = OpenAI(api_key=openai_api_key)
        print("OpenAI client initialized.")

        test_text = "This is a test sentence for embedding."
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=test_text
        )
        embedding = response.data[0].embedding
        print(f"Successfully generated embedding. First 5 dimensions: {embedding[:5]}")
        print("OpenAI connection successful.")
    except Exception as e:
        print(f"Error connecting to OpenAI or generating embedding: {e}")

if __name__ == "__main__":
    test_openai_embedding()