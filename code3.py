from langchain_google_genai import ChatGoogleGenAI
import os
from dotenv import load_dotenv


load_dotenv()
llm = ChatGoogleGenAI(model="gemini-1.5-pro",temperature=0.7,google_api_key=os.getenv("GOOGLE_API_KEY"))

#chatbot
while True:
    input_text = input("You: ")
    if input_text.lower() in ["exit","quit"]:
        print("Exiting the chat.")
        break
    result=llm.invoke(input_text)
    print(result.content)