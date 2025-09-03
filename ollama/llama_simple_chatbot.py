from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = ChatOllama(model="llama3.2:3b")

messages = [
    SystemMessage("You are a helpful assistant."),
]

while True:
    user_input = input("You\t: ").strip()

    if user_input in ["exit", "quit", "q"]:
        print("Goodbye!")
        break

    messages.append(HumanMessage(user_input))

    response = llm.invoke(messages)
    print("Bot\t :", response.content)

    messages.append(AIMessage(response.content))

    