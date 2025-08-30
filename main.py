import os
import sys
from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
import importlib
import pkgutil
import os, importlib, inspect
from langchain.tools import BaseTool
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType


def discover_tools():
    tools = []
    for file in os.listdir("tools"):
        if file.endswith(".py") and file not in ["__init__.py"]:
            name = file[:-3]
            module = importlib.import_module(f"tools.{name}")
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                # check if it's a LangChain tool
                if isinstance(attr, BaseTool):
                    tools.append(attr)
                # check if it's a function decorated with @tool
                elif callable(attr) and hasattr(attr, "name") and hasattr(attr, "description"):
                    tools.append(attr)
    return tools

langchain_tools = discover_tools()
tool_list_str = "\n".join([f"- {t.name}: {t.description}" for t in langchain_tools])


# Print available tools for debugging and LLM context
print(f"\nAvailable tools: {tool_list_str} \n\n")

# System prompt for the agent/LLM
SYSTEM_PROMPT = """
You are a Desktop Assistant. You have access to the following tools:

{tool_list}

RULES:
1. You MUST always use a tool to answer, unless it is absolutely impossible. 
2. Your output must ALWAYS be valid JSON in this format:
{
  "action": "<tool_name>",
  "action_input": "<string or object input>"
}

Never add explanations, text, or code fences. Output JSON only.

3. Do NOT just answer directly. If you cannot complete the task with the current tools, use the run_python tool to create a new one.
4. Only use tools from the provided list.

EXAMPLES:

User: What time is it?
{
  "action": "get_time",
  "action_input": ""
}

User: Open the calculator
{
  "action": "open_calculator",
  "action_input": ""
}

User: Search Wikipedia for 'LangChain'
{
  "action": "search_wikipedia",
  "action_input": "LangChain"
}

User: I need to download a file
{
  "action": "run_python",
  "action_input": "
import requests
url = "https://example.com/file.txt"
r = requests.get(url)
with open("file.txt", "wb") as f:
    f.write(r.content)
print("File downloaded successfully.")"
}


Begin now. Remember: Always respond with Action + Action Input. 
"action_input" must always be a string (use "" if no input).
You MUST always use a tool to answer, unless it is absolutely impossible. 
Do not add any coding comments. 
Any output to the user must be in a normal human tone.
"""

# Initialize Ollama LLM (Phi-3 Mini)
llm = Ollama(model="phi3", system=SYSTEM_PROMPT)

# Conversation memory
memory = ConversationBufferMemory()

# Conversation chain (fallback)
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Try to initialize agent (if supported by LLM)
try:
    agent = initialize_agent(
        langchain_tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

    agent_supported = True

except Exception as e:
    print(f"Agent init failed: {e}")
    agent = None
    agent_supported = False


def main():
    print("LangChain CLI Assistant (Phi-3 Mini via Ollama)")
    print("Type 'help' for commands. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        cmd = user_input.lower().strip()
        if cmd in {"exit", "quit", "close", "bye"}:
            print("Goodbye!")
            break
        # Try agent first
        if agent_supported:
            try:
                result = agent.invoke(
                    {"input": user_input},
                    return_intermediate_steps=True
                )
                if "intermediate_steps" in result:
                    print("--- LLM Reasoning (Agent Steps) ---")
                    for action, observation in result["intermediate_steps"]:
                        print(f"Action: {action.tool}")
                        print(f"Action Input: {action.tool_input}")
                        print(f"Observation: {observation}\n")
                    print("--- End Reasoning ---\n")
                print(f"Assistant: {result['output']}\n")
                continue
            except Exception as e:
                print(f"[Agent error: {e}] Fallback to LLM.\n")
        # Fallback: LLM conversation
        response = conversation.predict(input=user_input)
        print(f"Assistant: {response}\n")
        print("Agent not supported with this LLM. Only basic conversation available.\n")

if __name__ == "__main__":
    main()
