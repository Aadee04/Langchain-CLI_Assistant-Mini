import os
import sys
from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from tools import get_time, open_calculator, search_wikipedia
from tools.tool_builder import run_python
import importlib
import pkgutil

# NLP imports for stemming/lemmatization
try:
    import nltk
    from nltk.stem import PorterStemmer, WordNetLemmatizer
except ImportError:
    nltk = None
    PorterStemmer = None
    WordNetLemmatizer = None

# Download NLTK data if needed
if nltk is not None:
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

# Initialize NLP tools
stemmer = PorterStemmer() if PorterStemmer else None
lemmatizer = WordNetLemmatizer() if WordNetLemmatizer else None

from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

def permission_check(tool_name):
    # Example: Only allow certain tools, or prompt user for permission
    allowed_tools = {"get_time", "open_calculator", "search_wikipedia", "run_python"}
    return tool_name in allowed_tools

def load_tools():
    tool_funcs = {}
    for finder, name, ispkg in pkgutil.iter_modules([os.path.join(os.path.dirname(__file__), 'tools')]):
        if name.startswith("__") or name == "tool_builder":
            continue
        module = importlib.import_module(f"tools.{name}")
        for attr in dir(module):
            if not attr.startswith("_") and callable(getattr(module, attr)):
                tool_funcs[attr] = getattr(module, attr)
    # Add run_python from tool_builder
    tool_funcs["run_python"] = run_python
    return tool_funcs

tool_funcs = load_tools()

langchain_tools = [
    Tool(
        name=tool_name,
        func=(lambda *args, tool=tool_name: tool_funcs[tool](*args) if permission_check(tool) else "Permission denied."),
        description=f"Dynamically loaded tool: {tool_name}"
    ) for tool_name in tool_funcs
]

# Initialize Ollama LLM (Phi-3 Mini)
llm = Ollama(model="phi3")

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
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )
    agent_supported = True
except Exception as e:
    agent = None
    agent_supported = False

def print_help():
    print("""
Available commands:
  time                - Show current system time
  calc                - Open calculator (Windows)
  wiki <query>        - Search Wikipedia for <query>
  help                - Show this help message
  exit, quit          - Exit the assistant
Any other input will be sent to the LLM assistant or agent.
""")

def preprocess_command(user_input):
    """Return stemmed and lemmatized first word and args."""
    if not user_input:
        return '', []
    words = user_input.strip().split()
    cmd = words[0].lower()
    args = words[1:]
    # Apply stemming and lemmatization
    if stemmer:
        cmd_stem = stemmer.stem(cmd)
    else:
        cmd_stem = cmd
    if lemmatizer:
        cmd_lemma = lemmatizer.lemmatize(cmd)
    else:
        cmd_lemma = cmd
    return cmd, cmd_stem, cmd_lemma, args

def detect_intent(user_input):
    """Detect intent for tool use from natural language input."""
    text = user_input.lower()
    # Time intent
    if any(kw in text for kw in ["what time", "current time", "date and time", "show time", "tell time", "now"]):
        return "time", []
    # Calculator intent
    if any(kw in text for kw in ["open calculator", "launch calculator", "start calculator", "calculator app", "calc app", "open calc", "launch calc"]):
        return "calc", []
    # Wikipedia intent
    if any(kw in text for kw in ["search wikipedia", "look up", "wikipedia", "wiki", "summarize from wikipedia", "find on wikipedia"]):
        # Try to extract query
        import re
        match = re.search(r'(?:search|look up|find|summarize from)? ?(?:wikipedia|wiki)? ?([\w\s]+)', text)
        if match:
            query = match.group(1).strip()
            if query:
                return "wiki", [query]
        # Fallback: ask for query
        return "wiki", []
    return None, []

def main():
    print("LangChain CLI Assistant (Phi-3 Mini via Ollama)")
    print("Type 'help' for commands. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        # 1. Intent detection (natural language)
        intent, intent_args = detect_intent(user_input)
        if intent == "time":
            print(f"[Time] {get_time()}\n")
            continue
        elif intent == "calc":
            print(f"[Calc] {open_calculator()}\n")
            continue
        elif intent == "wiki":
            if not intent_args:
                print("Usage: wiki <query>\n")
            else:
                print(f"[Wiki] {search_wikipedia(' '.join(intent_args))}\n")
            continue
        # 2. Command parsing (stem/lemma)
        cmd, cmd_stem, cmd_lemma, args = preprocess_command(user_input)
        time_cmds = {"time", "tim", "times"}
        calc_cmds = {"calc", "calcul", "calculate", "calculator"}
        wiki_cmds = {"wiki", "wikipedia", "search"}
        help_cmds = {"help", "assist", "h"}
        exit_cmds = {"exit", "quit", "close", "bye"}
        if cmd in exit_cmds or cmd_stem in exit_cmds or cmd_lemma in exit_cmds:
            print("Goodbye!")
            break
        elif cmd in help_cmds or cmd_stem in help_cmds or cmd_lemma in help_cmds:
            print_help()
            continue
        elif cmd in time_cmds or cmd_stem in time_cmds or cmd_lemma in time_cmds:
            print(f"[Time] {get_time()}\n")
            continue
        elif cmd in calc_cmds or cmd_stem in calc_cmds or cmd_lemma in calc_cmds:
            print(f"[Calc] {open_calculator()}\n")
            continue
        elif cmd in wiki_cmds or cmd_stem in wiki_cmds or cmd_lemma in wiki_cmds:
            if not args:
                print("Usage: wiki <query>\n")
            else:
                print(f"[Wiki] {search_wikipedia(' '.join(args))}\n")
            continue
        # 3. LangChain agent tool-calling (if supported)
        if agent_supported:
            try:
                response = agent.run(user_input)
                print(f"Assistant: {response}\n")
                continue
            except Exception as e:
                print(f"[Agent error: {e}] Fallback to LLM.\n")
        # 4. Fallback: LLM conversation
        response = conversation.predict(input=user_input)
        print(f"Assistant: {response}\n")

if __name__ == "__main__":
    main()
