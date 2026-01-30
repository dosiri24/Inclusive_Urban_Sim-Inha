from dotenv import load_dotenv
load_dotenv()

from llm_api import get_llm

models = ["gemini", "chatgpt", "claude", "exaone"]
for name in models:
    try:
        llm = get_llm(name)
        response = llm.chat([{"role": "user", "content": "Say hi in 5 words"}])
        print(f"{name}: {response}")
    except Exception as e:
        print(f"{name}: {type(e).__name__}")
