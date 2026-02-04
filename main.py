from dotenv import load_dotenv
load_dotenv()

from agent_api import Memory, Agent

memory = Memory(
    system_context="Respond in Korean.",
    debate_rule="This is a redevelopment debate.",
    local_context="Area ratio 93%.",
    persona="62 year old male, homeowner."
)

agent = Agent("resident_01", "gemini", memory)

memory.add_utterance("resident_02", "I oppose redevelopment.")

response = agent.respond("What do you think about resident_02's opinion?")
print(response)
