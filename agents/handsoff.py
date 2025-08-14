from agents import Agent, function_tool, Runner, OpenAIChatCompletionsModel,set_tracing_disabled,RunContextWrapper
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os



set_tracing_disabled(True)
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")


external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

external_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

math_tutor = Agent(
    name="Math Tutor",
    instructions="You are a math Tutor and answer to only math related queries.",
    model=external_model,
    handoff_description="You are a math Tutor and answer to only math related queries." 
)

history_tutor = Agent(
    name="History Tutor",
    instructions="You are a history Tutor and answer to only history related queries.",
    model=external_model,
    handoff_description="You are a history Tutor and answer to only history related queries." 
)  

triage_agent = Agent(
    name="Triage Agent",
    instructions="You are triage agent and your task is to handoff user queries according provided agent description." \
    "if user query is not related to handsoff agents than answer to user through triage agent.",
    model=external_model,
    handoffs=[history_tutor,math_tutor]
)

result = Runner.run_sync(triage_agent,"Define cell in biology.")
print(result.last_agent.name)
print(result.final_output)