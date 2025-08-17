from agents import Agent, function_tool, Runner, OpenAIChatCompletionsModel,set_tracing_disabled,RunContextWrapper,enable_verbose_stdout_logging
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os


# enable_verbose_stdout_logging()
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



@function_tool
def get_weather(city: str) -> str:
    """Returns weather info for the specified city."""
    return f" {city} is Sunny!."

@function_tool
def sum_numbers(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b


math_tutor = Agent(
    name="Math Tutor",
    instructions="You are a math Tutor and answer to only math related queries.",
    model=external_model,
    tools=[sum_numbers],
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
    tools=[get_weather],
    handoffs=[history_tutor,math_tutor]
)

result = Runner.run_sync(triage_agent,"What is 2+2.")
print(result.final_output)
for item in result.new_items:
    print(type(item).__name__)
