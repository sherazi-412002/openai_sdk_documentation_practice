from agents import Agent, function_tool, Runner, OpenAIChatCompletionsModel,set_tracing_disabled,ModelSettings,enable_verbose_stdout_logging
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from agents.agent import StopAtTools


enable_verbose_stdout_logging()
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

@function_tool
def subtract_numbers(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b

agent = Agent(
    name="Stop At Stock Agent",
    instructions="Get weather or sum numbers."
    "if query is not related to tools then answer to user query anyway.",
    model=external_model,
    tools=[get_weather, sum_numbers,subtract_numbers],
    model_settings=ModelSettings(
        parallel_tool_calls=True
    )
    # tool_use_behavior="stop_on_first_tool",
    # tool_use_behavior=StopAtTools(stop_at_tool_names=["get_weather",])
)


result = Runner.run_sync(agent,"what is weather in Lahore.add 2 and 2.Subtract 4 from 8.")
print(result.last_agent.name)
print(result.final_output)