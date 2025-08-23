
from agents.extensions import handoff_filters
from agents import Agent, function_tool, Runner, OpenAIChatCompletionsModel, set_tracing_disabled, RunContextWrapper, handoff, enable_verbose_stdout_logging
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX



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


async def on_handoff(ctx:RunContextWrapper[None]):
    print(f"Handoff called.")

faq_agent = Agent(name="FAQ agent",instructions="You can add numbers and answer to weather related user query.",tools=[get_weather,sum_numbers],model=external_model)

handoff_obj = handoff(
    tool_description_override="You can add numbers and answer to weather related user query.",
    agent=faq_agent,
    input_filter=handoff_filters.remove_all_tools, 
    on_handoff=on_handoff
)


agent = Agent(
    name="Triage Agent",
    instructions=f"{RECOMMENDED_PROMPT_PREFIX}",
    model=external_model,
    handoffs=[handoff_obj],
)

result = Runner.run_sync(agent, "What is Today's weather in karachi.")

print("Last Agent:", result.last_agent.name)
print("Final Output:", result.final_output)