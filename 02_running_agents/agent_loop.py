from agents import Agent, enable_verbose_stdout_logging,function_tool, Runner,ModelSettings, OpenAIChatCompletionsModel,set_tracing_disabled,RunContextWrapper
from openai import AsyncOpenAI
from agentsdk_gemini_adapter import config
import os
from dotenv import load_dotenv
import asyncio

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
def weather(city):
    return f"The weather in {city} is sunny!"

agent = Agent(
    name=1,
    instructions="You are a helpful assistant.",
    model=external_model,
    tools=[weather],
    # model_settings=ModelSettings(tool_choice="required"),   ####
    # tool_use_behavior="stop_on_first_tool"                  ####
)


async def main():
    result = await Runner.run(agent,"what is weather in Karachi?",max_turns=1)    ####
    print(result.final_output)


asyncio.run(main())