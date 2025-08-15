from agents import Agent, enable_verbose_stdout_logging, Runner, OpenAIChatCompletionsModel,set_tracing_disabled,RunContextWrapper
from openai import AsyncOpenAI
from agentsdk_gemini_adapter import config
import os
from dotenv import load_dotenv
import asyncio

enable_verbose_stdout_logging()
set_tracing_disabled(False)
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



agent = Agent(
    name=1,
    instructions="You are a helpful assistant.",
    model=external_model,
)


async def main():
    result = await Runner.run(agent,"what is  2+2?")
    print(result.final_output)


asyncio.run(main())