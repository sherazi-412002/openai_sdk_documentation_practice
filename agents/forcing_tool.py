from agents import Agent, function_tool, Runner, OpenAIChatCompletionsModel,set_tracing_disabled,ModelSettings
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

@function_tool
def math():
    return "The weather is sunny!"


agent = Agent(
    name="Math Tutor",
    # instructions="You are a math Tutor and answer to only math related queries.",
    model=external_model,
    tools=[math],
    # handoff_description="You are a math Tutor and answer to only math related queries.",
    model_settings= ModelSettings(tool_choice="auto")
)


result = Runner.run_sync(agent,"What is answer of 2 + 2.")
print(result.last_agent)
print(result.final_output)