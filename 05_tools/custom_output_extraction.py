import asyncio
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from agents import Agent, OpenAIChatCompletionsModel, RunResult, ToolCallOutputItem, set_tracing_disabled, Runner

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


async def extract_json_payload(run_result: RunResult) -> str:
    # Scan the agent’s outputs in reverse order until we find a JSON-like message from a tool call.
    for item in reversed(run_result.new_items):
        if isinstance(item, ToolCallOutputItem) and item.output.strip().startswith("{"):
            return item.output.strip()
    # Fallback to an empty JSON object if nothing was found
    return "{}"


data_agent = Agent(
    name="Data agent",
    model=external_model,
    instructions="You translate the user's message to Spanish",
)

json_tool = data_agent.as_tool(
    tool_name="get_data_json",
    tool_description="Run the data agent and return only its JSON payload",
    custom_output_extractor=extract_json_payload,
)



orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "Run the data agent and return only its JSON payload"
    ),
    model=external_model,
    tools=[json_tool]
)

async def main():
    result = await Runner.run(
        orchestrator_agent,
        f"give json payload"
    )
    print(result.final_output)

asyncio.run(main())