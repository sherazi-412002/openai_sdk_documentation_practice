from typing import Any
import rich
import json
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from agents import Agent, FunctionTool, RunContextWrapper, OpenAIChatCompletionsModel,set_tracing_disabled, Runner


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


def do_some_work(data: str) -> str:
    return "done"


class FunctionArgs(BaseModel):
    username: str
    age: int


async def run_function(ctx: RunContextWrapper[Any], args: str) -> str:
    parsed = FunctionArgs.model_validate_json(args)
    return do_some_work(data=f"{parsed.username} is {parsed.age} years old")



tool = FunctionTool(
    name="process_user",
    description="Processes extracted user data",
    params_json_schema=FunctionArgs.model_json_schema(),
    on_invoke_tool=run_function,
)

# user_info = FunctionArgs("Ali",25)

agent = Agent(
    name="Assistant",
    model=external_model,
    tools=[tool],  
)

# for tool in agent.tools:
#     if isinstance(tool, FunctionTool):
#         print(tool.name)
#         print(tool.description)
#         print(json.dumps(tool.params_json_schema, indent=2))
#         print()

result = Runner.run_sync(agent,"My name is Syed Shoaib And my age is 23.")

print(result.final_output)
# rich.print(result.new_items)
print(do_some_work())