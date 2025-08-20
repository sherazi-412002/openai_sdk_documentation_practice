import asyncio
from dataclasses import dataclass
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from agents import Agent, OpenAIChatCompletionsModel,function_tool, set_tracing_disabled, Runner,RunContextWrapper

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

@dataclass
class UserInfo:  
    name: str
    uid: int

@function_tool
async def fetch_user(wrapper: RunContextWrapper[UserInfo]) -> str:  
    """Fetch the age and name of the user. Call this function to get user's name and age information."""
    return f"The user {wrapper.context.name} is 47 years old"

@function_tool
async def fetch_user_id(wrapper: RunContextWrapper[UserInfo]) -> str:  
    """Fetch the user_id of the user. Call this function to get user's id information."""
    return f"The user id is {wrapper.context.uid} ."


async def main():
    user_info = UserInfo(name="Ali", uid=123)

    agent = Agent[UserInfo](  
        name="Assistant",
        model=external_model,
        tools=[fetch_user,fetch_user_id],
    )

    result = await Runner.run(  
        starting_agent=agent,
        input="What is the name, age and user id of the user?",
        context=user_info,
    )

    print(result.final_output)  
    # The user John is 47 years old.

if __name__ == "__main__":
    asyncio.run(main())