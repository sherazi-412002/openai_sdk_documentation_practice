from dataclasses import dataclass
from agents import Agent, function_tool, Runner, OpenAIChatCompletionsModel,set_tracing_disabled,model_settings
from openai import AsyncOpenAI
from agentsdk_gemini_adapter import config
import os
from dotenv import load_dotenv
import asyncio 
from typing import List


set_tracing_disabled(True)   # Global Configuration
load_dotenv()                 
gemini_api_key = os.getenv("GEMINI_API_KEY")


# Agent COnfiguration
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

external_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)



# Dummy Purchase class
@dataclass
class Purchase:
    item: str
    price: float

# 🧠 User-specific context with fetch_purchases
@dataclass
class UserContext:  # <- must inherit from Context
    uid: str
    is_pro_user: bool

    async def fetch_purchases(self) -> List[Purchase]:
        # Simulate fetching from DB or API
        return [
            Purchase(item="Python Course", price=49.99),
            Purchase(item="Streamlit Pro", price=20.00)
        ]


# 🛠️ Tool the agent can use
@function_tool
async def get_recent_purchases(context: UserContext) -> str:
    """Get the user's most recent purchases."""
    purchases = await context.fetch_purchases()
    if not context.is_pro_user == "False":
        return "No purchases found."
    
    lines = [f"- {p.item}: ${p.price}" for p in purchases]
    return "Here are your recent purchases:\n" + "\n".join(lines)


# 🤖 Create the agent
agent = Agent[UserContext](
    name="Helper",
    instructions=""""
    You are a helpful assistant.
        'You can retrieve the user's recent purchases by calling the get_recent_purchases tool when asked. '
        'The user context already contains all necessary information (including user ID and Pro status). '
        'Do NOT ask for confirmation of Pro status or user ID — just use the tool if the user asks about purchases.'
        'If the user is not a Pro user, the tool will return an error or no data.'
    """,
    tools=[get_recent_purchases],
    model=external_model
)


# 🔁 Run a single message
async def main():
    user_ctx = UserContext(uid="user-123", is_pro_user=False)

    response = await Runner.run(
        agent,
        "Can you show me my recent purchases?",
        context=user_ctx,
    )

    print(response.final_output)


# Run it
if __name__ == "__main__":
    asyncio.run(main())