from agents import Agent, Runner,trace, OpenAIChatCompletionsModel,set_tracing_disabled
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import asyncio


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

async def main():
    agent = Agent(name="Assistant",model=external_model, instructions="Reply very concisely.")

    thread_id = "thread_123"  # Example thread ID
    with trace(workflow_name="Conversation", group_id=thread_id):
        # First turn
        result = await Runner.run(agent, "What city is city of lights in pakistan?")
        print(result.final_output)
       

        # Second turn
        new_input = result.to_input_list() + [{"role": "user", "content": "In which province it is?"}]
        result = await Runner.run(agent, new_input)
        print(result.final_output)
        
        # Third Turn 
        new_input_2 = result.to_input_list() + [{"role": "user", "content": "name the airport in it?"}]
        result = await Runner.run(agent, new_input_2)
        print(result.final_output)


asyncio.run(main())