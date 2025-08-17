import asyncio
from agents import Agent, function_tool, Runner, OpenAIChatCompletionsModel,set_tracing_disabled
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
import os
from dotenv import load_dotenv
import rich
import time



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


# async def main():
#     agent = Agent(
#         name="Joker",
#         instructions="You are a helpful assistant.",
#         model=external_model
#     )

#     # Start the timer
#     start_time = time.perf_counter()

#     result = Runner.run_streamed(agent, "Give me 20 Jokes.")
#     async for events in result.stream_events():
#         if events.type == "raw_response_event" and isinstance(events.data, ResponseTextDeltaEvent):
#             rich.print(events.data.delta, end="", flush=True)

#     # End the timer
#     end_time = time.perf_counter()
#     elapsed_time = end_time - start_time

#     # Print the total time taken
#     rich.print(f"\nTotal time taken to print jokes: {elapsed_time:.2f} seconds")

# # Run the async function
# asyncio.run(main())



async def main():
    agent = Agent(
        name="Joker",
        instructions="You are a helpful assistant.",
        model=external_model
    )

    result = Runner.run_streamed(agent, "Give me 20 Jokes.")
    async for events in result.stream_events():
        # if events.type == "raw_response_event" and isinstance(events.data, ResponseTextDeltaEvent):
        rich.print(events)



# Run the async function
asyncio.run(main())