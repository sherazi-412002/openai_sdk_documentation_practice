# import asyncio
# from pydantic import BaseModel
# from openai import AsyncOpenAI
# from dotenv import load_dotenv
# import os
# from agents import Agent, FunctionTool, RunContextWrapper, OpenAIChatCompletionsModel,set_tracing_disabled, Runner


# set_tracing_disabled(True)
# load_dotenv()
# gemini_api_key = os.getenv("GEMINI_API_KEY")


# external_client = AsyncOpenAI(
#     api_key=gemini_api_key,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
# )

# external_model = OpenAIChatCompletionsModel(
#     model="gemini-2.0-flash",
#     openai_client=external_client
# )


# spanish_agent = Agent(
#     name="Spanish agent",
#     model=external_model,
#     instructions="You translate the user's message to Spanish",
# )

# french_agent = Agent(
#     name="French agent",
#     model=external_model,
#     instructions="You translate the user's message to French",
# )

# orchestrator_agent = Agent(
#     name="orchestrator_agent",
#     instructions=(
#         "You are a translation agent. You use the tools given to you to translate."
#         "If asked for multiple translations, you call the relevant tools."
#     ),
#     model=external_model,
#     tools=[
#         spanish_agent.as_tool(
#             tool_name="translate_to_spanish",
#             tool_description="Translate the user's message to Spanish",
            
#         ),
#         french_agent.as_tool(
#             tool_name="translate_to_french",
#             tool_description="Translate the user's message to French",
#         ),
#     ],
# )

# async def main():
#     result = await Runner.run(orchestrator_agent, input="Say 'Hello, how are you?' in Spanish.")
#     print(result.final_output)


# asyncio.run(main())




import asyncio
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from agents import Agent, FunctionTool, RunContextWrapper, OpenAIChatCompletionsModel, RunResult, ToolCallOutputItem, set_tracing_disabled, Runner

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

# ex 01

# # Define a custom output extractor to get only the final translated text
# def extract_translation(result):
#     # Assuming result.final_output contains the translation
#     return result.final_output if result.final_output else "No translation found"


# ex 02
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

spanish_agent = Agent(
    name="Spanish agent",
    model=external_model,
    instructions="You translate the user's message to Spanish",
)

french_agent = Agent(
    name="French agent",
    model=external_model,
    instructions="You translate the user's message to French",
)

orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "You are a translation agent. You use the tools given to you to translate. "
        "If asked for multiple translations, you call the relevant tools."
    ),
    model=external_model,
    tools=[
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translate the user's message to Spanish",
            custom_output_extractor=extract_json_payload  # Add custom extractor
        ),
        french_agent.as_tool(
            tool_name="translate_to_french",
            tool_description="Translate the user's message to French",
            custom_output_extractor=extract_json_payload  # Add custom extractor
        ),
        json_tool
    ],
)

async def main():
    result = await Runner.run(
        orchestrator_agent,
        "Say 'Hello, how are you?' in Spanish and French."
    )
    print(result.final_output)

asyncio.run(main())