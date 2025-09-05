from agents import Agent, function_tool, Runner,ModelSettings, OpenAIChatCompletionsModel,set_tracing_disabled,RunContextWrapper, handoff,enable_verbose_stdout_logging
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os


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


# def on_handoff(ctx:RunContextWrapper[None]):
#     print("Handoff called!")

# @function_tool
# def weather(city):
#     return f"The weather in {city} is sunny!"



# math_tutor = Agent(
#     name="Math Tutor",
#     instructions="You are a math Tutor and answer to only math related queries.",
#     model=external_model,
#     handoff_description="You are a math Tutor and answer to only math related queries." 
# )

# history_tutor = Agent(
#     name="History Tutor",
#     instructions="You are a Tutor and answer to only queries.",
#     model=external_model,
#     handoff_description="You are a history Tutor and answer to only history related queries."
# )  


# biology_tutor = handoff(
#     agent=history_tutor,
#     tool_name_override="biology_tutor",
#     tool_description_override="You are a biology Tutor and answer to only biology related queries.",
#     on_handoff=on_handoff,    
# )


triage_agent = Agent(
    name="Triage Agent",
    instructions="You are triage agent and your task is to handoff user queries according provided agent description." \
    "if user query is not related to handsoff agents than answer to user through triage agent.",
    model=external_model,
    model_settings=ModelSettings(
        temperature=0.9,
        frequency_penalty=1
        # max_tokens=300,    
        # presence_penalty=1,
    ),
    # tools=[weather],
    # handoffs=[history_tutor,math_tutor,biology_tutor]
)

result = Runner.run_sync(triage_agent,"Name three animals that lives in the jungle.")
print(result.last_agent.name)
print(result.final_output)