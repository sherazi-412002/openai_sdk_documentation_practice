from agents import Agent, function_tool, Runner, OpenAIChatCompletionsModel, set_tracing_disabled, RunContextWrapper, handoff, enable_verbose_stdout_logging
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel


enable_verbose_stdout_logging()
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


class EscalationData(BaseModel):
    reason: str


async def on_handoff(ctx: RunContextWrapper[None], input_data: EscalationData):
    print(f"Escalation agent called with reason: {input_data.reason}")


escalating_agent = Agent(name="Escalation Agent",instructions="You are an escalating agent your task is to take imediate action on the response you recive from triage agent and handle the situation professionaly.", model=external_model)

handoff_obj = handoff(
    agent=escalating_agent,
    on_handoff=on_handoff,
    input_type=EscalationData,  # Must match the JSON structure returned
)

agent = Agent(
    name="Triage Agent",
    instructions="""
You are a triage agent. If the user's query is urgent, angry, or mentions legal action, refund, or complaint, 
you must handoff to escalating agent with a JSON object: {"reason": "your reason here"}.

Do NOT say anything else if escalating transfer to escalating_agent to handle the situation. If no escalation is needed, answer normally.
""",
    model=external_model,
    handoffs=[handoff_obj],
)

# ✅ This query should now trigger the handoff
result = Runner.run_sync(agent, "I'm furious — your service is terrible and I want to speak to a manager.")

print("Last Agent:", result.last_agent.name)
print("Final Output:", result.final_output)


