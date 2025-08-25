from pydantic import BaseModel
from agents import (
    Agent,
    GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    output_guardrail,
    set_tracing_disabled,
    enable_verbose_stdout_logging,
    OpenAIChatCompletionsModel,
    
)
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio




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


class MessageOutput(BaseModel): 
    response: str

class MathHomeworkOutput(BaseModel):
    is_math: bool
    reasoning: str

guardrail_agent = Agent( 
    name="Guardrail check",
    instructions="Check if the output includes any math.",
    output_type=MathHomeworkOutput,
    model=external_model
)



@output_guardrail
async def math_guardrail( 
    ctx: RunContextWrapper[None], agent: Agent, output:MessageOutput
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, output.response , context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output, 
        tripwire_triggered=result.final_output.is_math,
    )


agent = Agent(  
    name="Customer support agent",
    instructions="You are a customer support agent. You help customers with their questions.",
    output_guardrails=[math_guardrail],
    output_type=MessageOutput,
    model=external_model
)

async def main():
    # This should trip the guardrail
    try:
        await Runner.run(agent, "Hello, can you help me solve for x: 2x + 3 = 11?")
        print("Guardrail didn't trip - this is unexpected")

    except OutputGuardrailTripwireTriggered:
        print("Math homework guardrail tripped")


asyncio.run(main())
