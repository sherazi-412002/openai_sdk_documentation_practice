from agents import Agent, function_tool, Runner, OpenAIChatCompletionsModel,set_tracing_disabled,RunContextWrapper
from openai import AsyncOpenAI
from agentsdk_gemini_adapter import config
import os
from dotenv import load_dotenv
from dataclasses import dataclass

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
class WeatherContext:
    default_city: str 

    def fetch_weather(self, city: str | None = None) -> str:
        return f"The weather in {city or self.default_city} is sunny!"


@function_tool
def fetch_weather_tool(context:RunContextWrapper[WeatherContext],city: str) -> str:
    print(context.usage)
    if city:
        return context.context.fetch_weather(city)
    
    return context.context.fetch_weather()
    



agent = Agent[WeatherContext](
    name="Haiku Agent",
    instructions="""
    You are a helpful assistant.
        You can fetch the weather by calling the fetch_weather_tool tool when asked.
        If you not find city from user prompt then use default city name.
        The weather context already contains all necessary information city name.
        Do NOT ask for confirmation for declare city name — just use the tool if the user asks about weather.
    """,
    model=external_model,
    tools=[fetch_weather_tool]
)

ctx = WeatherContext(default_city="Lahore")

result = Runner.run_sync(agent,"what is the weather in karachi?",context=ctx)
print(result.final_output)

