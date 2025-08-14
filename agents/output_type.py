
from agents import Agent, function_tool, Runner
from agentsdk_gemini_adapter import config
from pydantic import BaseModel


class CalendarEvent(BaseModel):
    name: str
    date_and_day: str  
    participants: list[str]


agent = Agent(
    name="Calendar extractor",
    instructions="Extract calendar events from text",
    output_type=CalendarEvent,
)

# query = "Add annual company picnic to calendar: October 12, 2025. Invite whole HR team."
# query = "Book a dentist appointment on September 5, 2025 for me and Dr. Hamid."
query = "Schedule project kickoff with Adeel, Faiza, and Sana."
result = Runner.run_sync(agent,query,run_config=config)
print(result.final_output)