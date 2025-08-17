
from agents import Agent, function_tool, Runner
from agentsdk_gemini_adapter import config
from dataclasses import dataclass


# class CalendarEvent(BaseModel):
#     name: str
#     date_and_day: str  
#     participants: list[str]
@dataclass
class StudentInfo():
    name:str
    st_id:int
    


agent = Agent(
    name="Calendar extractor",
    instructions="Always respond in haiku form with poetic flair, avoiding structured JSON output.",
    output_type=StudentInfo,
)

# query = "Add annual company picnic to calendar: October 12, 2025. Invite whole HR team."
# query = "Book a dentist appointment on September 5, 2025 for me and Dr. Hamid."
query = "my name is ali and my id is 00123."

result = Runner.run_sync(agent,query,run_config=config)
print(result.final_output)
print(type(result.final_output))