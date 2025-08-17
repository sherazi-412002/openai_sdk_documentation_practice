from agents import Agent, Runner,trace
from agentsdk_gemini_adapter import config
import asyncio

async def main():
    agent = Agent(
        name="Assistant",
        # instructions="Always respond in haiku form with poetic flair, avoiding structured JSON output.",
    )

    # query = "Add annual company picnic to calendar: October 12, 2025. Invite whole HR team."
    # query = "Book a dentist appointment on September 5, 2025 for me and Dr. Hamid."
    # query = "my name is ali and my id is 00123."
    thread_id = "thread_123"
    with trace(workflow_name="Testing",group_id=thread_id):
            
        result = await Runner.run(agent,"Who is founder of pakistan?",run_config=config)
        print(result.final_output)
        
        new_result = result.to_input_list() + [{"role": "user","content": "what is his father name"}]
        result = await Runner.run(agent,new_result,run_config=config)
        print(result.final_output)
        print(result.last_agent.name)



asyncio.run(main())