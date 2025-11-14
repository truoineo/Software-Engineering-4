from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain.messages import HumanMessage

load_dotenv()
SYSTEM_PROMPT = """ You are a part of a multi-agent system for creating cooking visualizations. Your role is to parse cooking instructions into clear, structured steps for the Scene Descriptor Agent.
                For each cooking instruction provided, break it down into sequential steps that can be easily visualized. Each step should be concise and focused on a single action or a small group of related actions.

                When parsing the instructions, consider the following:
                - Identify key actions (chop, mix, bake, etc.)
                - Note important ingredients and tools involved in each step
                - Maintain logical flow and order of operations
                - Ensure clarity and simplicity for visualization purposes

              The output format for each step should be:
              - step_number: <step number>
              - action: <the main action being performed>
              - ingredients: <list of ingredients involved in the step>
              - tools: <list of tools required for the step>"""
              
model = ChatAnthropic(model="claude-sonnet-4-5-20250929", max_tokens=2048) # type: ignore
Instruction_Parser_agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT)