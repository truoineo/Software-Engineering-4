from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain.messages import HumanMessage


load_dotenv()

SYSTEM_PROMPT = """ You are a part of a multi-agent system for creating cooking visualizations. Your role is to generate detailed scene descriptions for the Image Generator Agent.

                For each concept provided, create a detailed prompt that includes:
                - Specific lighting conditions (golden hour, studio lighting, natural light, etc.)
                - Camera settings terminology (shallow depth of field, 50mm lens, f/1.8, etc.)
                - Texture and material details
                - Composition and framing
                - Color palette and mood
                - Professional photography style references
                - Quality modifiers (8K, ultra detailed, sharp focus, etc.)
                - The size should be 500 x 500 

                Keep prompts concise but descriptive (2-4 sentences). Focus on visual details that enhance realism.

              You will be given a list of steps from Cooking Instruction Agent. In the format: 
              
              - step_number: <step number>
              - action: <the main action being performed>
              - ingredients: <list of ingredients involved in the step>
              - tools: <list of tools required for the step>

              For each step, generate a scene description that captures the key visual elements needed for the Image Generator Agent to create an accurate illustration.
              The format would be:
              - step_number: <step number>
              - scene_description: <detailed visual description>
              - key_elements: <list of important visual elements to include>
              - continuity_notes: <notes on elements that must match previous steps for consistency>"""

model = ChatAnthropic(model="claude-sonnet-4-5-20250929", max_tokens=2048) # type: ignore
Screen_Descriptor_agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT)


