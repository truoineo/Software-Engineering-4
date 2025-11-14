from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
import base64
from pathlib import Path

from screen_descriptor_agent import Screen_Descriptor_agent
from instruction_parser_agent import Instruction_Parser_agent
from image_generator_agent import Image_Generator_agent


@tool("instruction_parser_agent", description = "Parses cooking instructions into structured steps.")
def call_instruction_parser_agent(query: str):
    result = Instruction_Parser_agent.invoke({"messages": [HumanMessage(content=query)]})
    parsed_steps = result["messages"][-1].content
    print("Instruction Parsers: ", parsed_steps)
    return parsed_steps

@tool("scene_descriptor_agent", description = "Generates detailed scene descriptions for image generation.")
def call_scene_descriptor_agent(parsed_steps: str):
    result = Screen_Descriptor_agent.invoke({"messages": [HumanMessage(content=parsed_steps)]})
    scene_descriptions = result["messages"][-1].content
    print("Scene Descriptors: ", scene_descriptions)
    return scene_descriptions

@tool("image_generator_agent", description = "Generates an image correspond to a single step.")
def call_image_generator_agent(scene_description: str, step_number: int, key_elements: str, continuity_notes: str):
    response = Image_Generator_agent.invoke({"messages": [HumanMessage(content=scene_description)]})
    #image_url = response["messages"][-1].content[-1]["image_url"]["url"].split(',')[1]
    ai_message_content = response["messages"][-1].content
    
    # Find the image_url in the content (it could be at any index)
    image_url = None
    for item in ai_message_content:
        if isinstance(item, dict) and 'image_url' in item:
            full_url = item['image_url']['url']
            # Extract base64 data after the comma (if present)
            if ',' in full_url:
                image_url = full_url.split(',', 1)[1]
            else:
                image_url = full_url
            break
    
    if not image_url:
        raise ValueError("No image found in response")
    
    #word = response["messages"][-1].content[0]
    image_bytes = base64.b64decode(image_url)
    output_path = Path(f"step_{step_number}_image.png")
    output_path.write_bytes(image_bytes)
    
    print(f"Image for step {step_number} saved to {output_path.absolute()}")
    return f"Image for step {step_number} saved to {output_path.absolute()}"
    

SYSTEM_PROMPT = """ You are the Orchestrator Agent in a multi-agent system for creating cooking visualizations. Your role is to coordinate between the Instruction Parser Agent, Scene Descriptor Agent, and Image Generator Agent to produce a series of images that illustrate cooking steps clearly and accurately.
Each agent will correspond to a tool that you have. Given a dish name, you will provide: 
1. Provide cooking instruction to the Instruction Parser Agent, receive structured steps
2. Pass those to the Scene Descriptor Agent to get detailed scene descriptions.
3. Finally send those descriptions to the Image Generator Agent to create the images.

What out for the format inputs and output of each agents:
- Instruction Parser Agent Input: Raw cooking instructions
- Instruction Parser Agent Output: Structured steps with step_number, action, ingredients, tools
- Scene Descriptor Agent Input: Structured steps from Instruction Parser Agent
- Scene Descriptor Agent Output: scene_description, step_number, key_elements, continuity_notes

After receiving the descriptions of each step from the Scene Descriptor Agent, you will sequentially pass them to the Image Generator Agent to create images for each step.
- Image Generator Agent Input: scene_description, step_number, key_elements, continuity_notes"""



Orchestrator_Agent = create_agent(
    model=ChatAnthropic(model="claude-sonnet-4-5-20250929", max_tokens=2048), # type: ignore
    system_prompt=SYSTEM_PROMPT,
    tools=[call_instruction_parser_agent, call_scene_descriptor_agent, call_image_generator_agent]
)

Orchestrator_Agent.invoke({"messages": [{"role": "user", "content": "Create cooking visualization images for 'Spaghetti Carbonara' recipe"}]})