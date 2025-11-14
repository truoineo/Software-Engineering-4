# Software-Engineering-4
Multi-Agent System
This project features an intelligent agent that generates step-by-step visual instructions for cooking recipes. You can request a recipe for any dish, and the agent will provide a detailed guide along with corresponding images for each step.


# Project Overview

Multi-agent system that generates step-by-step cooking visualizations. Given a dish name, it:

- Parses cooking instructions into structured steps  
- Creates detailed scene descriptions for each step  
- Generates comic-style illustrations for each step  

# Architecture

The system uses 4 agents:

## Instruction Parser Agent (`instruction_parser_agent.py`)
- Uses Claude Sonnet 4.5  
- Breaks raw cooking instructions into structured steps with:  
  - `step_number`  
  - `action`  
  - `ingredients`  
  - `tools`

## Scene Descriptor Agent (`screen_descriptor_agent.py`)
- Uses Claude Sonnet 4.5  
- Converts parsed steps into detailed visual descriptions  
- Outputs:  
  - `step_number`  
  - `scene_description`  
  - `key_elements`  
  - `continuity_notes`

## Image Generator Agent (`image_generator_agent.py`)
- Uses Google Gemini 2.5 Flash (image generation model)  
- Generates comic-style cooking illustrations (1024x1024px)  
- Maintains visual consistency across steps (character, kitchen, ingredients)

## Orchestrator Agent (`orchestrator.py`)
- Coordinates all agents  
- Manages workflow: **Parse → Describe → Generate Images**

---

### Example
Creates images for the **Spaghetti Carbonara** recipe.
