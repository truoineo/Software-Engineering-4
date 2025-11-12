import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# turn into tools nad subagent 
# return array of prompts 
# create it as an agent 

load_dotenv()

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    max_tokens=500,
    temperature=0.7
)

def create_prompt_chain():
    """
    Create a LangChain chain for generating photo-realistic prompts.
    
    Returns:
        Chain: LangChain runnable chain
    """
    
    template = """You are an expert at creating prompts for AI image generation that produce clear, crisp, and photo-realistic images. 

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

Concept: {concept}

Photo-realistic prompt:"""

    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"concept": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

def generate_photorealistic_prompt(concept, chain):
    """
    Generate a detailed, photo-realistic prompt using LangChain.
    
    Args:
        concept (str): Basic concept or subject for the image
        chain: LangChain chain for generation
    
    Returns:
        str: Enhanced prompt optimized for photo-realistic images
    """
    try:
        result = chain.invoke(concept)
        return result.strip()
    except Exception as e:
        return f"Error generating prompt: {e}"

def batch_generate_prompts(concepts, chain):
    """
    Generate prompts for multiple concepts using batch processing.
    
    Args:
        concepts (list): List of image concepts
        chain: LangChain chain for generation
    
    Returns:
        list: List of enhanced prompts
    """
    try:
        results = chain.batch(concepts)
        return [result.strip() for result in results]
    except Exception as e:
        print(f"Batch processing error: {e}")
        return [generate_photorealistic_prompt(concept, chain) for concept in concepts]

def process_prompt_array(concepts, use_batch=True):
    """
    Process an array of concepts and generate prompts for each.
    
    Args:
        concepts (list): List of image concepts
        use_batch (bool): Whether to use batch processing
    
    Returns:
        dict: Dictionary mapping concepts to their enhanced prompts
    """
    results = {}
    chain = create_prompt_chain()
    
    print("Generating photo-realistic prompts with LangChain...\n")
    print("=" * 70)
    
    if use_batch and len(concepts) > 1:
        print(f"\nProcessing {len(concepts)} concepts in batch mode...")
        enhanced_prompts = batch_generate_prompts(concepts, chain)
        
        for i, (concept, prompt) in enumerate(zip(concepts, enhanced_prompts), 1):
            print(f"\n[{i}/{len(concepts)}] Concept: {concept}")
            print("-" * 70)
            print(f"Enhanced Prompt:\n{prompt}\n")
            results[concept] = prompt
    else:
        for i, concept in enumerate(concepts, 1):
            print(f"\n[{i}/{len(concepts)}] Processing: {concept}")
            print("-" * 70)
            
            enhanced_prompt = generate_photorealistic_prompt(concept, chain)
            results[concept] = enhanced_prompt
            
            print(f"Enhanced Prompt:\n{enhanced_prompt}\n")
    
    return results

def save_prompts_to_file(results, filename="gemini_prompts.txt"):
    """
    Save generated prompts to a text file.
    
    Args:
        results (dict): Dictionary of concepts and prompts
        filename (str): Output filename
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("GEMINI AI PHOTO-REALISTIC IMAGE PROMPTS\n")
        f.write("=" * 70 + "\n\n")
        
        for concept, prompt in results.items():
            f.write(f"CONCEPT: {concept}\n")
            f.write(f"PROMPT: {prompt}\n")
            f.write("-" * 70 + "\n\n")
    
    print(f"\n✓ Prompts saved to: {filename}")

def main():
    """Main function to run the prompt generator."""
    print("\n GEMINI AI PHOTO-REALISTIC PROMPT GENERATOR ")
    
    # Put recipe parse here
    concepts = [
        "Boil Water",
        "Add pasta",
        "Add salt",
        "Drain water using a strainer"
    ]
    
    print("Input concepts:")
    for i, concept in enumerate(concepts, 1):
        print(f"  {i}. {concept}")
    
    print("\n" + "=" * 70)
    
    results = process_prompt_array(concepts, use_batch=True)
    
    save_prompts_to_file(results)
    
    print("\n All prompts generated successfully!")
    print("\nYou can now copy these prompts into Gemini AI for image generation.")

if __name__ == "__main__":
    main()