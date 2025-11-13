import os
from typing import List, Optional

# -----------------------------
# Document loaders
# -----------------------------
from langchain_community.document_loaders import TextLoader, PyPDFLoader

def load_document(file_path: str) -> str:
    """Load text or PDF and return content as string."""
    file_path = os.path.abspath(file_path)
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file format. Use .txt or .pdf")
    docs = loader.load()
    return "\n".join([d.page_content for d in docs])

# -----------------------------
# Pydantic models
# -----------------------------
from pydantic import BaseModel, Field, ValidationError

class Ingredient(BaseModel):
    name: str
    quantity: Optional[str] = None
    notes: Optional[str] = None

class StepAction(BaseModel):
    action: str
    details: str
    ingredients_used: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)

class ParsedRecipe(BaseModel):
    title: Optional[str] = None
    yield_: Optional[str] = None
    ingredients: List[Ingredient] = Field(default_factory=list)
    steps: List[StepAction] = Field(default_factory=list)
    notes: Optional[str] = None
    source_filename: Optional[str] = None

# -----------------------------
# LangChain imports
# -----------------------------
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain.chains import LLMChain


# -----------------------------
# Pydantic parser setup
# -----------------------------
pydantic_parser = PydanticOutputParser(pydantic_object=ParsedRecipe)

SYSTEM_PROMPT = """
You are an instruction parser for recipes. Return JSON exactly matching the schema.
Return ONLY JSON—no explanation.
"""

HUMAN_PROMPT = """
Recipe text:
{document_text}

Format instructions:
{format_instructions}
"""

# -----------------------------
# Claude model
# -----------------------------
def get_anthropic_model(temperature: float = 0.0):
    return ChatAnthropic(
        temperature=temperature,
        model="claude-2.1",
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

# -----------------------------
# Parse instructions function
# -----------------------------
def parse_instructions(file_path: str, temperature: float = 0.0) -> ParsedRecipe:
    raw_text = load_document(file_path)
    format_instructions = pydantic_parser.get_format_instructions()
    human_text = HUMAN_PROMPT.format(document_text=raw_text, format_instructions=format_instructions)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(human_text)
    ])

    model = get_anthropic_model(temperature=temperature)
    chain = LLMChain(llm=model, prompt=prompt)
    raw_response = chain.run(document_text=raw_text)

    try:
        parsed = pydantic_parser.parse_raw(raw_response)
    except ValidationError:
        first = raw_response.find("{")
        last = raw_response.rfind("}")
        if first != -1 and last != -1:
            parsed = pydantic_parser.parse_raw(raw_response[first:last+1])
        else:
            raise RuntimeError("No valid JSON in model output")

    parsed.source_filename = os.path.basename(file_path)
    return parsed

# -----------------------------
# Example run
# -----------------------------
if __name__ == "__main__":
    example_file = "examples/sample_recipe.txt"
    parsed_recipe = parse_instructions(example_file)
    print(parsed_recipe.model_dump_json(indent=2))
