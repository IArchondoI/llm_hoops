"""Main entry point for the llm_hoops project."""

from llm_hoops.utils import load_data, load_text
from pathlib import Path
from llm_hoops.model.model import start_mistral_client, execute_prompt
from llm_hoops.rag.rag import add_rag_capabilities
from llm_hoops.function_calling.function_calling import setup_function_calling

USE_RAG = False
USE_FUNCTIONS = False
MODEL = "mistral-small-latest"



def main():
    """Main function to execute the program."""
    prompt = load_text(Path("data/query.txt"))

    df = load_data(Path("data/nba.csv"))

    client = start_mistral_client()
    
    rag = add_rag_capabilities(client,prompt) if USE_RAG else None

    function_call_objects = setup_function_calling(df) if USE_FUNCTIONS else None

    return execute_prompt(client,prompt,model=MODEL,rag=rag,function_calling=function_call_objects)

if __name__ == "__main__":
    main()
