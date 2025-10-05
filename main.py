"""Main entry point for the llm_hoops project."""

from llm_hoops.utils import load_data, load_query
from pathlib import Path
from llm_hoops.model.model import start_mistral_client, execute_prompt
from llm_hoops.rag.rag import add_rag_capabilities
from llm_hoops.function_calling.function_calling import setup_function_calling

USE_RAG = False
USE_FUNCTIONS = False


def main():
    """Main function to execute the program."""
    query = load_query(Path("data/query.txt"))

    _df = load_data(Path("data/nba.csv"))

    client = start_mistral_client()
    if USE_RAG:
        client = add_rag_capabilities(client)
    if USE_FUNCTIONS:
        client = setup_function_calling(client)

    return execute_prompt(client, query)


if __name__ == "__main__":
    main()
