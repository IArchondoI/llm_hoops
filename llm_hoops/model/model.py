"""Hold basic model specific capabilities."""

from llm_hoops.model.mistral_credentials import MISTRAL_API_KEY
from mistralai import Mistral


def start_mistral_client()->Mistral:
    """Start Mistral Model."""
    return Mistral(api_key=MISTRAL_API_KEY)


def execute_prompt(client:Mistral,prompt:str,model:str="mistral-large-latest")->str:
    """Execute prompt."""
    response = client.chat.complete(
        model=model,
        messages=[
            {
                "role":"user",
                "content":prompt
            },
        ]
    )
    return str(response.choices[0].message.content)

    