"""Hold basic model specific capabilities."""

from llm_hoops.model.mistral_credentials import MISTRAL_API_KEY
from mistralai import Mistral
from llm_hoops.function_calling.function_calling import FunctionCallObjects
from typing import TypedDict, Union, List, Dict, Optional, Sequence, Mapping, Any
import json
from dataclasses import dataclass


@dataclass
class Output:
    """Hold LLM output."""

    response: str
    message_history: List[Dict[str, Union[str, None]]]


class Parameters(TypedDict):
    type: str
    properties: Mapping[str, str]
    required: Sequence[str]


class Function(TypedDict):
    name: str
    description: str
    parameters: Parameters


def start_mistral_client() -> Mistral:
    """Start Mistral Model."""
    return Mistral(api_key=MISTRAL_API_KEY)


def execute_function_calling_prompt(
    client: Mistral,
    message_history: Any,
    function_calling: FunctionCallObjects,
    model: str = "mistral-large-latest",
    max_iterations: int = 5,
) -> Output:
    """Execute prompt with possible multi-step function calling."""
    # Initial call
    response = client.chat.complete(
        model=model,
        messages=message_history,
        tools=function_calling.tools,
        tool_choice="any" if function_calling else None,
        parallel_tool_calls=False,
    )

    iteration = 0

    # Keep looping while model wants to call tools
    while (
        hasattr(response.choices[0].message, "tool_calls")
        and response.choices[0].message.tool_calls
        and iteration < max_iterations
    ):
        iteration += 1
        tool_calls = response.choices[0].message.tool_calls
        message_history.append(response.choices[0].message)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = {}

            if tool_call.function.arguments:
                try:
                    function_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    print(
                        f"Invalid JSON for {function_name}: {tool_call.function.arguments}"
                    )
                    continue

            func = function_calling.names_to_functions.get(function_name)
            if not func:
                print(f"Unknown function: {function_name}")
                continue

            # Call the tool function
            result = func(**function_args) if function_args else func()

            # Append tool result
            message_history.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": str(result),
                }
            )

        # Send the new message history back to the model
        response = client.chat.complete(
            model=model,
            messages=message_history,
            tools=function_calling.tools,
            tool_choice="auto",
            parallel_tool_calls=False,
        )

    # When model stops calling tools, take the final message as the answer
    final_answer = str(response.choices[0].message.content)
    message_history.append({"role": "assistant", "content": final_answer})

    return Output(response=final_answer, message_history=message_history)


def execute_prompt(
    client: Mistral,
    prompt: str,
    model: str = "mistral-large-latest",
    function_calling: Optional[FunctionCallObjects] = None,
) -> Output:
    """Execute prompt."""
    message_history: Any = [
        {"role": "user", "content": prompt},
    ]

    if not function_calling:
        response = client.chat.complete(model=model, messages=message_history)
        return Output(response=str(response), message_history=message_history)
    else:

        return execute_function_calling_prompt(
            client, message_history, function_calling, model
        )
