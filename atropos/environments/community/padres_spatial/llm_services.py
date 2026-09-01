import asyncio
import json
import os
from pathlib import Path

import anthropic  # Ensure anthropic is imported
from dotenv import load_dotenv

print(
    "DEBUG LLM_SERVICES: Top of llm_services.py (v_robust_init)"
)  # Added version marker

dotenv_path = Path(__file__).resolve().parent.parent / ".env"
loaded_successfully = False
if dotenv_path.exists():
    # override=True ensures that values from .env file will replace existing env variables.
    # verbose=True will print messages about what it's doing.
    loaded_successfully = load_dotenv(
        dotenv_path=dotenv_path, override=True, verbose=True
    )
    print(
        f"DEBUG LLM_SERVICES: load_dotenv attempted from: {dotenv_path}. returned: {loaded_successfully}"
    )
else:
    print(f"DEBUG LLM_SERVICES: .env file not found at {dotenv_path}.")

API_KEY_FROM_ENV = os.getenv("ANTHROPIC_API_KEY")
print(f"DEBUG LLM_SERVICES: API_KEY_FROM_ENV raw value: '{API_KEY_FROM_ENV}'")

anthropic_client = None
IS_CLIENT_SUCCESSFULLY_INITIALIZED = False

if API_KEY_FROM_ENV:
    print(
        "DEBUG LLM_SERVICES: API_KEY_FROM_ENV is TRUTHY. Attempting to initialize Anthropic client."
    )
    try:
        anthropic_client = anthropic.Anthropic(api_key=API_KEY_FROM_ENV)
        IS_CLIENT_SUCCESSFULLY_INITIALIZED = True
        print("DEBUG LLM_SERVICES: Anthropic client INITIALIZED successfully.")
    except Exception as e:
        print(f"DEBUG LLM_SERVICES: FAILED to initialize Anthropic client. Error: {e}")
        # IS_CLIENT_SUCCESSFULLY_INITIALIZED remains False (or set explicitly)
        IS_CLIENT_SUCCESSFULLY_INITIALIZED = False
else:
    print(
        "DEBUG LLM_SERVICES: API_KEY_FROM_ENV is FALSY or None. Anthropic client will not be initialized."
    )
    IS_CLIENT_SUCCESSFULLY_INITIALIZED = False


async def get_anthropic_completion(
    prompt_text: str, model_name: str = "claude-3-5-sonnet-20240620"
):
    print("DEBUG LLM_SERVICES: Entered get_anthropic_completion function.")
    # Debug the state of client check variables right before the check
    print(
        f"DEBUG LLM_SERVICES: Inside get_anthropic_completion - "
        f"IS_CLIENT_SUCCESSFULLY_INITIALIZED: {IS_CLIENT_SUCCESSFULLY_INITIALIZED}, "
        f"anthropic_client is None: {anthropic_client is None}"
    )

    if not IS_CLIENT_SUCCESSFULLY_INITIALIZED or not anthropic_client:
        print(
            "DEBUG LLM_SERVICES: Anthropic client not available or not initialized. Returning mock response."
        )
        mock_action = {
            "action_type": "move_object",
            "object_id": "red_cube",
            "target_position": [0.1, 0.1, 0.1],
        }
        return json.dumps(mock_action)

    print(
        f"DEBUG LLM_SERVICES: Client seems OK. Proceeding to call Anthropic API with model: {model_name}"
    )
    try:
        loop = asyncio.get_event_loop()
        api_call_params = {
            "model": model_name,
            "max_tokens": 300,
            "messages": [
                {"role": "user", "content": prompt_text},
                {
                    "role": "assistant",
                    "content": "{",
                },  # Guide the model to start with JSON
            ],
        }
        print(
            f"DEBUG LLM_SERVICES: API Call Parameters: {json.dumps(api_call_params, indent=2)}"
        )

        response = await loop.run_in_executor(
            None, lambda: anthropic_client.messages.create(**api_call_params)
        )

        print(
            f"DEBUG LLM_SERVICES: Full API Raw Response Object: {str(response)}"
        )  # Use str(response) for safety

        if (
            response
            and response.content
            and isinstance(response.content, list)
            and len(response.content) > 0
            and hasattr(response.content[0], "text")
            and response.content[0].text
        ):
            raw_text_from_llm = response.content[0].text.strip()
            if raw_text_from_llm:
                llm_json_response = "{" + raw_text_from_llm
                print(f"DEBUG LLM_SERVICES: Raw LLM text part: '{raw_text_from_llm}'")
                print(
                    f"DEBUG LLM_SERVICES: Reconstructed LLM JSON: {llm_json_response}"
                )
                return llm_json_response
            else:
                print(
                    "DEBUG LLM_SERVICES: LLM response content[0].text was empty after stripping."
                )
                raise Exception("LLM returned empty text content.")
        else:
            print(
                f"DEBUG LLM_SERVICES: LLM response content was missing or malformed. "
                f"Response content: {str(response.content) if response else 'No response object'}"
            )
            raise Exception(
                "No valid content in LLM response or unexpected response structure."
            )

    except Exception as e:
        print(
            f"DEBUG LLM_SERVICES: Error during Anthropic API call or processing response: {e}"
        )
        mock_action = {
            "action_type": "move_object",
            "object_id": "red_cube",
            "target_position": [0.3, 0.3, 0.3],
        }  # Different mock for API error
        return json.dumps(mock_action)


print(
    "DEBUG LLM_SERVICES: End of llm_services.py execution during import (v_robust_init)."
)
