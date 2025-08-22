# app/llms/custom.py

import requests
import json
from typing import Any, List, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage, AIMessage

class CustomChatModel(SimpleChatModel):
    """A custom chat model that calls a remote FastAPI endpoint."""

    api_url: str

    @property
    def _llm_type(self) -> str:
        return "custom_chat_model"

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        raw_prompt = messages[-1].content
        headers = {"Content-Type": "application/json"}
        data = {"prompt": raw_prompt}

        try:
            response = requests.post(self.api_url, headers=headers, data=json.dumps(data))
            # Raise an error for bad responses
            response.raise_for_status()

            result = response.json()
            full_text = result.get("response", "")

            parts = full_text.split("<|start_header_id|>assistant<|end_header_id|>\n\n")
            if len(parts) > 1:
                # an empty response is also treated as a failure
                assistant_response = parts[1].replace("<|eot_id|>", "").strip()
                if not assistant_response:
                    raise ValueError("Model returned an empty response.")
                return assistant_response
            else:
                # If parsing fails, raise an error to trigger the fallback
                raise ValueError("Could not parse the model's response.")

        except (requests.exceptions.RequestException, ValueError) as e:
            
            print(f"Custom model failed: {e}. Attempting fallback.")
            raise