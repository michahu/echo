from enum import Enum
import os

from azure.identity import ManagedIdentityCredential, get_bearer_token_provider
from openai import AzureOpenAI
from src.async_collab.llm.llm_client import LLMClient


class LLMModelName(Enum):
    dev_gpt_4_turbo = "dev-gpt-4-turbo"
    dev_gpt_35_turbo = "dev-gpt-35-turbo"
    dev_gpt_4_turbo_chat_completions = "dev-gpt-4-turbo-chat-completions"
    dev_gpt_4o_2024_05_13 = "dev-gpt-4o-2024-05-13"
    dev_phi3_medium_128k_instruct = "dev-phi-3-medium-128k-instruct"
    dev_gpt_4o_2024_05_13_chat_completions = "dev-gpt-4o-2024-05-13-chat-completions"


class MyLLMClient(LLMClient):
    def __init__(self, model: str = "gpt-4o-11-20"):
        self.model = model

        endpoint = ""
        credential = ManagedIdentityCredential()
        token_provider = get_bearer_token_provider(
            credential, ""
        )

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version="2024-12-01-preview",
        )

    def send_request(self, request, model) -> dict:
        """
        Send a request to the LLM API.
        """

        return self.client.chat.completions.create(model=model, **request)

    def get_response_str(
        self,
        user_prompt: str,
        temperature: float = 0,
        max_tokens: int = 10000,
        top_p: float = 0.95,
        system_instruction: str = "",
        stop: str | None = None,
        model: str | None = None,
    ) -> str | None:
        if model is None:
            model = self.model

        try:
            # Prepare messages array
            messages = []

            # Add system message if provided
            if len(system_instruction) > 0:
                messages.append({"role": "system", "content": system_instruction})

            # Add user message
            messages.append({"role": "user", "content": user_prompt})

            
            if model == "o3-mini":
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    stop=stop,
                )
            else:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop=stop,
                )

            # Extract and return the response content
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                print("[MyLLMClient] get_response_str: No choices in response")
                return None

        except Exception as e:
            print(f"[MyLLMClient] get_response_str: Exception occurred: {e}")
            return None


llm_client: LLMClient | None = None


def get_llm_client(model) -> LLMClient:
    global llm_client
    if llm_client is None or llm_client.model != model:
        llm_client = MyLLMClient(model=model)
    return llm_client
