import json

from azure.identity import ManagedIdentityCredential, get_bearer_token_provider
from openai import AzureOpenAI

from llm_local import get_vllm_client


model = "gpt-4o"
endpoint = ""
credential = ManagedIdentityCredential()
token_provider = get_bearer_token_provider(
    credential, ""
)

client = AzureOpenAI(
    azure_endpoint=endpoint,
    azure_ad_token_provider=token_provider,
    api_version="2024-12-01-preview",
)


def get_response_from_gpt_azure(
    model,
    messages,
    reasoning_effort="low",
):
    print("*** model = ", model)
    if model == "o3-mini":
        # For o3-mini, we use a different response format
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=10000,
            n=1,
            stop=None,
            response_format={"type": "json_object"},
            reasoning_effort=reasoning_effort,
        )
        content = response.choices[0].message.content
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            n=1,
            stop=None,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content

    json_dict = json.loads(content)
    return json_dict, None, None


def my_process_choice(choice):
    return choice


response_processor_factory = None


def extract_thinking_and_content(raw_content):
    """Extract thinking content and main response from raw content.

    Args:
        raw_content: The raw response content that may contain <think>...</think> tags

    Returns:
        tuple: (thought, cleaned_content) where thought is the extracted thinking
               and cleaned_content is the main response after the thinking tags
    """
    start_tag = "<think>"
    end_tag = "</think>"

    start_index = raw_content.find(start_tag)
    end_index = raw_content.find(end_tag, start_index + len(start_tag))

    if start_index != -1 and end_index != -1:
        # Extract the thought content between the tags
        thought = raw_content[start_index + len(start_tag) : end_index].strip()
        # The main content comes after the closing tag
        content = raw_content[end_index + len(end_tag) :].strip()
        return thought, content
    else:
        # No thinking tags found, return None for thought and full content
        return None, raw_content.strip()


def get_response_from_local(
    model,
    messages,
    is_thinking=False,
):
    """
    Use the local vLLM client for making requests
    """
    vllm_client = get_vllm_client(model_name=model)

    # Check if server is available
    if not vllm_client.is_available():
        raise Exception(
            f"vLLM server is not available. Start it with: python -m vllm.entrypoints.openai.api_server --model {model} --port 8000"
        )

    # Prepare the request
    response = vllm_client.send_request(
        messages=messages, temperature=0.0, max_tokens=800
    )

    # Ensure response is parsed as JSON
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON response from vLLM backend.")

    content = response["choices"][0]["message"]["content"].strip()

    print("content = ", content)

    if is_thinking:
        thought, content = extract_thinking_and_content(content)
    else:
        thought = None

    content_dct = content
    if isinstance(content_dct, str):
        try:
            content_dct = json.loads(content_dct)
        except json.JSONDecodeError:
            content_dct = {"content": content}

    return content_dct, None, thought



def main():
    # Simple test like in llm_substrate.py
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    # Test with local vLLM backend
    print("\nTesting with local vLLM backend...")
    try:
        result, _, thought = get_response_from_local(
            model="Qwen/Qwen2.5-7B-Instruct", messages=messages, is_thinking=False
        )
        print(f"Local Qwen Result: {result}")
        print(f"Local Qwen Thought: {thought}")
    except Exception as e:
        print(f"Local vLLM error: {e}")
        print("💡 To use local backend, start vLLM server:")
        print(
            "   python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-32B-Instruct --port 8000"
        )


if __name__ == "__main__":
    main()

# PYTHONPATH=src python api_call.py
