import requests
from typing import List, Dict, Any


class VLLMClient:
    """Client for vLLM server running locally"""
    
    def __init__(self, base_url: str = "http://localhost:8000", model_name: str = "Qwen/Qwen2.5-32B-Instruct"):
        self.base_url = base_url
        self.model_name = model_name
        self.chat_completions_url = f"{base_url}/v1/chat/completions"
        
    def send_request(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Send chat completion request to vLLM server"""
        
        # Default parameters
        request_data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 800),
            "top_p": kwargs.get("top_p", 0.95),
            "stream": False
        }
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in request_data:
                request_data[key] = value
        
        try:
            response = requests.post(
                self.chat_completions_url,
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"vLLM request failed with status {response.status_code}: {response.text}")
            
            return response.json()
            
        except requests.exceptions.ConnectionError:
            raise Exception(f"Could not connect to vLLM server at {self.base_url}. Make sure the server is running.")
        except requests.exceptions.Timeout:
            raise Exception("Request to vLLM server timed out")
        except Exception as e:
            raise Exception(f"vLLM request failed: {str(e)}")
    
    def get_response_str(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Get string response from vLLM server"""
        response = self.send_request(messages, **kwargs)
        return response["choices"][0]["message"]["content"].strip()
    
    def is_available(self) -> bool:
        """Check if vLLM server is available"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False


def get_vllm_client(base_url: str = "http://localhost:8000", model_name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> VLLMClient:
    """Get vLLM client instance"""
    return VLLMClient(base_url=base_url, model_name=model_name)


if __name__ == "__main__":
    # Test the vLLM client
    print("Testing vLLM client...")

    import argparse
    parser = argparse.ArgumentParser(description="Test vLLM client")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-32B-Instruct", help="Model name to use")
    parser.add_argument("--port", type=int, default=8000, help="Port where vLLM server is running")
    args = parser.parse_args()

    client = get_vllm_client(base_url=f"http://localhost:{args.port}", model_name=args.model)
    
    # Check if server is available
    if not client.is_available():
        print("❌ vLLM server is not available. Start it with:")
        print("   python -m vllm.entrypoints.openai.api_server --model <model_name> --port 8000")
        exit(1)
    
    print("✅ vLLM server is available")
    
    # Test basic chat completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    try:
        response = client.send_request(messages)
        print(f"Full response: {response}")
        
        content = response["choices"][0]["message"]["content"].strip()
        print(f"Response content: {content}")
        
        # Test with get_response_str
        simple_response = client.get_response_str(messages)
        print(f"Simple response: {simple_response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test with different parameters
    print("\nTesting with different parameters...")
    try:
        response = client.get_response_str(
            messages=[
                {"role": "system", "content": "You are a creative assistant."},
                {"role": "user", "content": "Write a short poem about programming."}
            ],
            temperature=0.7,
            max_tokens=200
        )
        print(f"Creative response: {response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


# Example usage:
# 
# To start vLLM server:
# python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B-Instruct --port 8000 --dtype=half
#


# python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --port 8000 --dtype=half --tensor-parallel-size=2
# python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-14B-Instruct --port 8000 --dtype=half --tensor-parallel-size=2

# Then run this script:
# python llm_local.py --model Qwen/Qwen2.5-0.5B-Instruct
# python llm_local.py --model Qwen/Qwen2.5-7B-Instruct
