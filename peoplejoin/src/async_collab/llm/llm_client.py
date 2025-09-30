from abc import abstractmethod
from dataclasses import dataclass


class LLMClient:
    model: str

    @abstractmethod
    def get_response_str(
        self,
        user_prompt: str,
        temperature: float = 0,
        max_tokens: int = 800,
        top_p: float = 0.95,
        system_instruction: str = "Complete user request",
        stop: str | None = None,
        model: str = "gpt-4-0125-preview",
    ) -> str | None:
        raise NotImplementedError

    def send_request(self, request, model) -> dict:
        raise NotImplementedError


@dataclass(frozen=True, eq=True, unsafe_hash=True)
class LLMAgentConfig:
    model: str
    simulator_model: str
