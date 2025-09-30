import os
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from src.async_collab.core.bot import Bot
from src.async_collab.core.message import Message
from src.async_collab.core.person import Person
from src.async_collab.llm.llm_client import LLMClient
from src.async_collab.llm.llm_client_service import get_llm_client
from src.async_collab.orchestrator.datum import AsyncCollabOutputDatum
from src.logging_config import evaluation_logger


@dataclass
class AsyncCollabDatumMessages:
    """
    A utulity class to extract messages from a given AsyncCollabDatum
    """

    all_messages: list[Message]
    messages_from_primary_to_bot: list[Message]
    messages_from_bot_to_primary: list[Message]
    messages_from_secondary_to_bot: list[Message]
    messages_from_bot_to_secondary: list[Message]
    messages_from_bot_to_users: list[Message]
    set_of_people_contacted_excluding_primary: set[str]

    @staticmethod
    def messages_using_conditions(
        message_history: list[Message],
        from_condition: Callable[[Person | Bot], bool] | None,
        to_condition: Callable[[Person | Bot], bool] | None,
    ) -> list[Message]:
        return [
            m
            for m in message_history
            if (from_condition is None or from_condition(m.sender))
            and (to_condition is None or to_condition(m.recipient))
        ]

    @classmethod
    def from_datum(cls, datum: AsyncCollabOutputDatum) -> "AsyncCollabDatumMessages":
        message_history = datum.content_messages_history
        messages_from_primary_to_bot = cls.messages_using_conditions(
            message_history=message_history,
            from_condition=lambda x: x == datum.primary,
            to_condition=None,
        )
        messages_from_bot_to_primary = cls.messages_using_conditions(
            message_history=message_history,
            from_condition=None,
            to_condition=lambda x: x == datum.primary,
        )
        messages_from_secondary_to_bot = cls.messages_using_conditions(
            message_history=message_history,
            from_condition=lambda x: x != datum.primary and (not isinstance(x, Bot)),
            to_condition=None,
        )
        messages_from_bot_to_secondary = cls.messages_using_conditions(
            message_history=message_history,
            from_condition=None,
            to_condition=lambda x: x != datum.primary and (not isinstance(x, Bot)),
        )
        messages_from_bot_to_users = cls.messages_using_conditions(
            message_history=message_history,
            from_condition=lambda x: isinstance(x, Bot),
            to_condition=lambda x: (not isinstance(x, Bot)),
        )
        set_of_people_contacted_excluding_primary = {
            m.recipient.person_id
            for m in messages_from_bot_to_secondary  # use messages_from_bot_to_secondary to remove
            if isinstance(
                m.recipient, Person
            )  # ercipients in messages_from_bot_to_secondary are all of type Person but adding this check
        }
        return cls(
            messages_from_primary_to_bot=messages_from_primary_to_bot,
            messages_from_bot_to_primary=messages_from_bot_to_primary,
            messages_from_secondary_to_bot=messages_from_secondary_to_bot,
            messages_from_bot_to_secondary=messages_from_bot_to_secondary,
            all_messages=message_history,
            messages_from_bot_to_users=messages_from_bot_to_users,
            set_of_people_contacted_excluding_primary=set_of_people_contacted_excluding_primary,
        )


class AsyncCollabMetric:
    metric_name: str
    count: int
    predictions: list[AsyncCollabOutputDatum]
    skipped_count: int

    def __init__(self, metric_name: str) -> None:
        self.metric_name = metric_name
        self.reset()

    @abstractmethod
    def __call__(
        self,
        prediction: AsyncCollabOutputDatum,
        prediction_messages: AsyncCollabDatumMessages | None = None,
    ):
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> dict[str, float]:
        raise NotImplementedError

    def reset(self):
        self.count = 0
        self.skipped_count = 0
        self.predictions = []

    def get_raw_scores(self) -> dict[str, float]:
        return {}


llm_client: LLMClient | None = None


def get_llm_response(prompt: str, end_tok: str = "\n", default_llm = 'gpt-4o-11-20') -> str:
    global llm_client
    if llm_client is None:
        llm_client = get_llm_client(default_llm)
    evaluation_logger.info(f"Using following end token: {end_tok}")
    response = llm_client.get_response_str(prompt)
    evaluation_logger.info(f"LLM response: {response}")
    if response is None:
        evaluation_logger.error("LLM response is None")
        return ""
    return response


class AsyncCollabLLMMetric(AsyncCollabMetric):
    scores: list[float]
    scores_raw: dict[str, float]
    end_token: str
    llm_metric_cache: dict[str, str]

    def __init__(
        self,
        metric_name: str,
        end_token: str = "```",
        cache_file_name: str | None = None,
    ) -> None:
        super().__init__(metric_name)
        self.end_token = end_token
        self.reset()
        self.llm_metric_cache = {}
        self.llm_metric_cache_writer = None
        # showl not be none and should exist
        if cache_file_name is not None:
            if os.path.exists(cache_file_name):
                cache_data = open(cache_file_name).readlines()
                for line in cache_data:
                    key, value = line.strip().split("\t")
                    self.llm_metric_cache[key] = value
            self.llm_metric_cache_writer = open(cache_file_name, "a")

    def __call__(
        self,
        prediction: AsyncCollabOutputDatum,
        prediction_messages: AsyncCollabDatumMessages | None = None,
        default_llm: str = 'gpt-4o-11-20',
    ):
        prompts = self._construct_prompts(prediction)
        if len(prompts) == 0:
            self.skipped_count += 1
            return
        aggregate_score = []
        for prompt in prompts:
            if prediction.datum_id in self.llm_metric_cache:
                evaluation_logger.info(
                    f"Metric: {self.metric_name} ; \nPrompt: {prompt}"
                )
                response = self.llm_metric_cache[prediction.datum_id]
                response = response.replace("\\n", "\n")
                evaluation_logger.info(f"Using cache\n REsponse: {response}")
            else:
                response = self._call_llm(prompt, default_llm=default_llm) 
                response = response.strip().replace("\n", "\\n")
                self.llm_metric_cache[prediction.datum_id] = response
                # assert self.llm_metric_cache_writer is not None
                # self.llm_metric_cache_writer.write(
                #     f"{prediction.datum_id}\t{response}\n"
                # )
            score = self._extract_response(response)
            aggregate_score.append(score)
        score = sum(aggregate_score) / len(prompts)
        self.scores.append(score)
        scores_raw_key = prediction.datum_id
        self.scores_raw[scores_raw_key] = score
        self.count += 1

    @abstractmethod
    def _construct_prompts(self, prediction: AsyncCollabOutputDatum) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def _extract_response(self, response: str) -> float:
        raise NotImplementedError

    def _call_llm(self, prompt: str, default_llm='gpt-4o-11-20') -> str:
        evaluation_logger.info(f"Metric: {self.metric_name} ; \nPrompt: {prompt}")
        response = get_llm_response(prompt, end_tok=self.end_token, default_llm=default_llm)
        evaluation_logger.info(f"Metric: {self.metric_name} ; \nResponse: {response}")
        return response

    def reset(self):
        evaluation_logger.info(f"Resetting metric: {self.metric_name}")
        self.scores = []
        self.scores_raw = {}
        super().reset()

    def compute(self) -> dict[str, float]:
        evaluation_logger.info(f"Computing metric: {self.metric_name}")
        return {
            f"{self.metric_name}_count": self.count,
            f"{self.metric_name}_skipped_count": self.skipped_count,
            f"{self.metric_name}": round(sum(self.scores) / self.count, 2)
            if self.count > 0
            else 0,
        }

    def get_raw_scores(self) -> dict[str, float]:
        return self.scores_raw
