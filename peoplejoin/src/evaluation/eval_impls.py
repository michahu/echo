import string

from nltk.tokenize import word_tokenize
from scipy.stats import describe

from src.async_collab.core.person import Person
from src.async_collab.orchestrator.datum import AsyncCollabOutputDatum
from src.evaluation.eval import (
    AsyncCollabDatumMessages,
    AsyncCollabLLMMetric,
    AsyncCollabMetric,
)
from src.logging_config import evaluation_logger
import re

translator = str.maketrans("", "", string.punctuation)


def count_message_tokens(message: str) -> int:
    # remove any puctuations
    message = message.translate(translator)
    # tokenize using nltk
    tokens = word_tokenize(message)
    return len(tokens)


class AsyncCollabMetricEfficiency(AsyncCollabMetric):
    metric_name = "task_efficiency"

    # counts
    message_count_from_primary_to_bot: list[int]
    message_count_from_bot_to_primary: list[int]
    message_count_from_secondary_to_bot: list[int]
    message_count_from_bot_to_secondary: list[int]
    messages_all: list[int]

    # wall clock time
    total_task_time: list[float]

    # message size
    messages_size_from_primary_to_bot: list[int]
    messages_size_from_bot_to_primary: list[int]
    messages_size_from_secondary_to_bot: list[int]
    messages_size_from_bot_to_secondary: list[int]

    # people contacted
    people_contacted: list[int]  # excludes primary user

    def __init__(self) -> None:
        super().__init__(metric_name=self.metric_name)
        self.reset()

    def __call__(
        self,
        prediction: AsyncCollabOutputDatum,
        prediction_messages: AsyncCollabDatumMessages | None = None,
    ):
        if prediction_messages is None:
            prediction_messages = AsyncCollabDatumMessages.from_datum(prediction)
        message_history = prediction_messages.all_messages
        # from_primary_to_bot
        messages_from_primary_to_bot = prediction_messages.messages_from_primary_to_bot
        message_count_from_primary_to_bot = len(messages_from_primary_to_bot)
        self.message_count_from_primary_to_bot.append(
            message_count_from_primary_to_bot
        )
        message_size_from_primary_to_bot = sum(
            [count_message_tokens(m.content) for m in messages_from_primary_to_bot]
        )
        self.messages_size_from_primary_to_bot.append(message_size_from_primary_to_bot)

        # from_bot_to_primary
        message_from_bot_to_primary = prediction_messages.messages_from_bot_to_primary
        message_count_from_bot_to_primary = len(message_from_bot_to_primary)
        self.message_count_from_bot_to_primary.append(message_count_from_bot_to_primary)
        message_size_from_bot_to_primary = sum(
            [count_message_tokens(m.content) for m in message_from_bot_to_primary]
        )
        self.messages_size_from_bot_to_primary.append(message_size_from_bot_to_primary)

        # from_secondary_to_bot
        messages_from_secondary_to_bot = (
            prediction_messages.messages_from_secondary_to_bot
        )
        message_count_from_secondary_to_bot = len(messages_from_secondary_to_bot)
        self.message_count_from_secondary_to_bot.append(
            message_count_from_secondary_to_bot
        )
        message_size_from_secondary_to_bot = sum(
            [count_message_tokens(m.content) for m in messages_from_secondary_to_bot]
        )
        self.messages_size_from_secondary_to_bot.append(
            message_size_from_secondary_to_bot
        )

        # from_bot_to_secondary
        messages_from_bot_to_secondary = (
            prediction_messages.messages_from_bot_to_secondary
        )
        message_count_from_bot_to_secondary = len(messages_from_bot_to_secondary)
        self.message_count_from_bot_to_secondary.append(
            message_count_from_bot_to_secondary
        )
        message_size_from_bot_to_secondary = sum(
            [count_message_tokens(m.content) for m in messages_from_bot_to_secondary]
        )
        self.messages_size_from_bot_to_secondary.append(
            message_size_from_bot_to_secondary
        )

        # people_contacted
        self.people_contacted.append(
            len(prediction_messages.set_of_people_contacted_excluding_primary)
        )

        # all messages
        self.messages_all.append(len(message_history))

        # wall clock time
        # extract the difference in seconds between the message with highest timestamp and the message with lowest timestamp
        # do not assume that the messages are in order of time
        timestamps = [m.created_on for m in message_history]
        task_time = (max(timestamps) - min(timestamps)).total_seconds()
        self.total_task_time.append(task_time)

        # increment count
        self.count += 1

    def reset(self):
        self.message_count_from_primary_to_bot = []
        self.message_count_from_bot_to_primary = []
        self.message_count_from_secondary_to_bot = []
        self.message_count_from_bot_to_secondary = []
        self.messages_all = []
        self.total_task_time = []
        self.messages_size_from_primary_to_bot = []
        self.messages_size_from_bot_to_primary = []
        self.messages_size_from_secondary_to_bot = []
        self.messages_size_from_bot_to_secondary = []
        self.people_contacted = []
        super().reset()

    def compute(self) -> dict:
        # also report count and skipped_count
        return {
            f"{self.metric_name}_message_count_from_primary_to_bot": round(
                sum(self.message_count_from_primary_to_bot) / self.count, 2
            ),
            f"{self.metric_name}_message_count_from_bot_to_primary": round(
                sum(self.message_count_from_bot_to_primary) / self.count, 2
            ),
            f"{self.metric_name}_message_count_from_secondary_to_bot": round(
                sum(self.message_count_from_secondary_to_bot) / self.count, 2
            ),
            f"{self.metric_name}_message_count_from_bot_to_secondary": round(
                sum(self.message_count_from_bot_to_secondary) / self.count, 2
            ),
            f"{self.metric_name}_messages_all": round(
                sum(self.messages_all) / self.count, 2
            ),
            f"{self.metric_name}_total_task_time": round(
                sum(self.total_task_time) / self.count, 2
            ),
            f"{self.metric_name}_messages_size_from_primary_to_bot": round(
                sum(self.messages_size_from_primary_to_bot) / self.count, 2
            ),
            f"{self.metric_name}_messages_size_from_bot_to_primary": round(
                sum(self.messages_size_from_bot_to_primary) / self.count, 2
            ),
            f"{self.metric_name}_messages_size_from_secondary_to_bot": round(
                sum(self.messages_size_from_secondary_to_bot) / self.count, 2
            ),
            f"{self.metric_name}_messages_size_from_bot_to_secondary": round(
                sum(self.messages_size_from_bot_to_secondary) / self.count, 2
            ),
            f"{self.metric_name}_people_contacted": round(
                sum(self.people_contacted) / self.count, 2
            ),
            f"{self.metric_name}_count": self.count,
            f"{self.metric_name}_skipped_count": self.skipped_count,
            f"{self.metric_name}_message_count_from_primary_to_bot_raw": self.message_count_from_primary_to_bot,
            f"{self.metric_name}_message_count_from_bot_to_primary_raw": self.message_count_from_bot_to_primary,
            f"{self.metric_name}_message_count_from_secondary_to_bot_raw": self.message_count_from_secondary_to_bot,
            f"{self.metric_name}_message_count_from_bot_to_secondary_raw": self.message_count_from_bot_to_secondary,
            f"{self.metric_name}_messages_all_raw": self.messages_all,
            f"{self.metric_name}_total_task_time_raw": self.total_task_time,
            f"{self.metric_name}_messages_size_from_primary_to_bot_raw": self.messages_size_from_primary_to_bot,
            f"{self.metric_name}_messages_size_from_bot_to_primary_raw": self.messages_size_from_bot_to_primary,
            f"{self.metric_name}_messages_size_from_secondary_to_bot_raw": self.messages_size_from_secondary_to_bot,
            f"{self.metric_name}_messages_size_from_bot_to_secondary_raw": self.messages_size_from_bot_to_secondary,
            f"{self.metric_name}_people_contacted_raw": self.people_contacted,
        }



class ChecklistMetric(AsyncCollabLLMMetric):
    metric_name: str = "checklist"
    template_path: str
    examples_path: str
    identifier: str

    def __init__(
        self,
        metric_name: str = "checklist",
        identifier: str = "Bot",
        template_path: str = "src/evaluation/prompts/llm_checklist.md",
        examples_path: str = "src/evaluation/prompts/llm_checklist_examples.md",
        cache_file_name: str | None = None,
    ) -> None:
        super().__init__(metric_name=metric_name, cache_file_name=cache_file_name)
        self.template_path = template_path
        self.examples_path = examples_path
        self.identifier = identifier

    def _get_assertions(self, prediction: AsyncCollabOutputDatum) -> list[str] | None:
        assertions = prediction.metadata.description_assertions
        return assertions

    def _construct_prompts(self, prediction: AsyncCollabOutputDatum) -> list[str]:
        template = open(self.template_path).read()
        # template has following variables: {identifier}, {primary_user}, {examples}, {conversation}, {criteria}
        message_history = prediction.content_messages_history
        message_history_formatted = [
            m.as_prompt_with_recipient for m in message_history
        ]
        message_history_formatted_str = "\n".join(message_history_formatted)

        identifier = self.identifier
        primary_user = prediction.primary.full_name
        examples = open(self.examples_path).read() + "\n"

        first_message = message_history[0]
        if len(message_history) > 1:
            # conversation_date should mention the month name in full and the date and day of week
            conversation_date = first_message.created_on.strftime("%B %d, %A") + "\n"
        else:
            conversation_date = "N/A\n"

        assertions = self._get_assertions(prediction)

        if assertions is None:
            return (
                []
            )  # this will be accruing to skipped_count as per AsyncCollabLLMMetric implementation

        return [
            template.format(
                identifier=identifier,
                primary_user=primary_user,
                examples=examples,
                conversation=message_history_formatted_str,
                conversation_date=conversation_date,
                criteria=criteria,
            )
            for criteria in assertions
        ]

    def _extract_response(self, response: str) -> float:
        # extract score
        # first identinfy the line with score. seaech for "score: " to find this line
        response_lines = response.strip().split("\n")
        evaluation_logger.info(f"[ChecklistMetric] response lines: {response_lines}")
        if len(response_lines) == 0:
            evaluation_logger.info("[ChecklistMetric] WARNING: empty response")
            return 0.0  # assuming default of 0 if no score is found
        for line in response_lines:
            if "score: " in line:
                try:
                    score_part = line.split("score: ")[1].strip()
                    # Extract just the first number (float) from the score part
                    match = re.search(r'[-+]?(?:\d*\.\d+|\d+)', score_part)
                    if match:
                        score = float(match.group())
                    else:
                        score = 0.0
                except (IndexError, ValueError):
                    score = 0.0
                # score should be between 0 and 2
                score = max(0.0, min(2.0, score))
                evaluation_logger.info(f"[ChecklistMetric] extracted score: {score}")
                return 50.0 * score  # rescale to be between 0 and 100
        evaluation_logger.info("[ChecklistMetric] WARNING: no score found in response")
        return 0.0


class InfoSeekingReferenceOverlapChecklist(ChecklistMetric):
    metric_name = "reference_overlap_using_llm"

    def __init__(self, cache_file_name: str | None = None) -> None:
        super().__init__(
            metric_name="reference_overlap_using_llm", cache_file_name=cache_file_name
        )

    def _get_assertions(self, prediction: AsyncCollabOutputDatum) -> list[str]:
        description_reference_response = (
            prediction.metadata.description_reference_response
        )
        if description_reference_response is None:
            return []
        return [
            f"{self.identifier} informs the user that the answer to their original request is {description_reference_response}. Exact format in which information is provided can be different. {self.identifier} does not provide any untrue or misleading extra information."
        ]


class InfoSeekingSummaryQualityChecklist(ChecklistMetric):
    metric_name = "summary_quality_using_llm"

    def __init__(self) -> None:
        super().__init__(metric_name="summary_quality_using_llm")

    def _get_assertions(self, prediction: AsyncCollabOutputDatum) -> list[str]:
        description_reference_response = (
            prediction.metadata.description_reference_response
        )
        if description_reference_response is None:
            return []
        return [
            f"{self.identifier} informs the user that the answer to their original request is {description_reference_response}. {self.identifier} does not provide any untrue or misleading extra information."
        ]


class AsyncCollabFairnessMetric(AsyncCollabMetric):
    metric_name = "fairness"
    message_count_to_users: dict[str, float]

    def __init__(self) -> None:
        super().__init__(metric_name=self.metric_name)
        self.reset()

    def reset(self):
        self.message_count_to_users = {}
        super().reset()

    def __call__(
        self,
        prediction: AsyncCollabOutputDatum,
        prediction_messages: AsyncCollabDatumMessages | None = None,
    ):
        """
        Cumulative count of messages to specific users.
        Also compute normalized count.
        """
        if prediction_messages is None:
            prediction_messages = AsyncCollabDatumMessages.from_datum(prediction)
        messages_from_bot_to_secondary = (
            prediction_messages.messages_from_bot_to_secondary
        )
        {message.recipient for message in messages_from_bot_to_secondary}
        for message in messages_from_bot_to_secondary:
            assert isinstance(message.recipient, Person)
            user_id = message.recipient.person_id
            if user_id not in self.message_count_to_users:
                self.message_count_to_users[user_id] = 0.0
            self.message_count_to_users[user_id] += 1
        self.count += 1

    def compute(self) -> dict[str, float]:
        """
        Compute normalized count of messages to each user.
        """
        message_count_to_users_values = list(self.message_count_to_users.values())
        total_messages = sum(message_count_to_users_values)
        if total_messages == 0:
            evaluation_logger.info(
                "[AsyncCollabFairnessMetric] WARNING: total_messages is 0"
            )
            return {
                f"{self.metric_name}_max_message_count_for_any_user": 0,
                f"{self.metric_name}_max_message_normalized_count_for_any_user": 0,
                f"{self.metric_name}_variance_across_users": 0,
                f"{self.metric_name}_count": self.count,
            }
        normalized_count = {
            user_id: count / total_messages
            for user_id, count in self.message_count_to_users.items()
        }
        # report following (wrt secondary users)
        #  maximum count of messages to a single user
        #  max normalized count
        #  how much is the distribution is skewed. report variance as per normalized distribution
        return {
            f"{self.metric_name}_max_message_count_for_any_user": round(
                max(message_count_to_users_values), 2
            ),  # lower is better in terms of fairness
            f"{self.metric_name}_max_message_normalized_count_for_any_user": round(
                max(normalized_count.values()), 2
            ),  # lower is better in terms of fairness
            f"{self.metric_name}_variance_across_users": round(
                describe(list(normalized_count.values())).variance, 4
            ),  # lower is better in terms of fairness. 0 is perfect fairness
            f"{self.metric_name}_count": self.count,
        }

    def get_raw_scores(self) -> dict[str, float]:
        return {
            f"{self.metric_name}_message_count_to_user_{user_id}": float(count)
            for user_id, count in self.message_count_to_users.items()
        }


# description_reference_people
# consider the set of people the system reached out to
# gold reference set is avalaible in the description_reference_people field
# Compute precision and recall of the set of people the system reached out to
class AsyncCollabPeopleReferenceMetric(AsyncCollabMetric):
    metric_name = "people_contacted"
    precision: float
    recall: float

    def __init__(self) -> None:
        super().__init__(metric_name=self.metric_name)
        self.reset()

    def reset(self):
        self.precision = 0.0
        self.recall = 0.0
        super().reset()

    def __call__(
        self,
        prediction: AsyncCollabOutputDatum,
        prediction_messages: AsyncCollabDatumMessages | None = None,
    ):
        # reference
        people_reference = (
            prediction.metadata.description_reference_people
        )  # these are user_ids
        if people_reference is None:
            self.skipped_count += 1
            return
        people_reference = set(people_reference)
        evaluation_logger.info([f"people_reference: {people_reference}"])
        if prediction_messages is None:
            prediction_messages = AsyncCollabDatumMessages.from_datum(prediction)
        messages_from_bot_to_secondary = (
            prediction_messages.messages_from_bot_to_secondary
        )
        people_reference_count = len(people_reference)
        # actual people reached out to
        people_reached_out_to = set()
        for message in messages_from_bot_to_secondary:
            assert isinstance(message.recipient, Person)
            people_reached_out_to.add(message.recipient.person_id)
        evaluation_logger.info([f"people_reached_out_to: {people_reached_out_to}"])
        people_reached_out_to_count = len(people_reached_out_to)
        # compute precision and recall
        true_positives = len(people_reached_out_to.intersection(people_reference))
        if people_reached_out_to_count == 0:
            precision = 0.0
        else:
            precision = true_positives / people_reached_out_to_count
        if people_reference_count == 0:
            recall = 1.0  # if there are no people to reach out to, then recall is cosidered 1
        else:
            recall = true_positives / people_reference_count
        self.precision += precision
        self.recall += recall
        self.count += 1

    def compute(self) -> dict[str, float]:
        return {
            f"{self.metric_name}_precision": round(self.precision / self.count, 2),
            f"{self.metric_name}_recall": round(self.recall / self.count, 2),
            f"{self.metric_name}_count": self.count,
            f"{self.metric_name}_skipped_count": self.skipped_count,
        }
