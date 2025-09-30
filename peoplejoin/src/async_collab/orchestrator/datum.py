import json
from dataclasses import dataclass, field
from typing import Any

import jsons

from src.async_collab.core.bot import Bot
from src.async_collab.core.message import Message
from src.async_collab.core.person import Person
from src.logging_config import general_logger

"""
Datum for async collaboration
Opting to use dataclasses for the following reasons (as opposed to a json):
- It will be easy to load data directly for consumption in train/eval
- It will easy to enforce consistency with core datatypes.
- Current design around 'MessageTriggeredActions' has nice parallels to `process_message_from_main_user` and `process_message_from_other_user` in agent, making it easy to 'replay' for evaluation.
"""


@dataclass
class ActionObservation:
    statement: str
    execution_result: str | None = None
    messages: tuple[Message, ...] = ()
    required_plugins: tuple[str, ...] = ()


@dataclass
class EventTriggeredActions:
    """an event such as a user message that triggers a reaction from bot and the resulting bot actions"""

    trigger_event: Message
    bot_actions: list[ActionObservation]


@dataclass
class AsyncCollabDatumMetadata:
    tenant_id: str  # config specifying the tenant in the scenario
    description: str = ""  # a description of the scenario
    description_assertions: list[str] | None = field(
        default_factory=list
    )  # assertions about the scenario; useful for evaluation through checklist
    description_reference_response: str | None = (
        None  # the expected response from the bot; useful for evaluation
    )
    description_reference_people: list[str] | None = field(
        default_factory=list
    )  # user ids of people contacted in the gold standard

    @staticmethod
    def from_dict(dct: dict) -> "AsyncCollabDatumMetadata":
        """
        Load an AsyncCollabDatumMetadata from a json string
        """
        # make sure that jsons.exceptions.DeserializationError: No deserializer for type "types.list" error is not raised
        general_logger.info(f"[AsyncCollabDatumMetadata] from_dict: dct: {dct}")
        dct["description_assertions"] = dct.get("description_assertions", [])
        if "description_reference_people" in dct:
            if dct["description_reference_people"] is None:
                dct["description_reference_people"] = []
            else:
                dct["description_reference_people"] = json.loads(
                    dct["description_reference_people"]
                )
        else:
            dct["description_reference_people"] = []
        if "description_reference_response" not in dct:
            dct["description_reference_response"] = None
        general_logger.info(f"[AsyncCollabDatumMetadata] from_dict: setting dct: {dct}")
        return AsyncCollabDatumMetadata(**dct)

    @staticmethod
    def from_processed_dict(dct: dict) -> "AsyncCollabDatumMetadata":
        """
        Load an AsyncCollabDatumMetadata from a processed dict
        """
        general_logger.info(f"[AsyncCollabDatumMetadata] from_dict: dct: {dct}")
        dct["description_assertions"] = dct.get("description_assertions", [])
        return AsyncCollabDatumMetadata(
            tenant_id=dct["tenant_id"],
            description=dct["description"],
            description_assertions=dct.get("description_assertions", []),
            description_reference_response=dct.get(
                "description_reference_response", None
            ),
            description_reference_people=dct.get("description_reference_people", []),
        )


@dataclass
class AsyncCollabDatum:
    """a datum for async collaboration"""

    datum_id: str  # a unique identifier for the conversation
    tenant_id: str  # config specifying the tenant in the scenario
    primary: Person  # the primary user in the scenario
    bot: Bot  # the bot in the scenario; TODO: can add a post init check to check that the bot owner is the primary user
    flow: list[
        EventTriggeredActions
    ]  # the flow of messages and bot actions in the example
    initial_message: Message | None  # the initial welcome message from the bot
    metadata: AsyncCollabDatumMetadata  # metadata about the scenario

    @classmethod
    def from_json(cls, dct: dict) -> "AsyncCollabDatum":
        """
        Load an AsyncCollabDatum from a json string
        """
        return jsons.load(dct, cls=AsyncCollabDatum)

    @property
    def all_message_history(self) -> list[Message]:
        """
        Get the message history of the async collab datum
        """
        messages = []
        # add initial_message if present
        if self.initial_message is not None:
            messages.append(self.initial_message)
        for event_trigger_actions in self.flow:
            messages.append(event_trigger_actions.trigger_event)
            for action in event_trigger_actions.bot_actions:
                messages.extend(action.messages)
        # sort by created_on
        messages.sort(key=lambda x: x.created_on)
        return messages

    @property
    def all_message_and_action_history(self) -> list[str]:
        """
        Get the message and action history of the async collab datum
        """
        history = []
        # add initial_message if present
        if self.initial_message is not None:
            history.append(self.initial_message.as_prompt)
        for event_trigger_actions in self.flow:
            history.append(event_trigger_actions.trigger_event.as_prompt)
            for action in event_trigger_actions.bot_actions:
                history.append(action.statement)
                for message in action.messages:
                    history.append(message.as_prompt)
        return history


@dataclass
class AsyncCollabOutputDatum:
    """
    An output datum for async collaboration
    """

    datum_id: str
    primary: Person
    metadata: AsyncCollabDatumMetadata
    initial_message: Message | None
    events: list[Message] = field(default_factory=list)
    sent_messages: list[Message] = field(default_factory=list)
    repl: list[str] = field(default_factory=list)
    repl_and_messages: list[str | Message] = field(default_factory=list)

    @property
    def all_message_and_action_history(self) -> list[str]:
        """
        Get the message and action history of the async collab datum
        """
        history = []
        for repl_or_message in self.repl_and_messages:
            if isinstance(repl_or_message, str):
                history.append(repl_or_message)
            else:
                history.append(repl_or_message.as_prompt_with_recipient)
        return history

    @property
    def all_message_history(self) -> list[Message]:
        """
        Get the message history of the async collab datum
        """
        all_messages = self.sent_messages + self.events
        # initial message is included in sent_messages
        all_message_sorted = sorted(all_messages, key=lambda x: x.created_on)
        return all_message_sorted

    @property
    def content_messages_history(self) -> list[Message]:
        """
        Get the message history of the async collab datum -- add message only if if m.is_content_type
        """
        return [m for m in self.all_message_history if m.is_content_type]

    @staticmethod
    def from_dict(dct: dict[str, Any]) -> "AsyncCollabOutputDatum":
        """
        Load an AsyncCollabOutputDatum from a json string
        """
        # return jsons.load(dct, cls=AsyncCollabOutputDatum)
        # jsons library does not handle `| ` type hinting
        return AsyncCollabOutputDatum(
            datum_id=dct["datum_id"],
            primary=Person.from_dict(dct["primary"]),
            metadata=AsyncCollabDatumMetadata.from_processed_dict(dct["metadata"]),
            initial_message=jsons.load(dct["initial_message"], cls=Message)
            if dct.get("initial_message", None)
            else None,
            events=[jsons.load(m, cls=Message) for m in dct["events"]],
            sent_messages=[jsons.load(m, cls=Message) for m in dct["sent_messages"]],
            repl=dct["repl"],
            repl_and_messages=[
                jsons.load(m, cls=Message) if isinstance(m, dict) else m
                for m in dct["repl_and_messages"]
            ],
        )
