from dataclasses import dataclass, field
from datetime import datetime
from typing import Union

from src.async_collab.core.bot import Bot
from src.async_collab.core.person import Person
from src.logging_config import general_logger


def _get_name(p: Person | Bot):
    return p.full_name


@dataclass(frozen=True, eq=True, unsafe_hash=True, kw_only=True)
class Message:
    """
    A message.

    sender: Union[Person, Bot]  # The person who sent the message.
    receiver: Person  # The person who received the message.
    content: str  # The content of the message.
    created_on: datetime  # The date and time the message was sent.
    """

    # using `Union[Person, Bot]` instead of `Person | Bot` for jsons' sake
    sender: Union[Person, Bot]  # noqa: UP007
    recipient: Union[Person, Bot]  # noqa: UP007
    content: str
    message_type: str
    # ignoring the ruff error for Optional because jsons has trouble deserializing NoneType
    # because datetime.utcnow doesn't include tzinfo(?!)
    created_on: datetime = field(default_factory=lambda: datetime.now())

    @property
    def is_content_type(self) -> bool:
        return self.message_type in ["chat"]

    @property
    def as_prompt(self) -> str:
        return f"{_get_name(self.sender)} says: {self.content.strip()}"

    @property
    def as_prompt_with_recipient(self) -> str:
        return f"{_get_name(self.sender)}-to-{_get_name(self.recipient)}: {self.content.strip()}"


@dataclass(frozen=True, eq=True, unsafe_hash=True)
class ChatMessage(Message):
    """
    A chat message.
    """

    message_type: str = "chat"


@dataclass(frozen=True, eq=True, unsafe_hash=True)
class SessionCompleted(Message):
    """
    A session completed message.
    """

    message_type: str = "session_completed"


@dataclass(frozen=True, eq=True, unsafe_hash=True)
class AgentConversation:
    """
    A conversation between one person (`user`, or A) and their bot.

    user: Person  # The human participant in the conversation (A).
    messages: list[Message]  # The messages in the conversation.
    """

    user: Person  # The human participant in the conversation.
    messages: list[Message]  # The messages in the conversation.
