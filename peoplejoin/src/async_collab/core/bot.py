from dataclasses import dataclass, field

from src.async_collab.core.person import Person


@dataclass(frozen=True, eq=True, unsafe_hash=True)
class Bot:
    """
    An LLM agent.
    """
    full_name = "Bot"
    owner: Person = field(metadata={"description": "The person who owns the bot."})
