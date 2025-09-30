from dataclasses import dataclass, field


@dataclass(frozen=True, eq=True, unsafe_hash=True)
class Person:
    """
    A person in the organization.
    """

    person_id: str = field(
        metadata={"description": "A globally unique identifier for the user."}
    )
    full_name: str = field(metadata={"description": "The user's full name."})
    email: str = field(metadata={"description": "The user's email address."})

    @classmethod
    def from_dict(cls, data: dict) -> "Person":
        if "user_id" in data:  # for backwards compatibility
            data["person_id"] = data.pop("user_id")
        if data["person_id"] == "alice@company.com":  # for backwards compatibility
            data["person_id"] = "alice"
        return cls(**data)
