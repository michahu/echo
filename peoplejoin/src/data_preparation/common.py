import json
from dataclasses import dataclass, field

from src.async_collab.core.document import Document
from src.async_collab.core.person import Person


@dataclass
class AsyncCollabTenantData:
    tenant_id: str
    primary_user: Person
    users: list[Person]
    user_id_to_documents: dict[str, list[Document]]
    user_id_to_descriptions: dict[str, str]
    user_id_to_table_names: dict[str, list[str]] = field(default_factory=dict)
    user_id_to_descriptions_templated: dict[str, str] = field(default_factory=dict)
    user_id_to_descriptions_lowprecision: dict[str, str] = field(default_factory=dict)
    user_id_to_descriptions_lowprecision_templated: dict[str, str] = field(
        default_factory=dict
    )

    @staticmethod
    def load_from_file(file_path: str) -> "AsyncCollabTenantData":
        with open(file_path) as f:
            json_content = json.load(f)
        return AsyncCollabTenantData(
            tenant_id=json_content["tenant_id"],
            primary_user=Person.from_dict(json_content["primary_user"]),
            users=[Person.from_dict(user) for user in json_content["users"]],
            user_id_to_documents={
                user_id: [Document.from_dict(doc) for doc in docs]
                for user_id, docs in json_content["user_id_to_documents"].items()
            },
            user_id_to_descriptions=json_content["user_id_to_descriptions"],
            user_id_to_table_names=json_content.get("user_id_to_table_names", {}),
            user_id_to_descriptions_lowprecision=json_content.get(
                "user_id_to_descriptions_lowprecision", {}
            ),
            user_id_to_descriptions_templated=json_content.get(
                "user_id_to_descriptions_templated", {}
            ),
            user_id_to_descriptions_lowprecision_templated=json_content.get(
                "user_id_to_descriptions_lowprecision_templated", {}
            ),
        )

    def get_documents_for_a_user(self, user_id: str) -> list[Document]:
        return self.user_id_to_documents[user_id]

    def get_all_documents(self) -> list[Document]:
        return [doc for docs in self.user_id_to_documents.values() for doc in docs]


@dataclass
class DatumAttributes:
    datum_requires_accessing_subtable_division: bool = False
    datum_requires_redirection_handling: bool = False
    datum_is_unanswerable: bool = False
