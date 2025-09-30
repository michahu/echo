from dataclasses import dataclass

from src.async_collab.core.person import Person
from src.data_preparation.common import AsyncCollabTenantData

VERSION = "v1"

data_dir = "data/peoplejoin-doc-creation"
multinews_version = f"multinews_{VERSION}"
tenant_data_version = f"{multinews_version}__tenant_data_{VERSION}"


def tenant_data_path_generator(tenant_id: str) -> str:
    split_and_group = tenant_id.split("/")[1]
    return f"{data_dir}/tenants/{split_and_group}.json"


_PRIMARY_USER_DICT = {
    "user_id": "alice",
    "full_name": "Alice Anastasiou",
    "email": "alice@company.com",
}

PRIMARY_USER = Person.from_dict(_PRIMARY_USER_DICT)

_OTHER_USERS_DICT = [
    {
        "user_id": "bhushan",
        "full_name": "Bhushan Magar",
        "email": "bhushan@company.com",
    },
    {"user_id": "cassie", "full_name": "Cassie Hicks", "email": "cassie@company.com"},
    {"user_id": "hannah", "full_name": "Hannah Jarvis", "email": "hannah@company.com"},
    {"user_id": "dewei", "full_name": "Dewei Peng", "email": "dewei@company.com"},
    {"user_id": "eden", "full_name": "Eden Berhe", "email": "eden@company.com"},
    {"user_id": "parker", "full_name": "Parker McLean", "email": "parker@company.com"},
    {
        "user_id": "farshid",
        "full_name": "Farshid Kamangar",
        "email": "farshid@company.com",
    },
]

OTHER_USERS = [Person.from_dict(u) for u in _OTHER_USERS_DICT]


@dataclass
class AsyncCollabMultiNews:
    datum_id: str
    tenant_id: str
    summary_prompt: str
    gold_summary: str
    gold_document_ids: list[str]

    @staticmethod
    def from_dict(json_content: dict) -> "AsyncCollabMultiNews":
        # TODO(jda): add gold_people to evaluate whether the right set of users was contacted
        return AsyncCollabMultiNews(
            datum_id=json_content["datum_id"],
            tenant_id=json_content["tenant_id"],
            summary_prompt=json_content["summary_prompt"],
            gold_summary=json_content["gold_summary"],
            gold_document_ids=json_content["gold_document_ids"],
        )

    def get_reference_people_excluding_primary(
        self, tenant: AsyncCollabTenantData
    ) -> set[str]:
        if tenant:
            assert tenant.tenant_id == self.tenant_id

        return {
            user_id
            for (user_id, docs) in tenant.user_id_to_documents.items()
            if (
                any(d.url in self.gold_document_ids for d in docs)
                and user_id != PRIMARY_USER.person_id
            )
        }
