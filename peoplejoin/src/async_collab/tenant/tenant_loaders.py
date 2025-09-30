import json

import jsons

from src.async_collab.core.document import Document, DocumentCollection
from src.async_collab.core.person import Person
from src.async_collab.tenant.tenant import Tenant
from src.logging_config import general_logger


class TenantLoader:
    tenant_id_to_path: dict[str, str] = {}
    tenant_id_to_loader: dict[str, str] = {}

    @staticmethod
    def from_id(tenant_id: str) -> Tenant:
        if tenant_id not in TenantLoader.tenant_id_to_path:
            raise ValueError(f"Tenant with id {tenant_id} not found.")
        loader = TenantLoader.tenant_id_to_loader.get(tenant_id, "from_dict")
        if loader == "load_peoplejoin_tenant":
            return TenantLoader.load_peoplejoin_tenant(tenant_id)
        elif loader == "from_dict":
            path = TenantLoader.tenant_id_to_path[tenant_id]
            return Tenant.from_dict(jsons.load(path))
        else:
            raise ValueError(f"Invalid loader {loader}")

    @classmethod
    def register_tenant(cls, tenant_id: str, path: str):
        cls.tenant_id_to_path[tenant_id] = path

    @classmethod
    def register_tenant_with_loader(cls, tenant_id: str, path: str, loader_func: str):
        cls.tenant_id_to_path[tenant_id] = path
        cls.tenant_id_to_loader[tenant_id] = loader_func

    @staticmethod
    def load_peoplejoin_tenant(tenant_id: str) -> Tenant:
        # 2024-11-09 23:36:58,478 - general - INFO - [TenantLoader] [load_peoplejoinqa_v2] Loaded tenant data has following keys: dict_keys(['tenant_id', 'primary_user', 'users', 'user_id_to_documents', 'user_id_to_descriptions', 'user_id_to_table_names', 'user_id_to_descriptions_templated', 'user_id_to_descriptions_lowprecision', 'user_id_to_descriptions_lowprecision_templated'])
        pth = TenantLoader.tenant_id_to_path[tenant_id]
        general_logger.info(
            f"[TenantLoader] [load_peoplejoinqa_v2] Loading tenant from path: {pth}"
        )
        data = json.load(open(pth))
        general_logger.info(
            f"[TenantLoader] [load_peoplejoinqa_v2] Loaded tenant data has following keys: {data.keys()}"
        )
        people = tuple(Person.from_dict(user_data) for user_data in data["users"])
        general_logger.info(
            f"[TenantLoader] [load_peoplejoinqa_v2] Creating tenant with people: {people}"
        )
        person_id_to_document_collection = {}
        if data.get("user_id_to_documents") is not None:
            for person in people:
                person_id_to_document_collection[person.person_id] = DocumentCollection(
                    documents=[
                        Document.from_dict(doc)
                        for doc in data["user_id_to_documents"][person.person_id]
                    ]
                )
                general_logger.info(
                    f"[TenantLoader] [load_peoplejoinqa_v2] Created #{len(person_id_to_document_collection[person.person_id].documents)} documents for person {person.person_id}"
                )
            general_logger.info(
                f"[TenantLoader] [load_peoplejoinqa_v2] Created document search index for following users = {person_id_to_document_collection.keys()}"
            )
        person_id_to_calendar = {}
        user_id_to_descriptions_draft = data.get("user_id_to_descriptions", {})
        user_id_to_descriptions = {
            person_id: user_id_to_descriptions_draft.get(person_id, "unknown")
            for person_id in [person.person_id for person in people]
        }  # default to unknown
        general_logger.info(
            f"[TenantLoader] [load_peoplejoinqa_v2] user_id_to_descriptions = {json.dumps(user_id_to_descriptions,indent=2)}"
        )
        general_logger.info(
            f"[TenantLoader] [load_peoplejoinqa_v2] created tenant with tenant_id = {tenant_id}"
        )
        return Tenant(
            tenant_id=tenant_id,
            people=people,
            person_id_to_document_collection=person_id_to_document_collection,
            person_id_to_calendar=person_id_to_calendar,
            person_id_to_description=user_id_to_descriptions,
        )


for tenant in ["battle_death"]:
    # print(f"Registering tenant {tenant}")
    TenantLoader.register_tenant_with_loader(
        f"peoplejoinqa/{tenant}",
        f"data/peoplejoin-qa/tenant_data_v2/{tenant}.json",
        "load_peoplejoin_tenant",
    )

TenantLoader.register_tenant_with_loader(
    "multinews_dummy_tenant",
    "data/peoplejoin-doc-creation/tenants/val_0.json",
    "load_peoplejoin_tenant",
)

# register peoplejoinqa tenants
with open("data/peoplejoin-qa/test.jsonl") as f:
    test_data = [json.loads(line) for line in f]
with open("data/peoplejoin-qa/dev.jsonl") as f:
    test_data += [json.loads(line) for line in f]
tenant_ids = list({data["tenant_id"] for data in test_data})
for tenant_id in tenant_ids:
    # print(f"Registering tenant {tenant_id}")
    TenantLoader.register_tenant_with_loader(
        f"{tenant_id}", f"data/peoplejoin-qa/{tenant_id}.json", "load_peoplejoin_tenant"
    )

# register peoplejoindoccreation tenants
with open("data/peoplejoin-doc-creation/test.scenario.jsonl") as f:
    tenants = [json.loads(line) for line in f]
with open("data/peoplejoin-doc-creation/val.scenario.jsonl") as f:
    tenants += [json.loads(line) for line in f]
tenant_ids = list({data["tenant_id"] for data in tenants})
for tenant_id in tenant_ids:
    _, tenant_filename = tenant_id.split("/")
    TenantLoader.register_tenant_with_loader(
        f"{tenant_id}",
        f"data/peoplejoin-doc-creation/tenants/{tenant_filename}.json",
        "load_peoplejoin_tenant",
    )
