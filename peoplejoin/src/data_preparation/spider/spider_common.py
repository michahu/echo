import json
from dataclasses import dataclass, field
from typing import Any

import jsons

from src.async_collab.core.person import Person
from src.data_preparation.utils import find_tables_in_sql_query, normalize_table_name
from src.data_preparation.common import AsyncCollabTenantData, DatumAttributes

####


@dataclass
class AsyncCollabSpider:
    datum_id: str
    question: str
    db_id: str
    table_names: list[str]
    gold_sql_query: str
    execution_result: str
    tenant_id: str  # to map to AsyncCollabTenantData
    reference_people: list[str]
    datum_attributes: DatumAttributes | None = field(default_factory=DatumAttributes)

    @staticmethod
    def from_dict(json_content: dict) -> "AsyncCollabSpider":
        return AsyncCollabSpider(
            datum_id=json_content["datum_id"],
            question=json_content["question"],
            db_id=json_content["db_id"],
            table_names=json_content["table_names"],
            gold_sql_query=json_content["gold_sql_query"],
            execution_result=json_content["execution_result"],
            tenant_id=json_content["tenant_id"],
            reference_people=json_content.get("reference_people", []),
            datum_attributes=jsons.load(
                json_content["datum_attributes"], DatumAttributes
            ),
        )

    @classmethod
    def get_reference_people_and_tables_from_query(
        cls, user_id_to_table_names: dict[str, list[str]], gold_query: str
    ) -> dict[str, set[str]]:
        table_names_in_query = [
            normalize_table_name(t) for t in find_tables_in_sql_query(gold_query)
        ]
        reference_people = []
        for table_name in table_names_in_query:
            for user_id, tables in user_id_to_table_names.items():
                tables_normalized = [normalize_table_name(t) for t in tables]
                if table_name in tables_normalized:
                    reference_people.append(user_id)
        return {
            "reference_people": set(reference_people),
            "reference_table_names": set(table_names_in_query),
        }

    def get_reference_people(
        self, tenant: AsyncCollabTenantData | None = None
    ) -> set[str]:
        # designing it in this way for backward compatibility
        if tenant:
            assert tenant.tenant_id == self.tenant_id
        return set(self.reference_people)

    def get_reference_people_excluding_primary(
        self, tenant: AsyncCollabTenantData
    ) -> set[str]:
        # designing it in this way for backward compatibility
        assert tenant.tenant_id == self.tenant_id
        reference_people = set(self.reference_people)
        reference_people.discard(tenant.primary_user.person_id)
        return reference_people

    def extract_table_names_from_query(self) -> list[str]:
        # extract table names from gold_sql_query
        # return a list of table names
        # for now, just extract table names from the query using a regex
        # in future, we can use a more sophisticated method to extract table names
        table_names = find_tables_in_sql_query(self.gold_sql_query)
        return list(table_names)


### Database related


@dataclass
class TableData:
    table_name: str
    rows: list[dict[str, Any]]

    @staticmethod
    def from_json(table_data_dict: dict[str, Any]) -> "TableData":
        table_data = TableData(
            table_name=table_data_dict["table_name"], rows=table_data_dict["rows"]
        )
        return table_data


@dataclass
class DatabaseSchema:
    db_id: str
    table_names_original: list[str]
    table_names: list[str]
    column_names_original: list[tuple[str, str]]
    column_names: list[tuple[str, str]]
    column_types: list[str]
    foreign_keys: list[tuple[int, int]]
    primary_keys: list[int]
    data: list[TableData]

    def __str__(self) -> str:
        schema_str = (
            f"Database ID: {self.db_id}\n"
            f"Original Table Names: {self.table_names_original}\n"
            f"Normalized Table Names: {self.table_names}\n"
            f"Original Column Names: {self.column_names_original}\n"
            f"Normalized Column Names: {self.column_names}\n"
            f"Column Types: {self.column_types}\n"
            f"Foreign Keys: {self.foreign_keys}\n"
            f"Primary Keys: {self.primary_keys}\n"
            f"Data-sample:\n"
        )
        for table_data in self.data:
            schema_str += f"Table: {table_data.table_name}\nRows:\n"
            for i, row in enumerate(table_data.rows):
                schema_str += f"{row}\n"
                if i == 2:
                    schema_str += f"... and {len(table_data.rows)-3} more rows\n"
                    break
        return schema_str

    def get_data_for_table(self, table_name: str) -> TableData:
        for td in self.data:
            if self._normalize_table_name(td.table_name) == self._normalize_table_name(
                table_name
            ):
                return td
        raise ValueError(
            f"Table `{table_name}` not found in schema. \nAll tables: {self.table_names}. \nAll table_names_original: {self.table_names_original}\nTable names in data: {[td.table_name for td in self.data]}"
        )

    def get_column_names_for_table(self, table_name: str) -> list[str]:
        return [
            col[1]
            for col in self.column_names
            if self._normalize_table_name(col[0])
            == self._normalize_table_name(table_name)
        ]

    @staticmethod
    def _normalize_table_name(table_name: str) -> str:
        return table_name.lower().replace(" ", "_")

    @staticmethod
    def from_json(schema_dict: dict[str, Any]) -> "DatabaseSchema":
        table_data_list = [TableData.from_json(td) for td in schema_dict["data"]]
        schema = DatabaseSchema(
            db_id=schema_dict["db_id"],
            table_names_original=schema_dict["table_names_original"],
            table_names=schema_dict["table_names"],
            column_names_original=[
                tuple(col) for col in schema_dict["column_names_original"]  # type: ignore
            ],
            column_names=[tuple(col) for col in schema_dict["column_names"]],  # type: ignore
            column_types=schema_dict["column_types"],
            foreign_keys=[tuple(fk) for fk in schema_dict["foreign_keys"]],  # type: ignore
            primary_keys=schema_dict["primary_keys"],
            data=table_data_list,
        )
        return schema


#### query and annoations related


@dataclass
class From:
    conds: list[Any]
    table_units: list[Any]


@dataclass
class SQL:
    except_clause: Any | None
    from_clause: From
    groupBy: list[Any]
    having: list[Any]
    intersect: Any | None
    limit: Any | None
    orderBy: list[Any]
    select: list[Any]
    union: Any | None
    where: list[Any]


@dataclass
class SpiderEntry:
    db_id: str
    query: str
    query_toks: list[str]
    question: str
    question_toks: list[str]
    sql: SQL
    execution_result: str

    @staticmethod
    def from_dict(entry_dict: dict[str, Any]) -> "SpiderEntry":
        entry = SpiderEntry(
            db_id=entry_dict["db_id"],
            query=entry_dict["query"],
            query_toks=entry_dict["query_toks"],
            question=entry_dict["question"],
            question_toks=entry_dict["question_toks"],
            sql=entry_dict["sql"],
            execution_result=entry_dict["execution_result"],
        )
        return entry


##### file names

# VERSION = 'v1'
VERSION = "v2"

spider_dir = "data/peoplejoin-qa"

dev_data_file = f"{spider_dir}/spider_data/dev.json"
dev_queries_results_file = f"{spider_dir}/processed_data/spider_dev_with_exec.jsonl"
train_data_file = f"{spider_dir}/spider_data/train_spider.json"
train_queries_results_file = f"{spider_dir}/processed_data/spider_train_with_exec.jsonl"

spider_processed_data_path: str = f"{spider_dir}/processed_data/spider_database.jsonl"
ex_tab_file = f"{spider_dir}/spider_data/tables.json"
db_directory_path = f"{spider_dir}/spider_data/database/"
spider_database_file = f"{spider_dir}/processed_data/spider_database.jsonl"


tenant_data_version = f"tenant_data_{VERSION}"
tenant_data_dir = f"{spider_dir}/{tenant_data_version}/"


def tenant_data_path_generator(tenant_id):
    return f"{spider_dir}/{tenant_id}.json"


async_collab_spider_data_dir = (
    f"{spider_dir}/processed_data/async_collab_spider_data_{VERSION}"
)


#### other constants


ALICE_USER_DCT = {
    "user_id": "alice",
    "full_name": "Alice Anastasiou",
    "email": "alice@company.com",
}
ALICE_USER = Person.from_dict(ALICE_USER_DCT)
ALL_OTHER_USERS_DCT = [
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
    {
        "user_id": "gorosti",
        "full_name": "Gorosti Egiagarai",
        "email": "gorosti@company.com",
    },
    {
        "user_id": "harpreet",
        "full_name": "Harpreet Thapar",
        "email": "harpreet@company.com",
    },
    {"user_id": "irena", "full_name": "Irena Jovanovic", "email": "irena@company.com"},
    {"user_id": "juan", "full_name": "Juan Quispe", "email": "juan@company.com"},
    {"user_id": "kerstin", "full_name": "Kerstin Mark", "email": "kerstin@company.com"},
    {"user_id": "lina", "full_name": "Lina Wagner", "email": "lina@company.com"},
]

ALL_OTHER_USERS_DCT_v2 = [
    *ALL_OTHER_USERS_DCT,
    {"user_id": "maname", "full_name": "Maname Mohlare", "email": "maname@company.com"},
    {"user_id": "niks", "full_name": "Niks Dzenis", "email": "niks@company.com"},
    {
        "user_id": "oubunmi",
        "full_name": "Oubunmi Gboyega",
        "email": "oubunmi@company.com",
    },
    {
        "user_id": "ruwaidah",
        "full_name": "Ruwaidah Fakhoury",
        "email": "ruwaidah@company.com",
    },
    {"user_id": "sylvie", "full_name": "Sylvie Rocher", "email": "sylvie@company.com"},
    {"user_id": "tulga", "full_name": "Tulga Bat-Erdene", "email": "tulga@company.com"},
    {
        "user_id": "valarie",
        "full_name": "Valarie Cabral",
        "email": "valarie@company.com",
    },
]

# ALL_OTHER_USERS = [Person.from_dict(user_dct) for user_dct in ALL_OTHER_USERS_DCT]
ALL_OTHER_USERS = [
    Person.from_dict(user_dct) for user_dct in ALL_OTHER_USERS_DCT_v2
]  # number of other users were increased by 7 in v2
ALL_USERS_CNT = len(ALL_OTHER_USERS) + 1


if __name__ == "__main__":
    # tenant_data_file = "src/data_preparation/spider/processed_data/tenant_data_v1/concert_singer.json"
    tenant_data_file = (
        "src/data_preparation/spider/processed_data/tenant_data_v2/concert_singer.json"
    )
    print(" loaded json = ", json.dumps(json.load(open(tenant_data_file)), indent=2))
    async_collab_tenant_data = AsyncCollabTenantData.load_from_file(tenant_data_file)
    print("async_collab_tenant_data = ", async_collab_tenant_data)
    print(type(async_collab_tenant_data.primary_user))
    print(type(async_collab_tenant_data.users))
    print(type(async_collab_tenant_data.user_id_to_documents))
    print(type(async_collab_tenant_data.user_id_to_descriptions))
    print(type(async_collab_tenant_data.user_id_to_table_names))
    print(type(async_collab_tenant_data.user_id_to_documents["alice"]))
    print(type(async_collab_tenant_data.user_id_to_documents["alice"][0]))
    print("----------------")

    dev_file_path = "src/data_preparation/spider/processed_data/async_collab_spider_data_v2/dev.jsonl"
    try:
        with open(dev_file_path) as f:
            data = [json.loads(line) for line in f]
    except (OSError, FileNotFoundError) as e:
        print(f"Error opening file {dev_file_path}: {e}")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {dev_file_path}: {e}")
        exit(1)

    # See 10 examples where datum_requires_accessing_subtable_division is True
    cnt = 0
    for i, d in enumerate(data):
        data_loaded = AsyncCollabSpider.from_dict(d)
        assert data_loaded.datum_attributes is not None
        if data_loaded.datum_attributes.datum_requires_accessing_subtable_division:
            print("i=", i)
            print("data_loaded = ", data_loaded)
            print("----------------")
            cnt += 1
        if cnt >= 10:
            break
    print("=====================================")

    # See 10 examples where datum_requires_redirection_handling is True
    cnt = 0
    for i, d in enumerate(data):
        data_loaded = AsyncCollabSpider.from_dict(d)
        assert data_loaded.datum_attributes is not None
        if data_loaded.datum_attributes.datum_requires_redirection_handling:
            print("i=", i)
            print("data_loaded = ", data_loaded)
            print("----------------")
            cnt += 1
        if cnt >= 10:
            break
    print("=====================================")

    # See 10 examples where datum_is_unanswerable is True
    cnt = 0
    for i, d in enumerate(data):
        data_loaded = AsyncCollabSpider.from_dict(d)
        assert data_loaded.datum_attributes is not None
        if data_loaded.datum_attributes.datum_is_unanswerable:
            print("i=", i)
            print("data_loaded = ", data_loaded)
            print("----------------")
            cnt += 1
        if cnt >= 10:
            break
    print("=====================================")


# python src/data_preparation/spider/spider_common.py
