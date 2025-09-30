from textwrap import dedent

import jsons

from data_preparation.spider.spider_common import (
    AsyncCollabSpider,
    AsyncCollabTenantData,
)

# test loading of asynccollabdatum


def test_loading_of_asynccollabdatum():
    with open("tests/test_data/async_collab_spider.json") as f:
        data = AsyncCollabSpider.from_dict(jsons.loads(f.read()))
        # jsons.loads(f.read(), AsyncCollabSpider)
    assert data.datum_id == "concert_singer_0"
    assert (
        data.question
        == "What is the name and capacity for the stadium with highest average attendance?"
    )
    assert data.db_id == "concert_singer"
    assert data.table_names == ["stadium", "singer", "concert", "singer in concert"]
    assert (
        data.gold_sql_query
        == "SELECT name ,  capacity FROM stadium ORDER BY average DESC LIMIT 1"
    )
    assert data.execution_result == '[["Stark\'s Park", 10104]]'
    assert data.tenant_id == "tenant_data_v2/concert_singer"

    # also test get_reference_people
    tenant = get_test_tenant_data("v2")
    reference_people = set(data.get_reference_people_excluding_primary(tenant))
    assert reference_people == set({}), f"reference_people = {reference_people}"
    reference_people = set(data.get_reference_people(tenant))
    assert reference_people == {"alice"}


# test loading of tenant


def get_test_tenant_data(version="v1"):
    if version == "v1":
        async_collab_tenant_data = AsyncCollabTenantData.load_from_file(
            "tests/test_data/concert_singer_spider_tenant.json"
        )
        return async_collab_tenant_data
    elif version == "v2":
        async_collab_tenant_data = AsyncCollabTenantData.load_from_file(
            "tests/test_data/concert_singer_spider_tenant_v2.json"
        )
        return async_collab_tenant_data
    else:
        raise ValueError(f"Invalid version {version}")


def test_loading_of_tenant():
    async_collab_tenant_data = get_test_tenant_data("v1")
    assert async_collab_tenant_data.tenant_id == "tenant_data_v1/concert_singer"
    assert async_collab_tenant_data.primary_user.person_id == "alice"
    assert async_collab_tenant_data.primary_user.full_name == "Alice Anastasiou"
    assert async_collab_tenant_data.primary_user.email == "alice@company.com"
    assert len(async_collab_tenant_data.users) == 4
    assert async_collab_tenant_data.users[0].person_id == "alice"
    assert async_collab_tenant_data.users[1].person_id == "lina"
    assert async_collab_tenant_data.users[2].person_id == "bhushan"
    assert async_collab_tenant_data.users[3].person_id == "gorosti"
    assert (
        async_collab_tenant_data.user_id_to_documents["alice"][0].url
        == "concert_singer/stadium"
    )
    assert async_collab_tenant_data.user_id_to_documents["alice"][0].title == "stadium"
    ref_content = dedent(
        """\
        Document:
        Collection name: concert singer
        Title: stadium
        Record 1: {'stadium id': 1, 'location': 'Raith Rovers', 'name': "Stark's Park", 'capacity': 10104, 'highest': 4812, 'lowest': 1294, 'average': 2106}
        Record 2: {'stadium id': 2, 'location': 'Ayr United', 'name': 'Somerset Park', 'capacity': 11998, 'highest': 2363, 'lowest': 1057, 'average': 1477}
        Record 3: {'stadium id': 3, 'location': 'East Fife', 'name': 'Bayview Stadium', 'capacity': 2000, 'highest': 1980, 'lowest': 533, 'average': 864}
        ... and 6 more records

        """
    )
    gen_content = async_collab_tenant_data.user_id_to_documents["alice"][0].get_content(
        trim_content=True
    )
    print("gen_content = ", gen_content)
    assert gen_content == ref_content
    assert (
        async_collab_tenant_data.user_id_to_descriptions["alice"]
        == "User might have information about `stadium` topic with respect to following attributes: ['stadium id', 'location', 'name', 'capacity', 'highest', 'lowest', 'average']"
    )


def test_loading_of_tenant_multinews():
    tenant_data = AsyncCollabTenantData.load_from_file(
        "tests/test_data/multinews_tenant.json"
    )

    assert tenant_data.tenant_id == "multinews_v1__tenant_data_v1/val_0"
    assert tenant_data.primary_user.person_id == "alice"
    assert len(tenant_data.users) == 6
    assert tenant_data.users[1].person_id == "cassie"
    assert tenant_data.user_id_to_documents["cassie"][0].url == "file://val/0/0/3"

    ref_content = dedent(
        """\
        Document:
        Title: val/0/0/3
        Content: a charity shop is urging people to stop donating the da vinci code after becoming overwhelmed with copies .     the oxfam shop in swansea has been receiving an average of one copy of the dan brown novel a week for months , leaving them with little room for any other books .     staff who are struggling to sell copies of the book have put a note up in the store saying they would rather donors hand in their vinyl instead ."""
    )
    gen_content = tenant_data.user_id_to_documents["cassie"][0].get_content(
        trim_content=True
    )
    assert gen_content == ref_content


# python -m pytest tests/test_data_loading.py::test_loading_of_tenant -vv
# python -m pytest tests/test_data_loading.py::test_loading_of_asynccollabdatum -vv -s
# python -m pytest tests/test_data_loading.py::test_loading_of_tenant_multinews -vv
