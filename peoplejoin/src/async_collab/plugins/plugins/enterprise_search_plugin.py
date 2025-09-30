import json
from typing import Annotated

from async_collab.agent.agent_config import AgentConfig
from async_collab.core.message import Message
from async_collab.plugins.plugin import Plugin, ReturnType
from async_collab.tenant.tenant import Tenant
from logging_config import enterprise_search_logger


class EnterpriseSearchPlugin(Plugin):
    plugin_id: str = "enterprise_search"
    plugin_name: str = "EnterpriseSearch"
    dependencies = ()

    def __init__(self, tenant: Tenant, primary_user_id: str):
        self.plugin_id = "enterprise_search"
        self.plugin_name = "EnterpriseSearch"
        self.dependencies = ()
        plugin_impls = {
            "search_documents": self.search_documents,
            "search_relevant_people": self.search_relevant_people,
        }
        super().__init__(tenant, primary_user_id, plugin_impls)

    def search_documents(
        self, query: Annotated[str, "the query to search"]
    ) -> Annotated[tuple[str, ...], ReturnType("tuple[str, ...]")]:
        """
        Returns the list of relevant documents (including document content/records)
        """
        enterprise_search_logger.info(
            "[EnterpriseSearchPlugin] [search_documents] ====================="
        )
        tenant = self.tenant
        primary_user_id = self.primary_user_id
        doc_search = tenant.search_documents_of_person(
            person_id=primary_user_id, query=query
        )
        ret = tuple(repr(doc) for doc in doc_search)
        enterprise_search_logger.info(
            f"[EnterpriseSearchPlugin] [search_documents] num_documents: {len(doc_search)}"
        )
        enterprise_search_logger.info(
            f"[EnterpriseSearchPlugin] [search_documents] ret = {ret}"
        )
        return ret

    def search_relevant_people(
        self, query: Annotated[str, "the query to search"]
    ) -> Annotated[str, ReturnType("str")]:
        """
        Returns the list of relevant person details and any accompanying rationale
        """
        enterprise_search_logger.info(
            "[EnterpriseSearchPlugin] [search_relevant_people] ====================="
        )
        tenant = self.tenant
        primary_user_id = self.primary_user_id
        person_id_to_description = {
            person_id: desc
            for person_id, desc in tenant.person_id_to_description.items()
            if person_id != primary_user_id
        }
        enterprise_search_logger.info(
            f"[EnterpriseSearchPlugin] [search_relevant_people] num_people: {len(person_id_to_description)}"
        )
        enterprise_search_logger.info(
            f"[EnterpriseSearchPlugin] [search_relevant_people] person_id_to_description: {person_id_to_description}"
        )
        return json.dumps(person_id_to_description)

    @staticmethod
    def get_plugin(
        agent_conf: AgentConfig,
        tenant: Tenant,
        message_send_queue: list[Message] | None,
    ) -> "Plugin":
        """
        # todo
        """
        primary_user_id = agent_conf.main_user_id
        active_plugins = agent_conf.plugin_ids
        for p in Plugin.dependencies:
            assert p in active_plugins, f"Plugin {p} is a dependency of this plugin"
        return EnterpriseSearchPlugin(tenant, primary_user_id)
