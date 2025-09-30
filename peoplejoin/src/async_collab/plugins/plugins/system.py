from typing import Annotated

from async_collab.agent.agent_config import AgentConfig
from async_collab.core.message import Message
from async_collab.plugins.plugin import Plugin, ReturnType
from async_collab.tenant.tenant import Tenant

SYSTEM_PLUGIN_NAME = "System"


class SystemPlugin(Plugin):
    plugin_id: str = "system"
    plugin_name: str = SYSTEM_PLUGIN_NAME
    dependencies = ()

    def finish(self) -> Annotated[None, ReturnType("None")]:
        """
        Terminate the current turn
        """
        pass  # noop

    def __init__(self, tenant: Tenant, primary_user_id: str):
        self.plugin_id = "system"  # identifier to determine active plugin
        self.plugin_name = SYSTEM_PLUGIN_NAME  # name used in annotations
        self.dependencies = ()
        plugin_impls = {"finish": self.finish}
        super().__init__(tenant, primary_user_id, plugin_impls)

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
        return SystemPlugin(tenant, primary_user_id)
