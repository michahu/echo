from async_collab.plugins.plugins.cot_plugin import SimpleReasoningPlugin
from async_collab.plugins.plugins.enterprise_plugin import EnterprisePlugin
from async_collab.plugins.plugins.enterprise_search_plugin import EnterpriseSearchPlugin
from async_collab.plugins.plugins.system import SystemPlugin

plugins_by_id = {
    "cot": SimpleReasoningPlugin,
    "system": SystemPlugin,
    "enterprise_search": EnterpriseSearchPlugin,
    "enterprise": EnterprisePlugin,
}
