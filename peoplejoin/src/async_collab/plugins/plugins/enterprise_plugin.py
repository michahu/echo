from datetime import datetime
from typing import Annotated

import jsons

from async_collab.agent.agent_config import AgentConfig
from async_collab.core.bot import Bot
from async_collab.core.message import ChatMessage, Message, SessionCompleted
from async_collab.core.person import Person
from async_collab.plugins.plugin import Plugin, ReturnType
from async_collab.tenant.tenant import Tenant


class EnterprisePlugin(Plugin):
    plugin_id: str = "enterprise"
    plugin_name: str = "Enterprise"
    dependencies = ()

    def __init__(
        self, tenant: Tenant, primary_user_id: str, message_send_queue: list[Message]
    ):
        self.plugin_id = "enterprise"
        self.plugin_name = "Enterprise"
        self.dependencies = ()
        plugin_impls = {
            "send_message": self.send_message,
            "resolve_person": self.resolve_person,
            "resolve_primary_user": self.resolve_primary_user,
            "send_session_completed": self.send_session_completed,
        }
        self.message_send_queue = (
            message_send_queue  # add messages to be sent to this queue
        )
        super().__init__(tenant, primary_user_id, plugin_impls)

    def resolve_person(
        self, name: Annotated[str, "The name."]
    ) -> Annotated[str, ReturnType("str")]:
        """
        Find list of persons matching a given name and return a string representation of the first match.
        Suitable for ReAct style orchestrators.
        """
        person_id_to_person_dict = self.tenant.person_id_to_person_dict
        person = person_id_to_person_dict.get(name.lower())
        if person is not None:
            return jsons.dumps(person, strip_privates=True)
        ret: set[Person] = set({})
        # relaxed matching
        test = name.lower()
        for person in person_id_to_person_dict.values():
            lowered_joined = person.full_name.replace(" ", "").strip().lower()
            lowered = person.full_name.strip().lower()
            if (
                test == person.full_name.lower()
                or test == lowered
                or test in lowered
                or test in lowered_joined
            ):
                ret.add(person)
        if len(ret) == 0:
            return "No user found for the given name."
        # otherwise return just one match
        user = ret.pop()
        # return a json string representation of the user
        return jsons.dumps(user, strip_privates=True)

    def resolve_primary_user(self) -> Annotated[str, ReturnType("str")]:
        """
        Return the primary user instance as a string.
        """
        return jsons.dumps(
            self.tenant.get_person_by_id(self.primary_user_id), strip_privates=True
        )

    def send_message(
        self,
        user_id: Annotated[str, "User id of the person to send message to."],
        content: Annotated[str, "The content of the message to send."],
        title: Annotated[str | None, "Optional title of the message"] = "Request",
    ) -> None:
        """
        Send a message to a user.
        """
        tenant = self.tenant
        # ignore `title` for now
        person_id_to_person_dict = tenant.person_id_to_person_dict
        primary_person = person_id_to_person_dict.get(self.primary_user_id)
        assert primary_person is not None, (
            f"User with id {self.primary_user_id} not found"
        )
        sender = Bot(owner=primary_person)
        assert sender is not None, f"User with id {self.primary_user_id} not found"
        recipient = tenant.get_person_by_id(user_id)
        if recipient is None:
            recipient = tenant.get_person_by_id_relaxed(user_id)
        assert recipient is not None, f"User with id {user_id} not found"
        message = ChatMessage(
            sender=sender,
            recipient=recipient,
            content=content,
            created_on=datetime.now(),
        )
        self.message_send_queue.append(message)

    def send_session_completed(self) -> None:
        """
        If the primary user indicates that they no longer need assistance, send a session completed message from bot to primary user.
        """
        tenant = self.tenant
        primary_person = tenant.get_person_by_id(self.primary_user_id)
        assert primary_person is not None, (
            f"User with id {self.primary_user_id} not found"
        )
        sender = Bot(owner=primary_person)
        assert sender is not None, f"User with id {self.primary_user_id} not found"
        recipient = primary_person
        assert recipient is not None, f"User with id {self.primary_user_id} not found"
        message = SessionCompleted(
            sender=sender,
            recipient=recipient,
            content="Session completed.",
            created_on=datetime.now(),
        )
        self.message_send_queue.append(message)

    @staticmethod
    def get_plugin(
        agent_conf: AgentConfig,
        tenant: Tenant,
        message_send_queue: list[Message] | None,
    ) -> "Plugin":
        primary_user_id = agent_conf.main_user_id
        active_plugins = agent_conf.plugin_ids
        for p in Plugin.dependencies:
            assert p in active_plugins, f"Plugin {p} is a dependency of this plugin"
        assert message_send_queue is not None, "message_send_queue must be provided"
        return EnterprisePlugin(tenant, primary_user_id, message_send_queue)
