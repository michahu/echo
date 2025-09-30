import asyncio
from dataclasses import dataclass, field
from datetime import datetime

import jsons
from quart import websocket

from src.async_collab.agent.agent_config import AgentConfig
from src.async_collab.core.bot import Bot
from src.async_collab.core.message import ChatMessage, Message
from src.async_collab.core.person import Person
from src.async_collab.llm.llm_client import LLMClient
from src.async_collab.llm.llm_client_service import get_llm_client
from src.async_collab.orchestrator.datum import (
    AsyncCollabDatumMetadata,
    AsyncCollabOutputDatum,
)
from src.async_collab.orchestrator.orchestrator import Orchestrator
from src.async_collab.orchestrator.orchestrators.event_reactive.reactive_orchestrator import (
    ReactiveOrchestrator,
)
from src.async_collab.tenant.tenant import Tenant
from src.async_collab.tenant.tenant_loaders import TenantLoader
from src.logging_config import general_logger

WAIT_TIME = 0.5


@dataclass
class AgentLogger:
    """
    We would like to record the messages and the statements that trigger the messages in an async collaboration task.
    And compile it into AsyncCollabDatum for evaluation and replay.
    This data structure helps in recording the messages and the statements that trigger the messages.
    And finally, converts it into AsyncCollabDatum.
    """

    events: list[Message] = field(default_factory=list)
    initial_message: Message | None = None
    sent_messages: list[Message] = field(default_factory=list)
    repl: list[str] = field(default_factory=list)
    repl_and_messages: list[str | Message] = field(default_factory=list)

    def register_event(self, event: Message):
        self.events.append(event)
        self.repl_and_messages.append(event)

    def register_send_message(self, message: Message):
        general_logger.info(f"[AgentLogger] register_send_message: {message}")
        self.sent_messages.append(message)
        if len(self.events) == 0:
            self.initial_message = message
        self.repl_and_messages.append(message)

    def register_repl(self, repl_code: str | None):
        if repl_code is None:
            return
        self.repl.append(repl_code.strip())
        self.repl_and_messages.extend(repl_code.strip().split("\n"))
        general_logger.info(f"[AgentLogger] [register_repl] repl_code = {repl_code}")

    def reset(self):
        self.events = []
        self.initial_message = None
        self.sent_messages = []
        self.repl = []
        self.repl_and_messages = []

    def get_async_collab_output_datum(
        self, datum_id: str, primary: Person, metadata: AsyncCollabDatumMetadata
    ) -> AsyncCollabOutputDatum:
        return AsyncCollabOutputDatum(
            datum_id=datum_id,
            primary=primary,
            metadata=metadata,
            initial_message=self.initial_message,
            events=self.events,
            sent_messages=self.sent_messages,
            repl=self.repl,
            repl_and_messages=self.repl_and_messages,
        )


class Agent:
    def __init__(self, agent_config: AgentConfig):
        tenant_id = agent_config.tenant_id
        self.tenant: Tenant = TenantLoader.from_id(tenant_id)
        owner = self.tenant.get_person_by_id(agent_config.main_user_id)
        general_logger.info(
            f"[Agent] agent_config.main_user_id = {agent_config.main_user_id}"
        )
        general_logger.info(f"[Agent] load_pth = {agent_config.load_pth}")
        assert owner is not None
        self.owner: Person = owner
        self.bot: Bot = Bot(owner=self.owner)
        self.llm_service: LLMClient = get_llm_client(agent_config.model_config.model)
        self.agent_logger = AgentLogger()
        self.send_queue: list[Message] = []
        self.orchestrator: Orchestrator = self._get_orchestrator(
            agent_config, self.llm_service, self.send_queue
        )

    def _get_orchestrator(
        self,
        agent_config: AgentConfig,
        llm_client: LLMClient,
        send_queue: list[Message],
    ) -> Orchestrator:
        if agent_config.orchestrator_id == "event_driven_reactive":
            return ReactiveOrchestrator(
                agent_config=agent_config,
                tenant=self.tenant,
                llm_client=llm_client,
                send_queue=send_queue,
            )
        elif agent_config.orchestrator_id == "event_driven_func_calling":
            from src.async_collab.orchestrator.orchestrators.event_reactive_func_calling.reactive_orchestrator import (
                FuncCallingReactiveOrchestrator,
            )
            return FuncCallingReactiveOrchestrator(
                agent_config=agent_config,
                tenant=self.tenant,
                llm_client=llm_client,
                send_queue=send_queue,
            )
        else:
            raise NotImplementedError

    async def sending(self):
        # continuously drain `send_queue` and yield elements as they are available
        while True:
            if self.send_queue:
                message = self.send_queue.pop(0)
                message_json = jsons.dumps(message)
                general_logger.info(f"[Agent] Sending message: {message_json}")
                if isinstance(message.recipient, Person):
                    await websocket.send(message_json)
                self.agent_logger.register_send_message(message)
            else:
                await asyncio.sleep(WAIT_TIME)

    async def receiving(self):
        while True:
            data = await websocket.receive()
            message = jsons.loads(data, cls=Message)
            self.agent_logger.register_event(message)
            assert self.bot == message.recipient
            assert isinstance(
                message.sender, Person
            ), "Messages received by agent must be from a person to a bot."
            self.on_message_received(message)

    def _initial_greet_message(self) -> ChatMessage:
        return ChatMessage(
            sender=self.bot,
            recipient=self.owner,
            content="Hello, I'm here to help you as your Agent.",
            created_on=datetime.now(),
        )

    def reset(self) -> None:
        # reset any stateful components of the agent
        # send the inital greeting Message
        self.agent_logger.reset()
        del self.send_queue[
            :
        ]  # remove content but keep the reference, since queue is shared with relevant plugins that populate it
        greet_message = self._initial_greet_message()
        self.send_queue.append(greet_message)
        self.orchestrator.reset()

    def on_message_received(self, event: Message):
        general_logger.info(f"[Agent] on_message_received: {event}")
        # call agent_logger.register_event(event)
        self._invoke_orchestrator(event)

    def _invoke_orchestrator(self, event: Message) -> str | None:
        repl = self.orchestrator.on_event(event)
        if repl is not None:
            self.agent_logger.register_repl(repl)

    @property
    def other_people_in_tenant(self) -> list[Person]:
        all_people = self.tenant.people
        return [person for person in all_people if person != self.owner]

    @property
    def list_of_plugins(self):
        return self.orchestrator.plugins

    def compile_into_datum(
        self, datum_id: str, metadata: AsyncCollabDatumMetadata
    ) -> AsyncCollabOutputDatum:
        # compile the event into a datum
        return self.agent_logger.get_async_collab_output_datum(
            datum_id=datum_id, primary=self.owner, metadata=metadata
        )
