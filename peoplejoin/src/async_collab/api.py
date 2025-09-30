import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import jsons
from quart import Quart, request

from async_collab.agent.agent_config import AgentConfig
from async_collab.agent.base_agent import Agent
from async_collab.orchestrator.datum import AsyncCollabDatumMetadata
from logging_config import general_logger


class App:
    """
    App is the core entrypoint of the backend of the application.
    Broadly, it initializes the state management in the form of a SessionBroker, it loads the relevant AgentConfig, and it then initializes the actual behavior in the form of a DemoAgent.
    """

    metadata: dict[str, str]
    agent: Agent | None  # only one agent for now;
    # can easily extend the setup to introduce session_id
    # agents: dict[str, Agent] # for multiple agents; session_id -> Agent
    # metadata: dict[str, dict[str, str]] # for multiple agents; session_id -> metadata

    def __init__(self, agent_conf_path: str | None = None, load_pth: str | None = None):
        self.app = Quart(__name__)
        self.metadata = {}
        self.load_pth = load_pth
        self.is_event_reactive = False
        if agent_conf_path is not None:
            config = AgentConfig.load(agent_conf_path, load_pth=load_pth)
            self.agent = Agent(agent_config=config)
            general_logger.info(f"[App.init()]: self.agent = {self.agent}")
        else:
            self.agent = None

        async def consume_and_produce() -> None:
            """Initiate the receiving and sending tasks for the agent relative to the given async context"""
            assert self.agent is not None
            producer = asyncio.create_task(self.agent.sending())
            consumer = asyncio.create_task(self.agent.receiving())
            try:
                await asyncio.gather(consumer, producer)
            finally:
                consumer.cancel()
                producer.cancel()

        @self.app.websocket("/ws/<user_id>")
        async def ws(user_id: str) -> None:
            """
            Initiates a user session for the given user_id as a primary user.
            This should be the default choice in general - it initializes the system
            to undertake actions on behalf of that user as that user's agent.
            """
            general_logger.info(f"[API] ws called with user_id = {user_id}")
            await consume_and_produce()

        # end point to load the given tenant
        @self.app.route("/init", methods=["POST"])
        async def init_setup():
            """
            Loads the tenant for the given session_id.
            """
            data = await request.get_json()
            metadata = data.get("metadata")
            if metadata is not None:
                self.metadata = json.loads(metadata)
                general_logger.info(
                    f"[API] init_setup called with metadata = {metadata}"
                )
            agent_config_path = data.get("agent_config_path")
            load_pth = data.get("load_pth", None)
            assert agent_config_path is not None
            # TODO: construct Agent; which implicitly builds orchestrator, tenant, plugins, etc.
            general_logger.info(
                f"[API] init_setup called with agent_config_path = {agent_config_path} and load_pth = {load_pth}"
            )
            agent_config = AgentConfig.load(agent_config_path, load_pth=load_pth)
            self.agent = Agent(agent_config=agent_config)
            self.agent.reset()
            self.is_event_reactive = (
                agent_config.orchestrator_id == "event_driven_reactive"
            )
            return "Loaded Agent and reset the state"

        @self.app.route("/clear")
        async def clear():
            """
            Clears the session state associated with the given user_id, assuming that user_id corresponds to a primary user.
            """
            general_logger.info("[API] clear called")
            self.metadata = {}
            del self.agent
            self.agent = None
            return "No location to clear"

        @self.app.route("/save")
        async def save():
            """
            Saves the information in the session for the given user_id, assuming that user_id corresponds to a primary user.
            """
            folder_path = request.args.get("folder_path", "logs")
            datum_id = request.args.get("datum_id", str(uuid4()))
            general_logger.info(
                f"[API] save called with folder_path = {folder_path} and datum_id = {datum_id}"
            )
            # create folder_path if not exists
            os.makedirs(folder_path, exist_ok=True)
            metadata = AsyncCollabDatumMetadata.from_processed_dict(self.metadata)
            assert self.agent is not None
            datum = self.agent.compile_into_datum(datum_id=datum_id, metadata=metadata)

            # dump agent.orchestrator.prompt_builder.messages
            def serialize_message(m):
                if isinstance(m, dict):
                    return m
                d = m.__dict__
                if "function_call" in d and d["function_call"] is not None:
                    d["function_call"] = d["function_call"].__dict__
                return d

            def convert_message_dict(m):
                if not isinstance(m, str):
                    if hasattr(m, "sender") and hasattr(m.sender, "owner"):
                        return m.sender.owner.full_name + "'s Bot: " + m.content
                    else:
                        return m.sender.full_name + ": " + m.content
                return m

            if self.is_event_reactive:
                with open(f"{folder_path}/{datum_id}.messages.json", "w") as f:
                    json.dump(
                        [
                            convert_message_dict(m)
                            for m in self.agent.agent_logger.repl_and_messages
                        ],
                        f,
                        indent=2,
                    )
            else:
                with open(f"{folder_path}/{datum_id}.messages.json", "w") as f:
                    json.dump(
                        [
                            serialize_message(m)
                            for m in self.agent.orchestrator.prompt_builder.messages
                        ],
                        f,
                        indent=2,
                    )
            general_logger.info(
                f"Saved messages to {folder_path}/{datum_id}.messages.json"
            )

            general_logger.info("STATE SAVED")
            # save in json format at folder_path/{self.datum_id}.datum.json
            # json_str = jsons.dumps(datum.__dict__, indent=2)
            json_str = jsons.dumps(asdict(datum), indent=2)
            with open(f"{folder_path}/{datum_id}.datum.json", "w") as f:
                f.write(json_str)
            general_logger.info(f"Saved datum to {folder_path}/{datum_id}.datum.json")
            # also save the yaml format using save_messages_from_datum_in_ui_yaml_format -- these can be loaded directly in the ui
            return "Saved at " + folder_path


def app(conf_name: str | None = None, load_pth: str | None = None):
    print("Creating the app with conf_name = ", conf_name)
    return App(agent_conf_path=conf_name, load_pth=load_pth).app


if __name__ == "__main__":
    # The port should be passed in by the process that starts this script.
    # For example, when using hypercorn, it will pass the port.
    # This block is for when the script is run directly.
    # We'll default to a common port like 8000 if not provided,
    # but the Makefile will override this.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--use_substrate",
        action="store_true",
        help="Use the substrate client for API requests",
    )
    args = parser.parse_args()

    if args.use_substrate:
        os.environ["USE_SUBSTRATE_CLIENT"] = "1"

    print(f"Starting server on port {args.port}")
    app().run(port=args.port)
