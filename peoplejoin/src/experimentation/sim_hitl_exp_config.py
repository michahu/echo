from dataclasses import dataclass, field
from typing import Any

from src.async_collab.agent.agent_config import AgentConfig
from src.async_collab.orchestrator.datum import AsyncCollabDatumMetadata


@dataclass(frozen=True, eq=True, unsafe_hash=True)
class ExpSimulHitlConfig:  # config for the experiment with hitl or simulation

    """
    The config is used in src/experimentation/experiment_with_hitl_or_simulation.py
    It defines the agent config, the initial messages, the hitl mode, and the participant descriptions.
    hitl: Human-in-the-loop mode; if True, the bot's responses are generated using a language model, but the user's responses are manually entered via the terminal
    if False, the user's responses are generated using a language model conditioned on the user's persona description and the message history
    """

    agent_config_path: str
    agent_config: AgentConfig
    tenant_id: str
    # hitl mode is set per user, so that some users can be simulated, while others can be through terminal
    participant_id_to_hitl_mode: dict[str, bool] = field(default_factory=dict)
    participant_descriptions: dict[str, str] = field(
        default_factory=dict
    )  # indexed by person_id
    participant_instructions: dict[str, str] = field(
        default_factory=dict
    )  # indexed by person_id
    datum_id: str = ""  # the id of the datum to be used in the experiment. logs will be saved with this id, if empty, a random id will be generated
    load_pth: str | None = None  # path to load the previous context from
    metadata: AsyncCollabDatumMetadata | None = None  # metadata for the experiment
    stop_token: str = "<eos>"  # stop token for simulated user responses
    default_instructions_file_name: str | None = field(
        default=None
    )  # the name of the file containing the default instructions for the participants. if None, the default instructions will be used

    @staticmethod
    def sim_config_builder(
        tenant_id: str,
        agent_config_path: str,
        participant_id_to_descriptions: dict[str, str],
        load_pth: str | None = None,
        participant_id_to_instructions: dict[str, str] | None = None,
        datum_id: str = "",
        metadata: dict[str, Any] | None = None,
        stop_token: str = "<eos>",
        default_instructions_file_name: str | None = None,
        participant_id_to_hitl_mode: dict[str, bool] | None = None,
    ):
        agent_conf = AgentConfig.load(agent_config_path, load_pth=load_pth)
        participant_descriptions = participant_id_to_descriptions
        participant_instructions = (
            participant_id_to_instructions
            if participant_id_to_instructions is not None
            else {}
        )
        print("[ExpSimulHitlConfig] sim_config_builder: metadata:", metadata)
        async_collab_metadata = (
            AsyncCollabDatumMetadata.from_dict(metadata)
            if metadata is not None
            else None
        )
        print(
            "[ExpSimulHitlConfig] sim_config_builder: async_collab_metadata:",
            async_collab_metadata,
        )
        participant_id_to_hitl_mode = (
            participant_id_to_hitl_mode
            if participant_id_to_hitl_mode is not None
            else {}
        )
        return ExpSimulHitlConfig(
            agent_config_path=agent_config_path,
            agent_config=agent_conf,
            tenant_id=tenant_id,
            participant_descriptions=participant_descriptions,
            participant_instructions=participant_instructions,
            datum_id=datum_id,
            load_pth=load_pth,
            metadata=async_collab_metadata,
            stop_token=stop_token,
            default_instructions_file_name=default_instructions_file_name,
            participant_id_to_hitl_mode=participant_id_to_hitl_mode,
        )
