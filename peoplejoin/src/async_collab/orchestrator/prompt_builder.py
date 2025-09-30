from abc import abstractmethod
from dataclasses import dataclass

from async_collab.orchestrator.datum import AsyncCollabDatum
from async_collab.plugins.plugin import Plugin
from async_collab.scenarios.all_exemplars import exemplar_by_id
from logging_config import general_logger


@dataclass
class PromptBuilder:
    """
    ...
    """

    prompt_version: str
    examples: list[AsyncCollabDatum | str]
    plugins: list[Plugin]
    prompt: str
    cur_event_repl: str
    is_function_calling: bool = False

    def __init__(self, plugins: list[Plugin], exemplar_ids: list[str], is_function_calling:bool = False) -> None:
        general_logger.info("[PromptBuilder] PromptBuilder.__init__")
        self.prompt_version: str = "1.0"
        self.plugins: list[Plugin] = plugins
        self.examples: list[AsyncCollabDatum | str] = []
        for exemplar_id in exemplar_ids:
            example = exemplar_by_id.get(exemplar_id)
            assert example is not None, f"Exemplar {exemplar_id} not found"
            self.examples.append(example)
        self.cur_event_repl: str = ""
        self.is_function_calling = is_function_calling
        self.tools = []
        self.reset()

    @abstractmethod
    def get_exemplars_prompt(self) -> str:
        """
        Get the prompt for the examples
        """
        raise NotImplementedError

    @abstractmethod
    def get_instruction_prompt(self) -> str:
        """
        Get the prompt for the instruction
        """
        raise NotImplementedError

    @abstractmethod
    def get_plugin_prompts(self) -> str:
        """
        Get the prompt for the plugins
        """
        raise NotImplementedError

    @abstractmethod
    def update_prompt(self, **kwargs):
        """
        Get the prompt for the instruction
        """
        raise NotImplementedError

    @abstractmethod
    def init_test_exemplar(self):
        """
        Any instruction or annotation at beginning of the test instance
        """
        raise NotImplementedError

    def reset_cur_event_repl(self):
        self.cur_event_repl = ""

    def get_cur_event_repl(self):
        return self.cur_event_repl

    def reset(self):
        """
        Reset the prompt
        """
        if self.is_function_calling:
            raise NotImplementedError("Function calling prompt builder is not implemented yet")
        else:
            self.prompt: str = (
                self.get_instruction_prompt()
                + self.get_plugin_prompts()
                + self.get_exemplars_prompt()
                + self.init_test_exemplar()
            )
