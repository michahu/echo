from async_collab.orchestrator.datum import AsyncCollabDatum
from async_collab.orchestrator.prompt_builder import PromptBuilder
from async_collab.plugins.plugins.cot_plugin import SimpleReasoningPlugin
from src.logging_config import general_logger

import os

IS_MESSAGE_NONE_MODE: bool = False


functions = {
    "finish": {
        "type": "function",
        "name": "finish",
        "description": "Call this function to indicate that the current turn is complete.",
    },
    "send_message": {
        "type": "function",
        "name": "send_message",
        "description": "Send a message to a user.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "The ID of the user to send the message to.",
                },
                "content": {
                    "type": "string",
                    "description": "The content of the message.",
                },
                "title": {
                    "type": ["string", "null"],
                    "description": "An optional title for the message.",
                },
            },
            "required": ["user_id", "content"],
            "additionalProperties": False,
        },
    },
    "send_session_completed": {
        "type": "function",
        "name": "send_session_completed",
        "description": "If the primary user indicates that they no longer need assistance, send a session completed message. If the conversation is going in circles, end the conversation using send_session_completed",
    },
    "resolve_primary_user": {
        "type": "function",
        "name": "resolve_primary_user",
        "description": "Return the primary user details.",
    },
    "resolve_person": {
        "type": "function",
        "name": "resolve_person",
        "description": "Find list of persons matching a given name and return details of the first match.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name of the person to resolve.",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
    "search_documents": {
        "type": "function",
        "name": "search_documents",
        "description": "Returns the list of relevant documents (including document content/records)",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant documents.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    "search_relevant_people": {
        "type": "function",
        "name": "search_relevant_people",
        "description": "Returns names of relevant person and any accompanying rationale",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant people.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    "thought": {
        "type": "function",
        "name": "thought",
        "description": "Assess the current state of the conversation and decide on the next action. Always use this function between other function calls.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "The thought or reflection on the current state of the conversation.",
                },
            },
            "required": ["thought"],
            "additionalProperties": False,
        },
    },
}


func_name_adjustments = {
    "System.finish": "finish",
    "Reflection.thought": "thought",
    "Enterprise.send_message": "send_message",
    "Enterprise.send_session_completed": "send_session_completed",
    "Enterprise.resolve_primary_user": "resolve_primary_user",
    "Enterprise.resolve_person": "resolve_person",
    "EnterpriseSearch.search_documents": "search_documents",
    "EnterpriseSearch.search_relevant_people": "search_relevant_people",
}


class FuncCallingReactivePromptBuilder(PromptBuilder):
    messages: list[dict[str, str]]

    def __init__(
        self,
        plugins: list,
        exemplar_ids: list[str],
        load_pth: str = "",
    ) -> None:
        self.load_pth = load_pth
        self.messages = []  
        # NTS: init calls other methods. 
        general_logger.info(
            f"Initializing FuncCallingReactivePromptBuilder with load_pth: {self.load_pth}"
        )
        super().__init__(plugins, exemplar_ids)

    def get_exemplars_prompt(self) -> str:
        """
        Get the prompt for the given example
        """
        prompt = ""
        for example in self.examples:
            if isinstance(example, AsyncCollabDatum):
                raise NotImplementedError  # TODO: implement this
            else:
                if IS_MESSAGE_NONE_MODE:
                    assert NotImplementedError, (
                        "IS_MESSAGE_NONE_MODE is not implemented for ReactivePromptBuilder"
                    )
                # apply func_name_adjustments
                for func_name, adjusted_name in func_name_adjustments.items():
                    example = example.replace(func_name, adjusted_name)
                # add example to prompt
                prompt += example + "\n\n"
        return prompt

    def get_plugin_prompts(self) -> str:
        """
        Get the prompt for the plugins
        """
        self.tools = [
            functions["finish"],
            functions["send_message"],
            functions["send_session_completed"],
            functions["resolve_primary_user"],
            functions["resolve_person"],
            functions["search_documents"],
            functions["search_relevant_people"],
        ]
        if SimpleReasoningPlugin.plugin_id in {
            plugin.plugin_id for plugin in self.plugins
        }:
            self.tools.append(functions["thought"])
        return ""  # dummy return \

    def reset(self) -> None:
        self.sys_prompt = self.get_instruction_prompt() + self.get_exemplars_prompt()
        if self.load_pth != "":
            # try to load latest_reflection.txt and latest_logs.txt
            if os.path.exists(self.load_pth):
                try:
                    with open(os.path.join(self.load_pth, "latest_hindsight.txt")) as f:
                        prev_reflection = f.read()
                    self.sys_prompt += "Here are your reflections about the current organization. Use this information to help you with your task. This information is more up to date than the search functions."
                    self.sys_prompt += (
                        "\n\n# Begin previous reflection log #\n"
                        + prev_reflection
                        + "\n# End previous reflection log #\n"
                    )
                    general_logger.info(
                        f"Loaded reflection from {self.load_pth + 'latest_reflection.txt'}"
                    )
                except FileNotFoundError:
                    general_logger.warning(
                        f"Reflection file not found at {self.load_pth + 'latest_reflection.txt'}"
                    )

                try:
                    with open(os.path.join(self.load_pth, "latest_logs.txt")) as f:
                        prev_logs = f.read()

                    if prev_logs:
                        self.sys_prompt += (
                            "\n\n# Begin previous interaction logs #\n"
                            + prev_logs
                            + "\n# End previous interaction logs #\n"
                        )
                        general_logger.info(
                            f"Loaded logs from {self.load_pth + 'latest_logs.txt'}"
                        )

                except FileNotFoundError:
                    general_logger.warning(
                        f"Logs file not found at {self.load_pth + 'latest_logs.txt'}"
                    )

        self.get_plugin_prompts()  # sets self.tools
        # important for setting the system prompt
        self.messages = [
            {
                "role": "system",
                "content": self.sys_prompt,
            },
        ]

    def get_instruction_prompt(self) -> str:
        """
        Get the prompt for the instruction
        """
        prompt = """# You are a clever and helpful assistant helping a user. To accomplish the user request, you must use the available functions. Always use a function for any actions, including sending a response back to user. Use the thought function to plan future function calls.Follow your own learned information about the organization first. Information about users from tools can be unreliable! \n"""
        if IS_MESSAGE_NONE_MODE:
            prompt += "# Do not send any messages to any user other than the primary user. If the primary insists to reach out to other users, tell the primary user that you are not allowed to do so.\n"
        return prompt

    def update_prompt(self, **kwargs) -> None:
        event = kwargs.get("event")
        if event is not None:
            self.messages.append(
                {
                    "role": "user",
                    "content": event,
                }
            )
        else:
            assert NotImplementedError, "update_prompt requires an event to be passed"
