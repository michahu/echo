import re
import os

from src.async_collab.agent.agent_config import AgentConfig
from src.async_collab.core.message import Message
from src.async_collab.llm.llm_client import LLMClient
from src.async_collab.orchestrator.orchestrator import Orchestrator
from src.async_collab.orchestrator.orchestrators.event_reactive.reactive_promptbuilder import (
    ReactivePromptBuilder,
)
from src.async_collab.tenant.tenant import Tenant
from src.logging_config import general_logger, prompt_logger


class ReactiveOrchestrator(Orchestrator):
    def __init__(
        self,
        agent_config: AgentConfig,
        tenant: Tenant | None = None,
        send_queue: list[Message] | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        super().__init__(agent_config, tenant, send_queue, llm_client=llm_client)

        self.load_pth = agent_config.load_pth
        self.use_mock_tools: bool = False

        # tool implementations
        general_logger.info(
            f"[ReactiveOrchestrator] self.plugin_name_to_plugin.keys() = {self.plugin_name_to_plugin.keys()}"
        )

        self.tool_implementations = {
            plugin.plugin_name: dict(plugin.plugin_impls.items())
            for plugin in self.plugins
        }

        # mock tool implementations for debugging
        self.mock_tool_implementations = {
            "EnterpriseSearch": {
                "search_documents": lambda query: [f"Document about {query} "],
                "search_relevant_people": lambda query: [f"Person relevant to {query}"],
            },
            "Enterprise": {
                "send_message": lambda user_idx,
                message,
                title=None: f"Sent message to user {user_idx} with message '{message}' and title '{title}'",
                "resolve_person": lambda name: f"Resolved Person('{name.lower()}', '{name.lower()}@example.com')",
                "resolve_primary_user": lambda: "Resolved primary user ",
                "send_session_completed": lambda: "Sent session completed message",
            },
            "System": {"finish": lambda: "Finished the session."},
            "Reflection": {"thought": lambda thought: f"Thought: {thought}"},
        }

    def init_prompt_builder(self, exemplar_ids: list[str]):
        self.prompt_builder = ReactivePromptBuilder(self.plugins, exemplar_ids)

    def on_event(self, event: Message) -> str | None:
        event_prompt = event.as_prompt
        if not event_prompt.strip().startswith("Enterprise.send_session_completed"):
            self.prompt_builder.update_prompt(event=event_prompt)
            return self.run_loop()
        return None

    def call_llm(self) -> str | None:
        """
        Call the LLM to get the next action
        """
        prompt = self.prompt_builder.prompt
        # make call to llm
        assert self.llm_client is not None
        response = self.llm_client.get_response_str(prompt, max_tokens=1000)
        return response

    def run_loop(self) -> str:
        """
        The main orchestration loop that interacts with the LLM to make decisions.
        """
        loop_active = True
        max_error_count = 3
        max_iter = 16
        self.prompt_builder.reset_cur_event_repl()
        while loop_active and max_iter > 0 and max_error_count > 0:
            # Simulate the LLM by calling a function to generate the next action
            max_iter -= 1

            next_action = self.call_llm()
            general_logger.info(f"[ReactiveOrchestrator] next_action = {next_action}")

            if next_action is None:
                print("Error in LLM response. Retrying...")
                max_error_count -= 1
                continue

            # Execute the action (parsed LLM response)
            result = self.execute_action(next_action)

            if result is not None:
                self.prompt_builder.update_prompt(action=next_action)
                self.prompt_builder.update_prompt(result=result)
            else:
                max_error_count -= 1
                # add a message to the prompt that the action was invalid
                # write it in form of a python comment
                self.prompt_builder.update_prompt(
                    error_msg=f"\n# Invalid action: `{next_action}`. Retry with valid action names and parameters. Ensure that prediction is within a single line and only valid plugins and tools are used."
                )
                general_logger.info(
                    f"Invalid action: `{next_action}`. Retry with valid action names and parameters. Ensure that prediction is within a single line."
                )

            if next_action.strip().startswith(
                "System.finish"
            ) or next_action.strip().startswith(">>> System.finish"):
                general_logger.info("System finish called. Exiting loop.")
                loop_active = False
                break

            if next_action.strip().startswith(
                "Enterprise.send_session_completed"
            ) or next_action.strip().startswith(
                ">>> Enterprise.send_session_completed"
            ):
                general_logger.info("Enterprise send_session_completed called.")
                loop_active = False
                break

        cur_event_repl = self.prompt_builder.get_cur_event_repl()
        return cur_event_repl

    def parse_action(self, action) -> tuple[str, str, list[str | tuple[str, str]]]:
        """
        Parses a function signature to extract the class name, function name, and parameters.

        Args:
            signature (str): The function signature as a string.

        Returns:
            tuple: A tuple containing the class name, function name, and a list of parameters.
                Named parameters are represented as tuples (name, value),
                and unnamed parameters are represented as strings.
        """
        action = action.strip()
        # strip any number of '>' characters
        action = action.lstrip(">").strip()

        # First try to match with parentheses
        match = re.match(r"(\w+)\.(\w+)\((.*)\)", action)
        if match:
            class_name = match.group(1)
            function_name = match.group(2)
            params_string = match.group(3).strip()
        else:
            # Try to match without parentheses (e.g., System.finish)
            match = re.match(r"(\w+)\.(\w+)$", action)
            if not match:
                raise ValueError(f"Invalid action: {action}")

            class_name = match.group(1)
            function_name = match.group(2)
            params_string = ""

        parameters = []
        if params_string:
            # Parse parameters while respecting quoted strings
            current_param = []
            in_quotes = False
            quote_char = None
            for char in params_string:
                if char in "\"'" and (not in_quotes or char == quote_char):
                    in_quotes = not in_quotes
                    quote_char = char if in_quotes else None
                elif char == "," and not in_quotes:
                    # End of a parameter
                    param = "".join(current_param).strip()
                    parameters.append(param)
                    current_param = []
                    continue
                current_param.append(char)
            # Add the last parameter
            if current_param:
                parameters.append("".join(current_param).strip())

            # Process each parameter to split named ones
            processed_parameters = []
            for param in parameters:
                if "=" in param:  # Named parameter
                    name, value = map(str.strip, param.split("=", 1))
                    processed_parameters.append((name, value.strip("\"'")))
                else:  # Unnamed parameter
                    processed_parameters.append(param.strip("\"'"))
            parameters = processed_parameters

        return class_name, function_name, parameters

    def execute_action(self, action: str) -> str | None:
        """
        Parses and executes the action. Returns the action and the result if applicable.
        Handles both named and unnamed parameters.
        """
        if self.use_mock_tools:
            tool_implementations = self.mock_tool_implementations
        else:
            tool_implementations = self.tool_implementations

        action = action.strip()
        general_logger.info(
            f"[ReactiveOrchestrator] [execute_action] action=`{action}`"
        )

        try:
            plugin_name, tool_name, parameters = self.parse_action(action)
        except ValueError as e:
            general_logger.error(
                f"[ReactiveOrchestrator] [execute_action] Error parsing action: {e}"
            )
            return None

        general_logger.info(
            f"[ReactiveOrchestrator] [execute_action] action {action} parsed into plugin_name={plugin_name} and tool_name={tool_name}"
        )

        if plugin_name not in tool_implementations:
            general_logger.error(
                f"[ReactiveOrchestrator] [execute_action] Plugin {plugin_name} not found in tool_implementations."
            )
            return None

        if tool_name not in tool_implementations[plugin_name]:
            general_logger.error(
                f"[ReactiveOrchestrator] [execute_action] Tool {tool_name} not found in plugin {plugin_name}."
            )
            return None

        impl = tool_implementations[plugin_name][tool_name]
        # Call the tool implementation with the parameters

        print("*parameters:", *parameters)

        # compute params_without_name: list of parameters without the name. if name exists, simply remove it
        parameters = [
            param[1] if isinstance(param, tuple) else param for param in parameters
        ]

        try:
            result = impl(*parameters)
            general_logger.info(
                f"[ReactiveOrchestrator] [execute_action] result = {result}"
            )
            return f"{result}" if result else ""
        except Exception as e:
            general_logger.error(
                f"[ReactiveOrchestrator] [execute_action] Error executing action: {e}"
            )
            return None


class Orch:
    mock_tool_implementations = {
        "EnterpriseSearch": {
            "search_documents": lambda query: [f"Document about {query} "]
            if isinstance(query, str)
            else "ERROR",
            "search_relevant_people": lambda query: [f"Person relevant to {query}"]
            if isinstance(query, str)
            else "ERROR",
        },
        "Enterprise": {
            "send_message": lambda user_idx,
            message,
            title=None: f"Sent message to user {user_idx} with message '{message}' and title '{title}'"
            if isinstance(user_idx, str) and isinstance(message, str)
            else "ERROR",
            "resolve_person": lambda name: f"Resolved Person('{name.lower()}', '{name.lower()}@example.com')"
            if isinstance(name, str)
            else "ERROR",
            "resolve_primary_user": lambda: "Resolved primary user ",
            "send_session_completed": lambda: "Sent session completed message",
        },
        "System": {"finish": lambda: "Finished the session."},
        "Reflection": {
            "thought": lambda thought: f"Thought: {thought}"
            if isinstance(thought, str)
            else "ERROR"
        },
    }

    def execute_action(self, signature: str):
        """
        Parses a function signature to extract the class name, function name, and parameters.

        Args:
            signature (str): The function signature as a string.

        Returns:
            tuple: A tuple containing the class name, function name, and a list of parameters.
                Named parameters are represented as tuples (name, value),
                and unnamed parameters are represented as strings.
        """
        signature = signature.strip()
        # Regex to match the class name, function name, and parameter list
        match = re.match(r"(\w+)\.(\w+)\((.*)\)", signature)
        if not match:
            raise ValueError(f"Invalid function signature: {signature}")

        class_name = match.group(1)
        function_name = match.group(2)
        params_string = match.group(3).strip()

        parameters = []
        if params_string:
            # Parse parameters while respecting quoted strings
            current_param = []
            in_quotes = False
            quote_char = None
            for char in params_string:
                if char in "\"'" and (not in_quotes or char == quote_char):
                    in_quotes = not in_quotes
                    quote_char = char if in_quotes else None
                elif char == "," and not in_quotes:
                    # End of a parameter
                    param = "".join(current_param).strip()
                    parameters.append(param)
                    current_param = []
                    continue
                current_param.append(char)
            # Add the last parameter
            if current_param:
                parameters.append("".join(current_param).strip())

            # Process each parameter to split named ones
            processed_parameters = []
            for param in parameters:
                if "=" in param:  # Named parameter
                    name, value = map(str.strip, param.split("=", 1))
                    processed_parameters.append((name, value.strip("\"'")))
                else:  # Unnamed parameter
                    processed_parameters.append(param.strip("\"'"))
            parameters = processed_parameters

        print("class_name Name:", class_name)
        print("Function Name:", function_name)
        print("Parameters:", parameters)

        parameters = [
            param[1] if isinstance(param, tuple) else param for param in parameters
        ]
        tool = self.mock_tool_implementations[class_name][function_name]
        print("*params:", *parameters)
        result = tool(*parameters)
        return result

        # return class_name, function_name, parameters
