import re
import os

from src.async_collab.agent.agent_config import AgentConfig
from src.async_collab.core.message import Message
from src.async_collab.llm.llm_client import LLMClient
from src.async_collab.orchestrator.orchestrator import Orchestrator
from src.async_collab.orchestrator.orchestrators.event_reactive_func_calling.reactive_promptbuilder import (
    FuncCallingReactivePromptBuilder,
)
from src.async_collab.tenant.tenant import Tenant
from src.logging_config import general_logger, prompt_logger


class FuncCallingReactiveOrchestrator(Orchestrator):
    def __init__(
        self,
        agent_config: AgentConfig,
        tenant: Tenant | None = None,
        send_queue: list[Message] | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.load_pth = agent_config.load_pth
        general_logger.info(
            f"[ReactiveOrchestrator] Initializing with load_pth = {self.load_pth}"
        )
        super().__init__(agent_config, tenant, send_queue, llm_client=llm_client)

        self.use_mock_tools: bool = False

        general_logger.info(
            f"[ReactiveOrchestrator] self.plugin_name_to_plugin.keys() = {self.plugin_name_to_plugin.keys()}"
        )
        self.tool_implementations = {}
        for plugin in self.plugins:
            for pname, pimpl in plugin.plugin_impls.items():
                self.tool_implementations[pname] = pimpl

        self.mock_tool_implementations = {
            "search_documents": lambda query: [f"Document about {query} "],
            "search_relevant_people": lambda query: [f"Person relevant to {query}"],
            "send_message": lambda user_idx,
            message,
            title=None: f"Sent message to user {user_idx} with message '{message}' and title '{title}'",
            "resolve_person": lambda name: f"Resolved Person('{name.lower()}', '{name.lower()}@example.com')",
            "resolve_primary_user": lambda: "Resolved primary user ",
            "send_session_completed": lambda: "Sent session completed message",
            "finish": lambda: "Finished the session.",
            "thought": lambda thought: f"Thought: {thought}",
        }

    def init_prompt_builder(self, exemplar_ids: list[str]):
        self.prompt_builder = FuncCallingReactivePromptBuilder(
            self.plugins, exemplar_ids, load_pth=self.load_pth
        )

    def call_llm_with_functions(self, messages, functions_schema):
        """
        Call the LLM with function calling enabled.
        """
        assert self.llm_client is not None
        request = {
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 2000,
            "functions": functions_schema,
        }
        prompt_logger.info(f"LLM REQUEST: {request}")
        response = self.llm_client.send_request(
            model=str(self.llm_client.model),
            request=request,
        )
        prompt_logger.info(f"LLM RESPONSE: {response}")
        return response

    def on_event(self, event: Message) -> str | None:
        event_prompt = event.as_prompt
        self.prompt_builder.update_prompt(event=event_prompt)
        repl = self.run_loop()
        return repl

    def run_loop(self) -> str:
        """
        The main orchestration loop that interacts with the LLM to make decisions using function calling.
        """
        # Build initial messages from the prompt builder
        messages = self.prompt_builder.messages
        max_error_count = 3
        max_iter = 50

        # Prepare functions schema as a list
        functions_schema = self.prompt_builder.tools

        for _ in range(max_iter):
            if max_error_count <= 0:
                general_logger.warning("Exiting loop due to max error count.")
                break
            response = self.call_llm_with_functions(messages, functions_schema)
            choice = response.choices[0]
            message = choice.message

            if hasattr(message, "function_call") and message.function_call:
                func_call = message.function_call
                tool_name = func_call.name
                try:
                    arguments = {}
                    if hasattr(func_call, "arguments") and func_call.arguments:
                        import json

                        arguments = json.loads(func_call.arguments)
                except Exception as e:
                    general_logger.error(
                        f"Error decoding JSON arguments for tool {tool_name}: {e}"
                    )
                    max_error_count -= 1
                    continue

                general_logger.info(
                    f"LLM called tool: {tool_name} with arguments: {arguments}"
                )

                # Find the plugin/tool implementation
                tool_impl = None
                tool_impl = (
                    self.tool_implementations.get(tool_name, None)
                    if not self.use_mock_tools
                    else self.mock_tool_implementations.get(tool_name, None)
                )

                if tool_impl is None:
                    general_logger.error(f"Tool {tool_name} not found in any plugin.")
                    max_error_count -= 1
                    continue

                # Call the tool implementation
                try:
                    # Pass arguments as positional or keyword as needed
                    import inspect

                    sig = inspect.signature(tool_impl)
                    if len(arguments) == 0:
                        tool_result = tool_impl()
                    elif (
                        len(arguments) == 1
                        and list(arguments.keys())[0] in sig.parameters
                    ):
                        tool_result = tool_impl(*arguments.values())
                    else:
                        tool_result = tool_impl(**arguments)

                except Exception as e:
                    general_logger.error(f"Error executing tool {tool_name}: {e}")
                    max_error_count -= 1
                    continue

                # Append the function call message and the function result
                messages.append(message)
                if tool_result:
                    messages.append(
                        {
                            "role": "function",
                            "name": tool_name,
                            "content": str(tool_result),
                        }
                    )

                # If the tool is 'finish', stop and return the result
                if tool_name == "finish" or tool_name == "send_session_completed":
                    general_logger.info(f"Tool {tool_name} completed the session.")
                    return str(tool_result)
            else:
                general_logger.info("LLM did not call any tool.")
                max_error_count -= 1

        # end the session
        tool_impl = self.tool_implementations.get("send_session_completed", None)
        tool_result = tool_impl()
        if tool_result:
            messages.append(
                {
                    "role": "function",
                    "name": tool_name,
                    "content": str(tool_result),
                }
            )
        return "Error: Too many iterations or errors in function calling loop."
