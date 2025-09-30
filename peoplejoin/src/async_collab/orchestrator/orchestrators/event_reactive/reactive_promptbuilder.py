from async_collab.orchestrator.datum import AsyncCollabDatum
from async_collab.orchestrator.prompt_builder import PromptBuilder
from async_collab.plugins.plugins.cot_plugin import SimpleReasoningPlugin

IS_MESSAGE_NONE_MODE: bool = False

system_plugin = '''class System:

    # Functions
    def finish() -> None:
        """Call this function to indicate that the current turn is complete."""

'''  # type: ignore

enterprise_plugin = '''class Enterprise:

    # Functions

    def send_message(user_id: str, content: str, title: str | None) -> None:
        """Send a message to a user."""

    def send_session_completed() -> None:
        """If the primary user indicates that they no longer need assistance, send a session completed message."""

    def resolve_primary_user() -> str:
        """Return the primary user details."""

    def resolve_person(name: str) -> str:
        """Find list of persons matching a given name and return details of the first match."""

'''  # type: ignore

enterprise_search_plugin = '''class EnterpriseSearch:

    # Functions
    def search_documents(query: str) -> tuple[str, ...]:
        """Returns the list of relevant documents (including document content/records)"""

    def search_relevant_people(query: str) -> str:
        """Returns names of relevant person and any accompanying rationale"""

'''  # type: ignore


reflection_plugin = '''class Reflection:

    # Functions
    def thought(thought: str) -> None:
        """Assess the current state of the conversation and decide on the next action"""

'''  # type: ignore


class ReactivePromptBuilder(PromptBuilder):
    def get_exemplars_prompt(self) -> str:
        """
        Get the prompt for the given example
        """
        prompt = ""
        for example in self.examples:
            if isinstance(example, AsyncCollabDatum):
                raise NotImplementedError  # TODO: implement this
            else:
                prompt += example + "\n\n"
        return prompt

    def get_instruction_prompt(self) -> str:
        """
        Get the prompt for the instruction
        """
        prompt = "# You are a clever and helpful assistant helping a user. To accomplish the user request, you must use the following Python functions. Each function call should be within a single line, separated by a newline. Never duplicate actions. \n"
        if IS_MESSAGE_NONE_MODE:
            prompt += "# Do not send any messages to any user other than the primary user. If the primary insists to reach out to other users, tell the primary user that you are not allowed to do so.\n"
        return prompt

    def get_plugin_prompts(self) -> str:
        """
        Get the prompt for the plugins
        """
        plugin_ids = {plugin.plugin_id for plugin in self.plugins}
        is_reflection_plugin_on = SimpleReasoningPlugin.plugin_id in plugin_ids
        ret = system_plugin + enterprise_plugin + enterprise_search_plugin
        if is_reflection_plugin_on:
            ret += reflection_plugin
        return ret

    def update_prompt(self, **kwargs):
        """
        Update the prompt with the given arguments
        """
        # if there is event keyword, then call on_event
        if "event" in kwargs:
            self.on_event(kwargs["event"])
        # if there is action keyword, then call on_action
        if "action" in kwargs:
            self.on_action(kwargs["action"])
        # if there is result keyword, then call on_result
        if "result" in kwargs:
            self.on_result(kwargs["result"])
        # for anything else, just add to prompt
        for key, value in kwargs.items():
            if key not in ["event", "action", "result"]:
                self.prompt += value  # note that newline is not added by default
                self.cur_event_repl += value

    def on_action(self, action: str):
        """
        Get the prompt for the instruction
        """
        action = "\n" + action.strip()
        self.prompt += action
        self.cur_event_repl += action

    def on_event(self, message: str):
        """
        Get the prompt for the instruction
        """
        self.prompt += f"\n\n# Event: {message}"
        self.cur_event_repl += f"\n\n# Event: {message}"

    def on_result(self, result: str):
        """
        Get the prompt for the instruction
        """
        if len(result) > 0:
            self.prompt += f"\n{result}"
            self.cur_event_repl += f"\n{result}"

    def init_test_exemplar(self):
        return "### Example ###\n"
