import os
import string
from datetime import datetime

from src.async_collab.core.bot import Bot
from src.async_collab.core.document import Document, DocumentCollection
from src.async_collab.core.message import Message
from src.async_collab.core.person import Person
from src.async_collab.llm.llm_client import LLMClient
from src.logging_config import simulated_user_logger

DEFAULT_DESCRIPTION = "This user is a professional worker at XYZ corp."


def repr_documents(documents: list[Document]) -> list[str]:
    """
    This method returns a string representation of the documents
    """
    ret = []
    for doc in documents:
        ret.append("#### Document:\n")
        ret.append(doc.get_content(trim_content=False))
    return ret


def format_date_time(date_time: datetime) -> str:
    return date_time.strftime("%Y-%m-%d %H:%M:%S")


class UserSimulatorPromptBuilder:
    instructions: str
    description: str

    # This is simple enough for now; if more additions are made, consider adding a unit test

    def __init__(self, instructions: str, description: str):
        self.instructions = instructions
        self.description = description

    def __call__(
        self,
        messages_history: list[Message],
        user_name: str,
        documents: list[Document] | None = None,
    ) -> str:
        """
        This method builds a prompt for the user simulator based on the messages history and the description of the user
        """
        prompt: list[str] = []
        prompt.append(self.instructions + "\n")
        prompt.append(f"### User Description: {self.description} \n")
        prompt.append(
            f"### Info: Today's date is {format_date_time(datetime.now())}.\n"
        )  # todo: load from config
        if documents:
            prompt.append("### User Documents:\n")
            prompt.extend(repr_documents(documents))
            # todo: might have to truncate later on if the content is too long
        prompt.append("### Messages History: \n")
        for message in messages_history:
            if isinstance(message.sender, Bot):
                prompt.append(f"Bot: {message.content}\n")
            else:
                prompt.append(f"{message.sender.full_name}: {message.content}\n")
        prompt.append("### Next Response:\n")
        prompt.append(f"{user_name}:")
        return "".join(prompt)


translator = str.maketrans("", "", string.punctuation)


def _check_if_skip_response(content: str) -> bool:
    content_to_check = content.translate(translator)
    content_to_check = content_to_check.strip().lower()
    return content_to_check.startswith(
        "skip"
    )  # this could create issues if user actually wants to send a message starting with 'skip'


def _strip_common_awkward_phrases(content: str) -> str:
    # if content has '"""', then strip anything after the first """
    if '"""' in content:
        return content[: content.index('"""')]
    return content


class UserSimulator:
    user: Person
    description: str
    instruction: str
    messages: list[Message]
    prompt_builder: UserSimulatorPromptBuilder
    llm_client: LLMClient | None
    hitl: bool
    doc_search: DocumentCollection | None
    stop_token: str
    is_main_user: bool

    def __init__(
        self,
        user: Person,
        description: str,
        llm_client: LLMClient | None = None,
        hitl: bool = False,
        provided_instruction: str | None = None,
        provided_instruction_name: str | None = None,
        doc_search: DocumentCollection | None = None,
        stop_token: str = "<eos>",
        is_main_user: bool = False,
    ):
        self.user = user
        self.description = description
        assert provided_instruction is None or provided_instruction_name is None
        if provided_instruction is not None:
            instruction = provided_instruction
            simulated_user_logger.info(
                f"UserSimulator: using provided_instruction: {instruction[:11]}..."
            )
        elif provided_instruction_name is not None:
            # assert that the file exists
            instruction_file_path = f"src/experimentation/simulated_user_instructions/{provided_instruction_name}.md"
            assert os.path.exists(
                instruction_file_path
            ), f"Provided default instruction file {instruction_file_path} does not exist."
            with open(instruction_file_path) as f:
                instruction = f.read()
            simulated_user_logger.info(
                f"UserSimulator: using instructionss from file {instruction_file_path}"
            )
        else:
            instruction_file_path = (
                "src/experimentation/simulated_user_instructions/default.md"
            )
            assert os.path.exists(
                instruction_file_path
            ), f"Default instruction file {instruction_file_path} does not exist."
            with open(instruction_file_path) as f:
                instruction = f.read()
            simulated_user_logger.info(
                f"UserSimulator: using default instruction file {instruction_file_path}"
            )
        self.prompt_builder = UserSimulatorPromptBuilder(instruction, self.description)
        self.messages = []
        self.llm_client = llm_client
        self.hitl = hitl
        assert (llm_client is not None) or hitl
        # create logs/ folder if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        self.doc_search = doc_search
        simulated_user_logger.info(
            f"UserSimulator created for user {self.user} with description: {self.description}."
        )
        if self.doc_search is not None:
            simulated_user_logger.info(
                f"UserSimulator: document search index for user {self.user} has {len(self.doc_search.documents)} documents."
            )
        self.stop_token = stop_token
        self.is_main_user = is_main_user

    def search_documents(self, query: str) -> list[Document]:
        """
        This method searches the documents for the user based on the query
        """
        assert self.doc_search is not None
        return self.doc_search.search_documents(query)

    def get_all_documents(self) -> list[Document] | None:
        """
        This method returns all the documents for the user
        """
        if self.doc_search is None:
            return None
        return self.doc_search.documents

    async def respond(self, message_from_bot: Message) -> tuple[Message | None, bool]:
        """
        This method simulates a user responding to a message from a bot.
        Can condition on the entire messages history and the description of the user
        Returns a tuple of the message to send and a boolean indicating whether to send the message or not
        We might want to skip sending a message when there is no response expected from the user. For example, when bot acknolwedges Alice's request
        """
        self.messages.append(message_from_bot)
        send_message: bool = True
        skip_response: bool = False

        # TODO: need to allow tool prediction for document search with a query. i.e.simulated user can use `search_documents(query)` to get a list of relevant documents
        # Currently putting all documents in the prompt
        relevant_documents = self.get_all_documents()

        if self.hitl:
            print("=" * 33)
            print(f"[HITL MODE for user {self.user}] ")
            if relevant_documents:
                print("## User Documents: ")
                doc_repr = repr_documents(relevant_documents)
                print("\n".join(doc_repr))
                print("=" * 11)
            print("## History: ")
            for message in self.messages:
                print(f"History: {message.sender}: {message.content}")
            print("=" * 11)
            # content = input("Enter your response (Enter SKIP to skip this turn)")
            print(
                "Enter/Paste your response (Can be multi-line). Type SKIP to skip this turn / no reply to bot. Always press Ctrl-D or Ctrl-Z ( windows ) to signal that you completed the response."
            )
            contents = []

            while True:
                try:
                    line = input()
                except EOFError:
                    break
                contents.append(line)
            content = "\n".join(contents)

            if content.strip().lower() == "skip":
                return_message = None
                send_message = False
                skip_response = True
                return_message = Message(  # record skip messages in transcript
                    sender=self.user,
                    recipient=message_from_bot.sender,
                    content="skip",
                    message_type="chat",
                )
            else:
                return_message = Message(
                    sender=self.user,
                    recipient=message_from_bot.sender,
                    content=content,
                    message_type="chat",
                )
        else:
            prompt = self.prompt_builder(
                self.messages,
                self.user.full_name,
                relevant_documents if not self.is_main_user else None,
            )
            # if not the main user, then don't show the documents

            simulated_user_logger.info(f"--->>> User {self.user}")
            simulated_user_logger.info(f"--->>>> prompt: {prompt}")
            simulated_user_logger.info(
                f"--->>>> stop=self.stop_token: {self.stop_token}"
            )
            assert self.llm_client is not None
            response = self.llm_client.get_response_str(
                prompt, stop=self.stop_token, max_tokens=4000
            )
            simulated_user_logger.info(f"--->>>> response: {response}")
            simulated_user_logger.info("\n=========================\n")
            if response is not None:
                response = _strip_common_awkward_phrases(response)
                skip_response = _check_if_skip_response(response)
            elif response is None or len(response) == 0:
                skip_response = True
            if skip_response:
                send_message = False
                return_message = Message(  # record skip messages in transcript
                    sender=self.user,
                    recipient=message_from_bot.sender,
                    content="skip",
                    message_type="chat",
                )
            else:
                content = response
                assert content is not None
                return_message = Message(
                    sender=self.user,
                    recipient=message_from_bot.sender,
                    content=content,
                    message_type="chat",
                )
        if send_message or skip_response:
            assert return_message is not None
            self.messages.append(return_message)
        return return_message, send_message
