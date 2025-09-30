import logging
import os

os.makedirs("logs", exist_ok=True)

# Configure the main logger for general purposes
general_logger = logging.getLogger("general")
general_logger.setLevel(logging.INFO)

general_file_handler = logging.FileHandler("logs/general.log", mode="a")
general_file_handler.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
general_file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

general_logger.addHandler(general_file_handler)
general_logger.addHandler(console_handler)


# logger for enterprise search plugin (document and person search)
action_results_logger = logging.getLogger("action_results")
action_results_logger.setLevel(logging.INFO)

action_results_file_handler = logging.FileHandler("logs/action_results.log", mode="w")
action_results_file_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
action_results_file_handler.setFormatter(formatter)

action_results_logger.addHandler(action_results_file_handler)
action_results_logger.propagate = False  # Add this line

# llm logger for logging LLM responses
llm_logger = logging.getLogger("llm")
llm_logger.setLevel(logging.INFO)
llm_file_handler = logging.FileHandler("logs/llm.log", mode="w")
llm_file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
llm_file_handler.setFormatter(formatter)
llm_logger.addHandler(llm_file_handler)
llm_logger.propagate = False  # Add this line
