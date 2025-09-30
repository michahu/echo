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


# Configure the prompt logger for logging created prompts
prompt_logger = logging.getLogger("prompt")
prompt_logger.setLevel(logging.INFO)

prompt_file_handler = logging.FileHandler("logs/prompt.log", mode="w")
prompt_file_handler.setLevel(logging.INFO)

prompt_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
prompt_file_handler.setFormatter(prompt_formatter)

prompt_logger.addHandler(prompt_file_handler)

# Configure the repl logger for logging created repls
repl_logger = logging.getLogger("repl")
repl_logger.setLevel(logging.INFO)

repl_file_handler = logging.FileHandler("logs/repl.log", mode="w")
repl_file_handler.setLevel(logging.INFO)

repl_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
repl_file_handler.setFormatter(repl_formatter)

repl_logger.addHandler(repl_file_handler)
repl_logger.addHandler(console_handler)  # print repl to console


# logger for the simulated user
simulated_user_logger = logging.getLogger("simulated_user")
simulated_user_logger.setLevel(logging.INFO)

user_file_handler = logging.FileHandler("logs/simulated_user.log", mode="w")
user_file_handler.setLevel(logging.INFO)

user_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
user_file_handler.setFormatter(user_formatter)

simulated_user_logger.addHandler(user_file_handler)
# simulated_user_logger.addHandler(console_handler)  

# logger for evaluation
evaluation_logger = logging.getLogger("eval")
evaluation_logger.setLevel(logging.INFO)

evaluation_file_handler = logging.FileHandler("logs/eval.log", mode="w")
evaluation_file_handler.setLevel(logging.INFO)

evaluation_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
evaluation_file_handler.setFormatter(evaluation_formatter)

evaluation_logger.addHandler(evaluation_file_handler)


# logger for data processing and preparation
data_processing_logger = logging.getLogger("data_processing")
data_processing_logger.setLevel(logging.INFO)

data_processing_file_handler = logging.FileHandler("logs/data_processing.log", mode="a")
data_processing_file_handler.setLevel(logging.INFO)

data_processing_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
data_processing_file_handler.setFormatter(data_processing_formatter)

data_processing_logger.addHandler(data_processing_file_handler)
data_processing_logger.addHandler(console_handler)  # print data processing to console


# logger for enterprise search plugin (document and person search)
enterprise_search_logger = logging.getLogger("enterprise_search")
enterprise_search_logger.setLevel(logging.INFO)

enterprise_search_file_handler = logging.FileHandler(
    "logs/enterprise_search.log", mode="a"
)
enterprise_search_file_handler.setLevel(logging.INFO)

enterprise_search_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
enterprise_search_file_handler.setFormatter(enterprise_search_formatter)

enterprise_search_logger.addHandler(enterprise_search_file_handler)
