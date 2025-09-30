import argparse
import os
import shutil


import json

from azure.identity import ManagedIdentityCredential, get_bearer_token_provider
from openai import AzureOpenAI


model = "gpt-4o-11-20"
endpoint = "" # TODO -- add this
credential = ManagedIdentityCredential()
token_provider = get_bearer_token_provider(
    credential, ""
)

client = AzureOpenAI(
    azure_endpoint=endpoint,
    azure_ad_token_provider=token_provider,
    api_version="2024-12-01-preview",
)


def get_response_from_gpt_azure(
    msg,
    model,
    system_message,
    reasoning_effort="low",
):
    # Prepare the user message content
    user_content = [{"type": "text", "text": msg}]

    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]

    if model == "o3-mini":
        # For o3-mini, we use a different response format
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=10000,
            n=1,
            stop=None,
            response_format={"type": "json_object"},
            reasoning_effort=reasoning_effort,
        )
        content = response.choices[0].message.content
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
            n=1,
            stop=None,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content

    if not content:
        return {}

    json_dict = json.loads(content)
    return json_dict

REFLEXION_PROMPT = """You are a helpful and clever teacher of language agents. You have access to a prior interaction between a language agent and other agents in an organization, as well as your own reflection about the organization. Using the prior interaction and reflection, write a better  reflection that will help a future language agent perform better in this organization.

Structure your reflection in the following json format: {'reflection': reflection}, where reflection is a string. The reflection should be concise and focused on giving instructions to future agents in this organization."""

LOGS = """# Begin previous reflection log #
##REFLECTION##
# End previous reflection log #

# Begin previous interaction log #
##INTERACTIONS##
# End previous interaction log #
"""

ECHO_PROMPT = """You are a helpful and clever teacher of language agents. Given a trajectory, write a simplified counterfactual workflow and final answer. If the trajectory is already efficient, you can simply summarize the events. If the correct final answer is unclear, then do not generate a workflow or final answer.

The counterfactual trajectory should include:
- the query
- a workflow for solving the query
- the final answer

Return a json {'query': query, 'workflow': workflow, 'final_answer': final_answer}. If either the correct workflow or final answer are unclear, then you should not generate a workflow or final answer. To abstain, return empty strings for 'workflow' and 'final_answer': {'query': query, 'workflow': '', 'final_answer': ''}.
"""

AWM_PROMPT = """You are a helpful and clever teacher of language agents. Attached below is a prior interaction between a language agent and other agents in an organization. If you deem the interaction to successfully and accurately answer the initial question, return a summary of the interaction so future agents can easily reference what to do in similar situations. The summary should contain the query, a summary of events, and the final answer.

If the interaction was successful, return a json {'successful': true, 'summary': summary}, where summary is a string. If the interaction was not successful, return a json {'successful': false, 'summary': ''}.
"""



def manage_sliding_window_history(hindsight_dir, new_logs, window_size, file_prefix):
    """
    Manage history using a sliding window approach.
    
    Args:
        hindsight_dir: Directory where history files are stored
        new_logs: New log entry to add
        window_size: Maximum number of recent entries to keep
        file_prefix: Prefix for the current file
    
    Returns:
        str: Combined history content within the window size
    """
    history_file = os.path.join(hindsight_dir, "sliding_window_history.json")
    
    # Load existing history or create empty list
    try:
        with open(history_file, 'r') as f:
            history_entries = json.load(f)
    except FileNotFoundError:
        history_entries = []
    
    # Add new entry with metadata
    new_entry = {
        "file_prefix": file_prefix,
        "logs": new_logs,
        "timestamp": os.path.getmtime(os.path.join(hindsight_dir, f"{file_prefix}.messages.json")) if os.path.exists(os.path.join(hindsight_dir, f"{file_prefix}.messages.json")) else None
    }
    history_entries.append(new_entry)
    
    # Keep only the last window_size entries
    if len(history_entries) > window_size:
        history_entries = history_entries[-window_size:]
    
    # Save updated history
    with open(history_file, 'w') as f:
        json.dump(history_entries, f, indent=2)
    
    # Create combined logs string for backwards compatibility
    combined_logs = []
    for entry in history_entries:
        combined_logs.append(entry["logs"])
    
    combined_logs_str = "\n".join(combined_logs)
    
    # Update latest_logs.txt with the sliding window content
    with open(os.path.join(hindsight_dir, "latest_logs.txt"), 'w') as f:
        f.write(combined_logs_str)
    
    # Save individual run backup
    with open(os.path.join(hindsight_dir, f"{file_prefix}.latest_logs.txt"), 'w') as f:
        f.write(new_logs)
    
    return combined_logs_str


def one_traj_only(new_logs, previous_reflection):
    """Run a simplified method that takes only a trajectory and outputs a counterfactual using ECHO_PROMPT.
    This method calls ECHO_PROMPT directly with just the trajectory, no learned info needed.
    Includes collision handling to save shorter workflows when duplicates occur."""
    
    # Load existing knowledge dict or create empty one
    knowledge_dict = json.loads(previous_reflection) if previous_reflection != "" else {}
    
    # Call ECHO_PROMPT directly with just the trajectory
    echo_response = get_response_from_gpt_azure(
        new_logs,
        model=model,
        system_message=ECHO_PROMPT,
    )
    
    # Extract the response components
    query = echo_response.get("query", "")
    workflow = echo_response.get("workflow", "")
    final_answer = echo_response.get("final_answer", "")
    
    if query and workflow and final_answer:
        combined_workflow_and_answer = f"# Workflow #\n {workflow}\n\nFinal Answer: {final_answer}"
        
        # Check if query already exists and save the shorter workflow based on raw length
        if query in knowledge_dict:
            existing_workflow = knowledge_dict[query]
            if len(combined_workflow_and_answer) < len(existing_workflow):
                knowledge_dict[query] = combined_workflow_and_answer
                print(f"Replaced existing workflow for '{query}' with shorter version ({len(combined_workflow_and_answer)} vs {len(existing_workflow)} chars)")
            else:
                print(f"Kept existing workflow for '{query}' as it's shorter ({len(existing_workflow)} vs {len(combined_workflow_and_answer)} chars)")
        else:
            knowledge_dict[query] = combined_workflow_and_answer
            print(f"# Generated counterfactual for query #\n {query}\n\n Final answer: {final_answer}")
    else:
        print("No counterfactual generated from ECHO_PROMPT")
    
    return json.dumps(knowledge_dict, indent=2)


def main():
    """
    Main function to run offline computation on experiment results.
    """
    parser = argparse.ArgumentParser(description="Run offline computation on experiment results.")
    parser.add_argument("--log_path", type=str, help="Path to the new messages.json file.")
    parser.add_argument("--algo", type=str, help="The algorithm to use for processing.")
    parser.add_argument("--history", choices=["full", "none"], default="full", help="Show the model prior interactions?")
    parser.add_argument("--history_window_size", type=int, default=5, help="Number of recent runs to keep in history (sliding window size)")
    args = parser.parse_args()

    hindsight_dir = os.path.dirname(args.log_path)
    log_filename = os.path.basename(args.log_path)
    file_prefix = log_filename.split(".messages.json")[0]

    if not os.path.exists(hindsight_dir):
        os.makedirs(hindsight_dir)

    # read latest logs in json file
    with open(args.log_path, 'r') as f:
        new_logs_list = json.load(f)

    # remove 1st entry in json list, which is the system prompt
    new_logs_list = new_logs_list[1:]
    new_logs = json.dumps(new_logs_list, indent=2)

    if args.history == "full":
        # Use sliding window history management
        windowed_logs = manage_sliding_window_history(hindsight_dir, new_logs, args.history_window_size, file_prefix)
    else:
        windowed_logs = new_logs

    try:
        with open(os.path.join(hindsight_dir, "latest_hindsight.txt"), 'r') as f:
            previous_reflection = f.read()
    except FileNotFoundError:
        previous_reflection = ""


    if args.algo == "awm":
        # AWM can use windowed logs for better context when history is full
        logs_to_use = windowed_logs if args.history == "full" else new_logs
        inp = LOGS.replace("##INTERACTIONS##", logs_to_use).replace("##REFLECTION##", previous_reflection)
        response = get_response_from_gpt_azure(
            inp,
            model=model,
            system_message=AWM_PROMPT,
        )
        out = str(response.get("summary", ""))
        out += "\n\n # Interaction summary # \n" + previous_reflection

    elif args.algo == "reflexion":
        # Use the reflection content to generate a new reflection
        # reflexion uses the windowed cumulative logs when history is full, otherwise just new logs
        logs_to_use = windowed_logs if args.history == "full" else new_logs
        inp = LOGS.replace("##INTERACTIONS##", logs_to_use).replace("##REFLECTION##", previous_reflection)
        response = get_response_from_gpt_azure(
            inp,
            model=model,
            system_message=REFLEXION_PROMPT,
        )
        out = str(response.get("reflection", ""))
        
    elif args.algo == "echo":
        # ONE_TRAJ_ONLY: Simple method that takes only trajectory and outputs counterfactual using ECHO_PROMPT
        out = one_traj_only(new_logs, previous_reflection)
        
    else:
        # For baseline or any other algorithm, set out to empty string
        out = ""


    if args.algo != "baseline":
        with open(os.path.join(hindsight_dir, f"{file_prefix}.hindsight.txt"), 'w') as f:
            f.write(out)

        with open(os.path.join(hindsight_dir, "latest_hindsight.txt"), 'w') as f:
            f.write(out)

    print("Offline computation finished.")


if __name__ == "__main__":
    main()
