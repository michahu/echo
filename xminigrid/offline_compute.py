import argparse
import json
import os
import re
from collections import deque

from prompts import generate_prompt
from api_call import get_response_from_gpt, get_response_from_gpt_azure


reflexion_instruction = """You are an agent in a 2D text-based environment. Reflect on your performance in the following episode and write some concise notes on how you can improve your performance in the next episodes. Reply with the following JSON format: {"reflection": X}
where X is your reflection. Ensure X is a parsable string!
"""

awm_instruction = """You are an agent in a 2D text-based environment. If the agent succeeds at accomplishing the given goal in the episode, convert the actions done in the following episode into abstract summary workflow. Discuss in high-level terms the steps a future agent should take to reach the goal. Include potential obstacles and landmarks in your workflow explanation.

Reply with the following JSON format: {"goal": "X", "workflow": Y} where X is the achieved goal and Y is your summary workflow. Ensure X and Y are parsable strings!

If the agent did not achieve the goal, then make Y an empty string.
"""

# single objective
echo_instruction = """You are a helpful teacher of agents navigating a 2D text-based environment. Given this summary, write new goals and trajectories to achieve these goals. The agent's goal is always to pick up a specific object.

Using the following trajectory, your job is to generate:

1. A new goal (e.g., "Pick up the red hexagon").
2. A **workflow of abstract actions** needed to achieve the goal.

The workflow should:
- Be **specific to the environment** in the trajectory. Avoid vague or generic phrases like "move toward the goal" or "open doors" unless referring to **specific objects or locations**.
- Use **environment-relevant abstractions**, like "navigate to the blue door".
- Begin from the agent's starting location, which is fixed.
- Avoid low-level actions in favor of high-level workflows.

Format your answer as:
```json
{
  "goal": "Pick up X",
  "workflow": "To pick up X: Step 1, Step 2..."
}"""

goal_instruction = """You are an expert at analyzing 2D text-based environments to identify potential agent objectives. Given a trajectory summary, extract all possible goals an agent could pursue. The agent's goal will always be to pick up a specific object.

## Task:
Identify all objects that could serve as pickup targets based on the environmental context shown in the summary.

## Requirements:
- **Extract specific objects** mentioned in the trajectory
- Avoid locations or non-portable objects

## Output Format:
```json
{
  "possible_goals": [
    "Pick up the [object1]",
    "Pick up the [object2]",
    "Pick up the [object3]"
  ]
}
```
"""

workflow_instruction = """You are an expert at creating action plans for agents in 2D text-based environments. Given a specific goal and a summary of a previous agent's actions, create a high-level workflow to achieve the goal.

## Task:
Design an abstract workflow for accomplishing the given goal using the environmental features from the trajectory summary.

## Requirements:
- **Environment-specific actions only**: reference actual locations, objects, or features from the summary
- Use high-level abstractions (e.g., "navigate to the blue door")
- **Avoid generic phrases** like "move toward goal" or "find the object"
- Start from the agent's known starting location
- Focus on strategic phases, not individual actions

## Output Format:
```json
{
  "goal": "[provided goal]",
  "workflow": "Step 1: [specific environment action]. Step 2: [specific environment action]. Step 3: [etc.]"
}
```
"""

trajectory_template = """## Summary of Agent's Actions
{summary}"""

summarize_instruction = """You are an expert at analyzing agent behavior in 2D text-based environments. Create a concise, high-level summary of the agent's trajectory.

## Instructions:

**What to Include:**
- Group low-level actions into high-level behaviors (e.g., "explored northern corridor" not individual moves)
- **All** objects discovered
- Completed objectives

**What to Exclude:**
- Individual movement steps, redundant actions, minor environmental details

**Format:** Chronological entries representing distinct phases or achievements

## Output Format:
```json
{
  "0": "Agent spawned in [location] and observed [key objects/features]",
  "1": "Agent navigated to [destination] and discovered [important findings]",
  "2": "Agent interacted with [object/entity] resulting in [outcome]",
  ...
}
```
"""


def extract_trial_id(trial_file):
    """Extract the last integer from the trial file name."""
    matches = re.findall(r"\d+", trial_file)
    if matches:
        return matches[-1]
    return "0"


def save_result(run_dir, mode, trial_id, result):
    """Save the result with hindsight_{id} naming convention."""
    filename = f"hindsight_{trial_id}.json"
    filepath = os.path.join(run_dir, filename)

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved {mode} result to {filepath}")
    return filepath


def perform_reflection(trial_data, llm_func):
    """
    Performs reflection on a single trial's data.
    """
    obs_queue = deque(trial_data["obs_queue"])
    acts_queue = deque(trial_data["acts_queue"])
    timestep = trial_data["timestep"]
    goal = trial_data["goal"]
    model = trial_data["model"]

    try:
        prompt = generate_prompt(
            goal=goal,
            deque_obs=obs_queue,
            deque_actions=acts_queue,
            timestep=timestep,
            last_k=len(obs_queue),
        )
        messages = [
            {"role": "system", "content": reflexion_instruction},
            {"role": "user", "content": prompt},
        ]

        reflection, _, _ = llm_func(
            model,
            messages,
        )
        return reflection

    except Exception as e:
        print(f"Error during reflection: {e}")
        return None


def perform_awm(trial_data, llm_func):
    """
    Performs reflection on a single trial's data.
    """
    obs_queue = deque(trial_data["obs_queue"])
    acts_queue = deque(trial_data["acts_queue"])
    timestep = trial_data["timestep"]
    goal = trial_data["goal"]
    model = trial_data["model"]

    try:
        prompt = generate_prompt(
            goal=goal,
            deque_obs=obs_queue,
            deque_actions=acts_queue,
            timestep=timestep,
            last_k=len(obs_queue),
        )
        messages = [
            {"role": "system", "content": awm_instruction},
            {"role": "user", "content": prompt},
        ]

        workflow, _, _ = llm_func(
            model,
            messages,
        )
        return workflow

    except Exception as e:
        print(f"Error during AWM: {e}")
        return None


def perform_echo(trial_data, llm_func):
    """
    Performs Goal-Conditioned Experience generation with 3 steps:
    1. Summarize the trajectory
    2. Extract goals from the summary
    3. Generate workflow to achieve the goal
    """
    obs_queue = deque(trial_data["obs_queue"])
    acts_queue = deque(trial_data["acts_queue"])
    timestep = trial_data["timestep"]
    goal = trial_data["goal"]
    model = trial_data["model"]

    try:
        # Step 1: Generate trajectory prompt
        prompt = generate_prompt(
            goal=goal,
            deque_obs=obs_queue,
            deque_actions=acts_queue,
            timestep=timestep,
            last_k=len(obs_queue),
        )

        # Step 1: Summarize the trajectory
        summarize_messages = [
            {"role": "system", "content": summarize_instruction},
            {"role": "user", "content": prompt},
        ]
        summary_response, _, _ = llm_func(model, summarize_messages)

        # Step 2: Extract goals from the summary
        summary_text = trajectory_template.format(
            summary=json.dumps(summary_response, indent=2)
        )
        goal_messages = [
            {"role": "system", "content": goal_instruction},
            {"role": "user", "content": summary_text},
        ]
        goals_response, _, _ = llm_func(model, goal_messages)

        # Step 3: Generate workflows for all extracted goals
        goal_workflow_pairs = []

        if (
            goals_response.get("possible_goals")
            and len(goals_response["possible_goals"]) > 0
        ):
            for goal in goals_response["possible_goals"]:
                workflow_messages = [
                    {"role": "system", "content": workflow_instruction},
                    {"role": "user", "content": f"Goal: {goal}\n\n{summary_text}"},
                ]
                workflow_response, _, _ = llm_func(model, workflow_messages)

                goal_workflow_pairs.append(
                    {"goal": goal, "workflow": workflow_response["workflow"]}
                )

        # Return all the data including the formatted goal-workflow pairs for backward compatibility
        result = {
            "summary": summary_response,
            "possible_goals": goals_response.get("possible_goals", []),
            "goal_workflow_pairs": goal_workflow_pairs,
        }

        return result

    except Exception as e:
        print(f"Error during ECHO: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Directory containing the agent's run data.",
    )
    parser.add_argument(
        "--trial-file",
        type=str,
        required=True,
        help="Name of the trial data file (e.g., results_0.json).",
    )
    parser.add_argument(
        "--llm-backend",
        type=str,
        default="default",
        choices=["default", "oai"],
        help="Which LLM backend to use: 'default' or 'azure'",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="reflexion",
        choices=["reflexion", "awm", "awmpp", "echo"],
        help="Mode to run: 'reflexion' for reflection, 'awm' for Abstract Workflow Memory, 'awmpp' for AWM with shortest path filtering, or 'echo' for goal-conditioned experience",
    )
    args = parser.parse_args()

    trial_file_path = os.path.join(args.run_dir, args.trial_file)
    trial_id = extract_trial_id(args.trial_file)

    with open(trial_file_path, "r") as f:
        trial_data = json.load(f)

    if args.llm_backend == "default":
        llm_func = get_response_from_gpt
    elif args.llm_backend == "oai":
        llm_func = get_response_from_gpt_azure
    else:
        raise ValueError(f"Unknown LLM backend: {args.llm_backend}")

    if args.mode == "reflexion":
        result = perform_reflection(trial_data, llm_func)
        print(f"Generated reflection: {result}")
        if result:
            save_result(args.run_dir, "reflexion", trial_id, result)

    elif args.mode == "awm":
        result = perform_awm(trial_data, llm_func)
        print(f"Generated workflow: {result}")
        if result:
            save_result(args.run_dir, "awm", trial_id, result)

    elif args.mode == "awmpp":
        result = perform_awm(trial_data, llm_func)
        print(f"Generated workflow: {result}")
        if result:
            save_result(args.run_dir, "awmpp", trial_id, result)

    elif args.mode == "echo":
        result = perform_echo(trial_data, llm_func)
        print(f"Generated ECHO: {result}")
        if result:
            save_result(args.run_dir, "echo", trial_id, result)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
