import argparse
import copy
import imageio
import os
import time
import json
import random
from collections import OrderedDict, deque

import jax
import numpy as np
import xminigrid
from tqdm import trange

from utils import (
    _text_encode_goal,
    generate_new_goal,
    generate_valid_actions_simplified,
    ACTION_LIST,
    NUM_ACTIONS,
)
from prompts import (
    generate_prompt,
    generate_prompt_append,
    react_system_message,
    default_system_message,
)

import logging

logging.disable(logging.CRITICAL)


PRINT_LLM_DEBUG = False

action_list = ACTION_LIST
num_actions = NUM_ACTIONS


def random_action():
    """Function to generate a random action instead of using a RandomAgent class."""
    return np.random.choice(list(range(num_actions)))


class Agent:
    def __init__(
        self,
        args,
        horizon,
        llm_func=None,
        react=False,
        chat_history_len=100,
        prompt_mode="append",
        **llm_kwargs,
    ):
        # Base agent attributes
        self.goal = None
        self.obs_queue = deque([])
        self.acts_queue = deque([])
        self.total_cost = 0
        self.timestep = -1
        self.archive = OrderedDict()
        self.horizon = horizon
        self.remaining_actions = OrderedDict()

        self.args = args
        self.model = args.model
        self.save_dir = args.save_dir
        self.chat_history_len = args.chat_history_len
        self.seed = args.seed
        self.agent_type = args.agent_type
        self.hindsight_type = args.hindsight_type
        self.hindsight_k = args.hindsight_k

        # For echo and awmpp, use a dictionary indexed by goals instead of a deque
        if self.hindsight_type in ["echo", "awmpp"]:
            self.hindsights_dict = {}
            self.hindsights = []  # We'll populate this from the dictionary when needed
        else:
            self.hindsights = deque([], maxlen=self.hindsight_k)

        if self.save_dir:
            self.load_hindsights()

        self.react = react
        self.thoughts_queue = None
        if self.react:
            self.thoughts_queue = deque()
        self.llm_kwargs = llm_kwargs
        self.llm_func = llm_func
        self.prompt_mode = prompt_mode
        if self.prompt_mode == "append":
            self.prompt = ""
        self.make_system_message()

    def observe(self, goal, infos, latest_valid):
        self.timestep += 1
        self.goal = goal
        self.obs_queue.append(infos)
        self.latest_valid = latest_valid

    def choose_new_state(self):
        return next(iter(self.archive.values()))

    def reset_state(self, reset_reflexion=True):
        # by default select the first state
        self.obs_queue.clear()
        self.acts_queue.clear()
        self.timestep = -1

    def load_hindsights(self):
        """Load hindsight files from previous seeds."""
        if not self.save_dir:
            return

        # Load from previous seeds (count backwards)
        for i in range(max(0, self.seed - self.hindsight_k), self.seed):
            # Look for hindsight file in the previous seed's directory
            hindsight_file = os.path.join(self.save_dir, f"hindsight_{i}.json")
            if os.path.exists(hindsight_file):
                try:
                    with open(hindsight_file) as f:
                        hindsight_data = json.load(f)

                        if self.hindsight_type in ["echo", "awmpp"]:
                            # For echo and awmpp, process goal-workflow pairs into dictionary
                            if "goal_workflow_pairs" in hindsight_data:
                                # Echo format: list of goal-workflow pairs
                                for pair in hindsight_data["goal_workflow_pairs"]:
                                    goal = pair["goal"]
                                    workflow = pair["workflow"]

                                    # If this goal doesn't exist or the new workflow is shorter
                                    if goal not in self.hindsights_dict or len(
                                        workflow
                                    ) < len(self.hindsights_dict[goal]):
                                        self.hindsights_dict[goal] = workflow
                            elif (
                                "goal" in hindsight_data
                                and "workflow" in hindsight_data
                            ):
                                # AWM format: single goal-workflow pair
                                goal = hindsight_data["goal"]
                                workflow = hindsight_data["workflow"]

                                # Only add if workflow is not empty and goal doesn't exist or new workflow is shorter
                                if workflow and (
                                    goal not in self.hindsights_dict
                                    or len(workflow) < len(self.hindsights_dict[goal])
                                ):
                                    self.hindsights_dict[goal] = workflow

                            # We'll rebuild the list for system_message generation later
                        else:
                            # For other hindsight types, use the deque as before
                            self.hindsights.append(hindsight_data)
                except (json.JSONDecodeError, OSError) as e:
                    print(
                        f"Warning: Could not load hindsight from {hindsight_file}: {e}"
                    )

    def make_system_message(self):
        if not self.react:
            self.system_message = default_system_message.replace(
                "#HORIZON#", str(self.horizon)
            )
        else:
            self.system_message = react_system_message.replace(
                "#HORIZON#", str(self.horizon)
            )

        # Add hindsights from previous episodes
        if self.hindsight_type in ["echo", "awmpp"] and self.hindsights_dict:
            # First, convert dictionary to list for message generation
            self.hindsights = []
            goal_workflow_pairs = []

            # Convert dict to list format for consistent message generation
            for goal, workflow in self.hindsights_dict.items():
                goal_workflow_pairs.append({"goal": goal, "workflow": workflow})

            # Store the top k workflows (limit to hindsight_k)
            self.hindsights = [
                {"goal_workflow_pairs": goal_workflow_pairs[: self.hindsight_k]}
            ]

            # Add hindsights to system message
            self.system_message += "\n\nYou also have access to hindsight analysis from previous episodes, which provide valuable insights for improving your performance:"
            self.system_message += "\n\n## Previous Episodes Analysis ##\n"
            self.system_message += "Goal-Workflow Pairs:\n"

            for pair in goal_workflow_pairs[: self.hindsight_k]:
                self.system_message += f"  - Goal: {pair['goal']}\n"
                self.system_message += f"    Workflow: {pair['workflow']}\n"

        elif len(self.hindsights) > 0:
            self.system_message += "\n\nYou also have access to hindsight analysis from previous episodes, which provide valuable insights for improving your performance:"

            for i, hindsight in enumerate(self.hindsights):
                self.system_message += f"\n\n## Previous Episode Analysis {i + 1} ##\n"

                if self.hindsight_type == "reflexion":
                    self.system_message += f"Reflection: {hindsight['reflection']}\n"
                elif self.hindsight_type == "awm":
                    if hindsight.get("workflow", "") != "":
                        self.system_message += (
                            f"Goal: {hindsight['goal']}\n"
                            f"Abstract Workflow: {hindsight['workflow']}\n\n"
                        )
                elif self.hindsight_type in ["echo", "awmpp"]:
                    if "goal_workflow_pairs" in hindsight:
                        self.system_message += "Goal-Workflow Pairs:\n"
                        for pair in hindsight["goal_workflow_pairs"]:
                            self.system_message += f"  - Goal: {pair['goal']}\n"
                            self.system_message += f"    Workflow: {pair['workflow']}\n"

    def act(self):
        # If model is not set, this is a basic agent that needs to be overridden
        if not hasattr(self, "model") or self.model is None:
            raise NotImplementedError(
                "act() method must be overridden for non-LLM agents"
            )

        # Check if LLM function is available
        if self.llm_func is None:
            raise ValueError("llm_func is required for LLM agents")

        # LLM agent behavior
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.prompt_mode == "rebuild":
                    prompt = generate_prompt(
                        goal=self.goal,
                        deque_obs=self.obs_queue,
                        deque_actions=self.acts_queue,
                        latest_valid=self.latest_valid,
                        timestep=self.timestep,
                        deque_thoughts=self.thoughts_queue,
                        last_k=self.chat_history_len,
                    )
                elif self.prompt_mode == "append":
                    if self.timestep == 0:
                        self.prompt = generate_prompt(
                            goal=self.goal,
                            deque_obs=self.obs_queue,
                            deque_actions=self.acts_queue,
                            latest_valid=self.latest_valid,
                            timestep=self.timestep,
                            deque_thoughts=self.thoughts_queue,
                            last_k=self.chat_history_len,
                        )
                    else:
                        self.prompt = generate_prompt_append(
                            goal=self.goal,
                            base_prompt=self.prompt,
                            deque_obs=self.obs_queue,
                            deque_actions=self.acts_queue,
                            timestep=self.timestep,
                            deque_thoughts=self.thoughts_queue if self.react else None,
                            latest_valid=self.latest_valid,
                        )
                    prompt = self.prompt
                else:
                    raise ValueError(f"Unknown prompt mode: {self.prompt_mode}")

                messages = [
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt},
                ]

                response, _, _ = self.llm_func(
                    self.model,
                    messages=messages,
                    **self.llm_kwargs,
                )
                if self.react and self.thoughts_queue is not None:
                    self.thoughts_queue.append(response.get("thought", ""))
                action = response["choice"]

                if action < 0 or action >= num_actions:
                    raise ValueError("Invalid action index.")
                self.acts_queue.append(action)

                return action
            except Exception as e:
                print(f"Error: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(3)
                print(f"Retrying LLM call (attempt {attempt + 2}/{max_retries})...")
        raise RuntimeError("LLM call failed after retries.")


def evaluate_single_performance(
    args,
    env,
    env_params,
    initial_timestep,
    ruleset,
    agent,
    obs_to_text_func,
):
    goal_encoding = generate_new_goal(ruleset, only_pickup_goals=args.only_pickup_goals)
    new_ruleset = ruleset.replace(goal=goal_encoding)
    env_params = env_params.replace(ruleset=new_ruleset)

    timestep = copy.deepcopy(initial_timestep)
    timestep = timestep.replace(
        state=timestep.state.replace(goal_encoding=goal_encoding)
    )
    infos = obs_to_text_func(timestep.observation, timestep.state)
    valid_actions = generate_valid_actions_simplified(
        timestep.observation, timestep.state
    )
    goal = _text_encode_goal(goal_encoding.tolist())
    print(f"\n Goal: {goal}\n")
    print(f"Infos: {infos}\n")

    agent.observe(goal, infos, valid_actions)

    actions_taken = 0
    reward = 0
    images = []

    if args.save_movie:
        images.append(env.render(env_params, timestep))

    for _ in trange(agent.horizon):
        action = agent.act()
        timestep = env.step(env_params, timestep, action)
        infos = obs_to_text_func(timestep.observation, timestep.state)
        valid_actions = generate_valid_actions_simplified(
            timestep.observation, timestep.state
        )
        agent.observe(goal, infos, valid_actions)

        print(
            f"\nThoughts: {agent.thoughts_queue[-1] if hasattr(agent, 'react') and agent.react and agent.thoughts_queue else 'N/A'}"
        )
        print(f"Action taken: {action_list[action]}")
        print(f"Infos: {infos}")

        if args.save_movie:
            images.append(env.render(env_params, timestep))

        reward = timestep.reward.item()
        if reward > 0:
            break

        actions_taken += 1

    # Save data for offline processing
    trial_data = {
        "reward": reward,
        "goal": goal,
        "obs_queue": list(agent.obs_queue),
        "acts_queue": list(agent.acts_queue),
        "timestep": agent.timestep,
        "model": agent.model if hasattr(agent, "model") else None,
        "llm_kwargs": agent.llm_kwargs if hasattr(agent, "llm_kwargs") else None,
        "system_message": agent.system_message
        if hasattr(agent, "system_message")
        else None,
    }
    if hasattr(agent, "thoughts_queue"):
        trial_data["thoughts_queue"] = list(agent.thoughts_queue)

    with open(os.path.join(args.save_dir, f"results_{args.seed}.json"), "w") as f:
        json.dump(trial_data, f, indent=2)

    if args.save_movie:
        # Save the movie
        imageio.mimsave(
            os.path.join(args.save_dir, f"movie_{args.seed}.mp4"),
            images,
            fps=16,
            format="mp4",
        )
        print(f"Movie saved to {args.save_dir}")

    agent.reset_state()


def main():
    # python run.py --env-name "XLand-MiniGrid-R9-25x25" --model dev-gpt-4o-2024-05-13 --llm-backend default --obs-style text  --agent-type react --max-steps 32 --only-pickup-goals --save-movie

    #  python run.py --env-name "XLand-MiniGrid-R6-17x17" --model dev-gpt-4o-2024-05-13 --llm-backend default --obs-style text  --agent-type react --max-steps 64 --only-pickup-goals --save-movie --save-dir ./output/react-40-with-thoughts --chat-history-len 10

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="XLand-MiniGrid-R1-9x9")
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    parser.add_argument("--debug-mode", action="store_true", default=False)
    parser.add_argument("--save-dir", type=str, default="./output/")
    parser.add_argument("--model", type=str, default="gpt-4.1")
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument(
        "--benchmark-id",
        type=str,
        default="trivial-1m",
        choices=xminigrid.registered_benchmarks(),
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        default="random",
        choices=["random", "naive_llm", "react"],
        help="Agent type to run",
    )
    parser.add_argument(
        "--hindsight-type",
        type=str,
        default="echo",
        choices=["none", "echo", "awm", "awmpp", "reflexion"],
    )
    parser.add_argument("--ruleset-id", type=int, default=0)
    parser.add_argument(
        "--llm-backend",
        type=str,
        default="substrate",
        choices=["default", "vllm"],
        help="Which LLM backend to use: 'default', 'oai', or 'substrate'",
    )
    parser.add_argument(
        "--save-movie",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of trials to run for each agent",
    )
    parser.add_argument(
        "--only-pickup-goals",
        action="store_true",
        default=False,
        help="If set, only 'pick up' goals will be generated.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="low",
        choices=["low", "medium", "high"],
        help="Reasoning effort for o3-mini model.",
    )
    parser.add_argument(
        "--obs-style",
        type=str,
        default="text",
        choices=["text", "cardinal", "full_obs"],
        help="Type of observation encoding to use.",
    )
    parser.add_argument(
        "--chat-history-len",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--view-size",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--hindsight-k",
        type=int,
        default=5,
        help="Number of previous hindsight analyses to load (default: 5)",
    )
    parser.add_argument(
        "--env-id",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.obs_style == "text":
        from utils import obs_to_text as obs_to_text_func
    elif args.obs_style == "cardinal":
        from utils import obs_to_text_cardinal as obs_to_text_func
    elif args.obs_style == "full_obs":
        from utils import obs_to_text_cardinal_fully_observable as obs_to_text_func
    else:
        raise ValueError(f"Unknown observation style: {args.observation_style}")

    env, env_params = xminigrid.make(args.env_name)
    bench = xminigrid.load_benchmark(args.benchmark_id)
    ruleset = bench.get_ruleset(args.ruleset_id)
    env_params = env_params.replace(ruleset=ruleset).replace(view_size=args.view_size)
    # note: the env seed must always be zero to keep objects in the same place.
    timestep = env.reset(env_params, jax.random.key(args.env_id))

    if args.llm_backend == "default":
        from api_call import get_response_from_gpt_azure

        llm_func = get_response_from_gpt_azure
    elif args.llm_backend == "vllm":
        from api_call import get_response_from_local

        llm_func = get_response_from_local
    else:
        raise ValueError(f"Unknown LLM backend: {args.llm_backend}")

    llm_kwargs = {}
    if "qwen3" in args.model or "qwen-3" in args.model:
        llm_kwargs["is_thinking"] = True
    if args.model == "o3-mini":
        llm_kwargs["reasoning_effort"] = args.reasoning_effort

    if args.agent_type == "random":
        # Create a simple agent that uses the random_action function
        agent = Agent(args, horizon=args.max_steps)
        agent.act = lambda: random_action()
    else:
        agent = Agent(
            args,
            horizon=args.max_steps,
            llm_func=llm_func,
            react=(args.agent_type == "react"),
            chat_history_len=args.chat_history_len,
            **llm_kwargs,
        )

    print(f"env name: {args.env_name}, env step limit: {args.max_steps}")
    os.makedirs(args.save_dir, exist_ok=True)
    evaluate_single_performance(
        args,
        env=env,
        env_params=env_params,
        initial_timestep=timestep,
        ruleset=ruleset,
        agent=agent,
        obs_to_text_func=obs_to_text_func,
    )


if __name__ == "__main__":
    main()


# python run.py --env-name "XLand-MiniGrid-R2-17x17" --model dev-gpt-4o-2024-05-13 --llm-backend substrate_client --obs-style text  --agent-type react --max-steps 64 --only-pickup-goals --save-movie --save-dir ./output/react-4o-with-thoughts
