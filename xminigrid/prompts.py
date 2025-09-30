from utils import ACTION_LIST


env_description = """You are an agent in a 2D gridworld. At each step you will receive a list of valid and invalid actions. Choose a valid action by its index. Complete the goal in #HORIZON# steps."""

"""Rules:
- Cannot move through walls or objects
- Pick up/drop/open door: only works for items directly in front of you
- Carry limit: 1 item.
- Drop only works if the space 
"""

# env_description = f"""You are an agent in a 2D text-based environment. You can face north, south, east, or west.
# The possible actions are: {format_action_list()}
# Your aim is to complete the task described in the goal. You cannot move through objects or walls. Doors open to new rooms with potentially new objects, but you should thoroughly explore your current room first.

# At each step, you will receive a list of valid actions from the list above. You can only pick up objects in front of you, and you can only drop an object if the space in front of you is unoccupied. You can only open doors if they are in the space in front of you. You can only carry one thing at a time.

# You have #HORIZON# steps to complete the task.
"""

#  Here, an LLM agent is trying to perform a task in a 2D gridworld. Based on the information learned in this trajectory, what are some goals that can be achieved in this environment? Respond in the following json format:
# {'goals': [goal_str_0, goal_str_1, ...]}


# Do not add any generic information to learned_info. Your advice should be specific to actions in this particular environment.


#  Here, an LLM agent is trying to perform a task in a 2D gridworld. The gridworld is static, and objects in the environment will remain in the same place from episode to episode.

#  From the observations in this trajectory, a different goal might be to "pick up the red hexagon." Give a succint summary of how a new agent should achieve this goal in the environment. Give your answer in the following json format:
#  {'goal': 'pick up the red hexagon', 'summary': summary_of_gold_trajectory}

# In the trajectory below, an LLM agent is trying to perform a task in a 2D gridworld. The gridworld is static, and objects in the environment will remain in the same place from episode to episode.

# From the observations in this trajectory, a different goal might be to "pick up the red hexagon." Using the given trajectory, write a succinct counterfactual trajectory of how a new agent should achieve this new goal in the environment. Give your answer in the following json format: {'goal': 'pick up the red hexagon', 'summary': summary_of_gold_trajectory}

# The following is critical to understand: dropping an object places it in the square in front of you, which depends on the direction you are facing. For example, if you are in square (3, 3) facing north, and you drop an object, it will be placed in square (2, 3). Similarly, if you occupy square (3, 3), it is impossible to drop anything into that coordinate, because dropping an object places it in the square in front of you, not in your current square.

# Objects, including yourself, cannot occupy the same square at the same time.

# If you do not see an object in the room, try opening a closed door.
"""

json_instruction = """Reply with the following JSON format:
{"choice": X}
where X is the index of the desired choice. Ensure X is a parsable integer!
"""

react_json_instruction = """Reply concisely with following JSON format:
{"thought": X, "choice": Y} where X is your reasoning and Y is the index of the desired choice.
Ensure Y is a parseable integer!"""


default_system_message = (
    env_description
    + "\n\n"
    + """You will be prompted at each turn to choose actions."""
    + "\n\n"
    + json_instruction
)

react_system_message = (
    env_description
    + "\n\n"
    + """You will be prompted at each turn to first reason about your plan and then choose actions."""
    + "\n\n"
    + react_json_instruction
)


def generate_prompt(
    goal,
    deque_obs,
    deque_actions,
    timestep,
    subgoal=None,
    deque_thoughts=None,
    latest_valid=None,
    last_k=5,
):
    if deque_thoughts is not None and len(deque_thoughts) > 0:
        assert len(deque_obs) - 1 == len(deque_thoughts)
    if len(deque_actions) > 0:
        assert len(deque_obs) - 1 == len(deque_actions)

    if last_k is not None:
        # Calculate how many observations to include
        ldo = min(last_k, len(deque_obs))
        # Calculate how many actions/thoughts to include (one less than observations)
        lda = min(ldo - 1, len(deque_actions))
        ldt = min(ldo - 1, len(deque_thoughts)) if deque_thoughts is not None else 0
    else:
        ldo = len(deque_obs)
        lda = len(deque_actions)
        ldt = len(deque_thoughts) if deque_thoughts is not None else 0

    prompt = f"Current goal: {goal}.\n"
    if subgoal is not None:
        prompt += f"Current subgoal: {subgoal}.\n"

    start_timestep = timestep - ldo + 1

    for i in range(ldo):
        # Index directly from the end of the deques
        obs_idx = len(deque_obs) - ldo + i
        prompt += f"Observation {start_timestep + i}: " + deque_obs[obs_idx] + "\n"

        if i < ldt and deque_thoughts is not None:
            thought_idx = len(deque_thoughts) - ldt + i
            prompt += f"Thought {start_timestep + i}: {deque_thoughts[thought_idx]}\n"

        if i < lda:
            act_idx = len(deque_actions) - lda + i
            action_num = deque_actions[act_idx]
            action_name = ACTION_LIST.get(action_num, f"unknown action {action_num}")
            prompt += f"Action {start_timestep + i}: {action_name}\n\n"

    if latest_valid is not None:
        prompt += f"Action space: {latest_valid}\n"
    return prompt


def generate_prompt_append(
    goal,
    base_prompt,
    deque_obs,
    deque_actions,
    timestep,
    deque_thoughts=None,
    latest_valid=None,
):
    if deque_thoughts is not None and len(deque_thoughts) > 0:
        assert len(deque_obs) - 1 == len(deque_thoughts)
    if len(deque_actions) > 0:
        assert len(deque_obs) - 1 == len(deque_actions)

    # Get the last elements to append
    new_obs = deque_obs[-1]
    prev_act = deque_actions[-1]
    prev_thought = (
        deque_thoughts[-1] if deque_thoughts and len(deque_thoughts) > 0 else None
    )

    prompt = f"Current goal: {goal}.\n"
    if prev_thought:
        prompt += f"Thought {timestep - 1}: {prev_thought}\n"
    action_name = ACTION_LIST.get(prev_act, f"unknown action {prev_act}")
    prompt += f"Action {timestep - 1}: {action_name}\n\n"
    prompt += f"Observation {timestep}: {new_obs}\n"
    if latest_valid is not None:
        prompt += f"Action space: {latest_valid}\n"
    return base_prompt + prompt
