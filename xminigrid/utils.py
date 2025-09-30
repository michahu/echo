import json
import random

import jax.numpy as jnp
from xminigrid.core.constants import PICKABLE

# Action mapping used across the codebase
ACTION_LIST = {
    0: "go forward",
    1: "turn right",
    2: "turn left",
    3: "pick up",
    4: "drop",
    5: "open door",
}

NUM_ACTIONS = len(ACTION_LIST)

COLOR_NAMES = {
    0: "white",  # Colors.EMPTY
    1: "red",  # Colors.RED
    2: "green",  # Colors.GREEN
    3: "blue",  # Colors.BLUE
    4: "purple",  # Colors.PURPLE
    5: "yellow",  # Colors.YELLOW
    6: "grey",  # Colors.GREY
    7: "black",  # Colors.BLACK
    8: "orange",  # Colors.ORANGE
    9: "white",  # Colors.WHITE
    10: "brown",  # Colors.BROWN
    11: "pink",  # Colors.PINK
}

TILE_NAMES = {
    0: "empty space",
    1: "floor",
    2: "wall",
    3: "ball",
    4: "square",
    5: "pyramid",
    6: "goal",
    7: "key",
    8: "locked door",
    9: "closed door",
    10: "open door",
    11: "hexagon",
    12: "star",
}


def generate_new_goal(ruleset, only_pickup_goals=False):
    """
    Replace the existing goal with a new one.
    Goal format: [goal, goal_entity1, goal_entity2]
    """
    # see https://github.com/dunnolab/xland-minigrid/blob/main/src/xminigrid/rendering/text_render.py
    # or _text_encode_goal()
    if only_pickup_goals:
        goal = 1
    else:
        goal = random.choice([1, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14])

    rule_tiles = ruleset.rules[:, -2:]
    possible_entities = jnp.concatenate([rule_tiles, ruleset.init_tiles], axis=0)
    # filter possible_entities: first value must be in PICKABLE
    possible_entities = possible_entities[
        jnp.isin(possible_entities[:, 0], PICKABLE)
    ].tolist()

    dtype = ruleset.rules.dtype

    if goal in [4, 7, 8, 9, 10]:
        # For goals that require two entities, randomly select two different entities
        idxs = random.sample(range(len(possible_entities)), 2)
        tilea = possible_entities[idxs[0]]
        tileb = possible_entities[idxs[1]]
        goal_encoding = jnp.array(
            [goal, tilea[0], tilea[1], tileb[0], tileb[1]], dtype=dtype
        )
    else:
        tile = random.choice(possible_entities)
        goal_encoding = jnp.array([goal, tile[0], tile[1], 0, 0], dtype=dtype)

    return goal_encoding


def _encode_tile(tile: list[int]) -> str:
    return f"{COLOR_NAMES[tile[1]]} {TILE_NAMES[tile[0]]}"


def _text_encode_goal(goal: list[int]) -> str:
    goal_id = goal[0]
    if goal_id == 1:
        return f"Pick up {_encode_tile(goal[1:3])}"
    elif goal_id == 3:
        return f"Stand next to {_encode_tile(goal[1:3])}"
    elif goal_id == 4:
        return f"Put {_encode_tile(goal[1:3])} next to {_encode_tile(goal[3:5])}"
    elif goal_id == 7:
        return (
            f"Put {_encode_tile(goal[1:3])} directly south of {_encode_tile(goal[3:5])}"
        )
    elif goal_id == 8:
        return (
            f"Put {_encode_tile(goal[1:3])} directly west of {_encode_tile(goal[3:5])}"
        )
    elif goal_id == 9:
        return (
            f"Put {_encode_tile(goal[1:3])} directly north of {_encode_tile(goal[3:5])}"
        )
    elif goal_id == 10:
        return (
            f"Put {_encode_tile(goal[1:3])} directly east of {_encode_tile(goal[3:5])}"
        )
    elif goal_id == 11:
        return f"Stand directly south of {_encode_tile(goal[1:3])}"
    elif goal_id == 12:
        return f"Stand directly west of {_encode_tile(goal[1:3])}"
    elif goal_id == 13:
        return f"Stand directly north of {_encode_tile(goal[1:3])}"
    elif goal_id == 14:
        return f"Stand directly east of {_encode_tile(goal[1:3])}"
    else:
        raise RuntimeError(f"Rendering: Unknown goal id: {goal_id}")


def obs_to_text(obs, state, see_through_walls=False):
    """
    Convert an egocentric gridworld observation to descriptive text.

    Args:
        obs: n x n x 2 JAX uint8 array where:
                - obs[:,:,0] contains tile types (using Tiles enum values)
                - obs[:,:,1] contains colors (using Colors enum values)
                - Agent is at position (n-1, n//2) facing forward (toward row 0)
        agent_state: object containing agent information including:
                - direction: int JAX array (0-3) where 0=north, 1=east, 2=south, 3=west
                - pocket: 2D array of shape (n, 2) with tile types and colors of carried items
        see_through_walls (bool): If False, objects occluded by walls are not described.

    Returns:
        str: Descriptive text of the observation including agent direction, carried items, and visible objects
    """
    agent_state = state.agent
    direction = agent_state.direction
    carrying = agent_state.pocket  # 2D array of shape (n, 2) with tile types and colors

    # Convert direction to string
    direction_names = {0: "north", 1: "east", 2: "south", 3: "west"}
    direction_int = int(direction)  # Convert JAX array to Python int
    facing_direction = direction_names.get(direction_int, "unknown")

    n = obs.shape[0]
    agent_row = n - 1
    agent_col = n // 2

    descriptions = []
    wall_distances: dict[str, int | None] = {"front": None, "left": None, "right": None}
    limits: dict[str, int | None] = {"front": None, "left": None, "right": None}

    # Simplified wall distance calculation
    # Loop forward
    for i in range(1, agent_row + 1):
        r = agent_row - i
        tile_type = int(obs[r, agent_col, 0])
        if tile_type == 2:  # WALL
            wall_distances["front"] = i
            limits["front"] = r
            break
        if tile_type == 9:  # CLOSED_DOOR
            limits["front"] = r
            break  # Stop if a closed door is hit

    # Loop left
    for i in range(1, agent_col + 1):
        c = agent_col - i
        tile_type = int(obs[agent_row, c, 0])
        if tile_type == 2:  # WALL
            wall_distances["left"] = i
            limits["left"] = c
            break
        if tile_type == 9:  # CLOSED_DOOR
            limits["left"] = c
            break

    # Loop right
    for i in range(1, n - agent_col):
        c = agent_col + i
        tile_type = int(obs[agent_row, c, 0])
        if tile_type == 2:  # WALL
            wall_distances["right"] = i
            limits["right"] = c
            break
        if tile_type == 9:  # CLOSED_DOOR
            limits["right"] = c
            break

    # Create a visibility map if we don't see through walls
    visibility_map = jnp.ones((n, n), dtype=bool)
    if not see_through_walls:
        visibility_map = jnp.zeros_like(visibility_map)
        # Agent's position is always visible.
        visibility_map = visibility_map.at[agent_row, agent_col].set(True)

        # Define visibility boundaries based on the calculated limits
        front_limit = limits["front"] if limits["front"] is not None else 0
        left_limit = limits["left"] if limits["left"] is not None else 0
        right_limit = limits["right"] if limits["right"] is not None else n - 1

        # Set all tiles within the bounding box to visible
        for r in range(front_limit, agent_row + 1):
            for c in range(left_limit, right_limit + 1):
                visibility_map = visibility_map.at[r, c].set(True)

    # Scan through the observation grid
    for row in range(n):
        for col in range(n):
            # Skip the agent's position
            if row == agent_row and col == agent_col:
                continue

            # If we can't see through walls, check visibility
            if not see_through_walls and not visibility_map[row, col]:
                continue

            tile_type = int(obs[row, col, 0])  # Convert JAX uint8 to Python int
            color = int(obs[row, col, 1])  # Convert JAX uint8 to Python int

            # Skip empty tiles and floor
            if tile_type in [0, 1, 2]:  # EMPTY, FLOOR, WALL
                continue

            # Calculate relative position from agent
            row_diff = row - agent_row  # negative = forward, positive = behind
            col_diff = col - agent_col  # negative = left, positive = right

            # Get object description
            color_name = COLOR_NAMES.get(color, "unknown color")
            tile_name = TILE_NAMES.get(tile_type, "unknown object")

            # Don't add color for empty/white objects or if color is the same as tile name
            if color == 0 or color == 9 or color_name == "white":
                obj_description = tile_name
            else:
                obj_description = f"{color_name} {tile_name}"

            # Build position description
            position_parts = []

            # Handle forward/backward
            if row_diff < 0:
                forward_dist = abs(row_diff)
                if forward_dist == 1:
                    position_parts.append("1 tile forward")
                else:
                    position_parts.append(f"{forward_dist} tiles forward")
            elif row_diff > 0:
                raise ValueError(
                    "Row difference should not be positive in this function."
                )

            # Handle left/right
            if col_diff < 0:
                left_dist = abs(col_diff)
                if left_dist == 1:
                    position_parts.append("1 tile to the left")
                else:
                    position_parts.append(f"{left_dist} tiles to the left")
            elif col_diff > 0:
                right_dist = col_diff
                if right_dist == 1:
                    position_parts.append("1 tile to the right")
                else:
                    position_parts.append(f"{right_dist} tiles to the right")

            # Combine position description
            if len(position_parts) == 0:
                position_desc = "at the same location"
            elif len(position_parts) == 1:
                position_desc = position_parts[0]
            else:
                position_desc = " and ".join(position_parts)

            description = f"The {obj_description} is {position_desc}"

            descriptions.append(description)

    # Create carrying description
    carrying_description = ""
    if carrying is not None and carrying.size > 0:
        tile_type = int(carrying[0])
        color = int(carrying[1])

        # Skip empty items
        if tile_type != 0:
            color_name = COLOR_NAMES.get(color, "unknown color")
            tile_name = TILE_NAMES.get(tile_type, "unknown object")

            if color == 0 or color == 9 or color_name == "white":
                item_description = tile_name
            else:
                item_description = f"{color_name} {tile_name}"
            carrying_description = f"You are carrying a {item_description}"

    # Add wall descriptions
    wall_descriptions = []

    if wall_distances["front"] is not None:
        dist = wall_distances["front"]
        wall_descriptions.append(
            f"Wall is {dist} tile{'s' if dist > 1 else ''} forward"
        )
    if wall_distances["left"] == 1 and wall_distances["right"] == 1:
        wall_descriptions.append("You are in a doorway")
    else:
        if wall_distances["left"] is not None:
            dist = wall_distances["left"]
            wall_descriptions.append(
                f"Wall is {dist} tile{'s' if dist > 1 else ''} to the left"
            )
        if wall_distances["right"] is not None:
            dist = wall_distances["right"]
            wall_descriptions.append(
                f"Wall is {dist} tile{'s' if dist > 1 else ''} to the right"
            )

    # Start with agent direction
    direction_description = f"You are facing {facing_direction}"

    # Combine all descriptions
    all_descriptions = [direction_description]
    if carrying_description:
        all_descriptions.append(carrying_description)
    all_descriptions.extend(wall_descriptions)
    all_descriptions.extend(descriptions)

    if len(all_descriptions) == 1:  # Only direction description
        return f"{direction_description}. You see nothing but empty space around you."

    return ". ".join(all_descriptions) + "."


def obs_to_text_cardinal(obs, state):
    """
    Convert an egocentric gridworld observation to descriptive text using cardinal directions.

    Args:
        obs: n x n x 2 JAX uint8 array where:
                - obs[:,:,0] contains tile types (using Tiles enum values)
                - obs[:,:,1] contains colors (using Colors enum values)
                - Agent is at position (n-1, n//2) facing forward (toward row 0)
        agent_state: object containing agent information including:
                - direction: int JAX array (0-3) where 0=north, 1=east, 2=south, 3=west
                - pocket: 2D array of shape (n, 2) with tile types and colors of carried items

    Returns:
        str: Descriptive text of the observation including agent direction, carried items,
            and visible objects with their cardinal positions.
    """
    agent_state = state.agent
    direction = agent_state.direction
    carrying = agent_state.pocket  # 2D array of shape (n, 2)

    # Convert direction to string
    direction_names = {0: "north", 1: "east", 2: "south", 3: "west"}
    direction_int = int(direction)  # Convert JAX array to Python int
    facing_direction = direction_names.get(direction_int, "unknown")

    n = obs.shape[0]
    agent_row = n - 1
    agent_col = n // 2

    descriptions = []
    # Use cardinal directions for tracking the closest walls
    wall_distances: dict[str, int | None] = {
        "north": None,
        "south": None,
        "east": None,
        "west": None,
    }

    # Scan through the observation grid
    for row in range(n):
        for col in range(n):
            # Skip the agent's position
            if row == agent_row and col == agent_col:
                continue

            tile_type = int(obs[row, col, 0])
            color = int(obs[row, col, 1])

            # Skip empty tiles and floor
            if tile_type == 0 or tile_type == 1:  # EMPTY or FLOOR
                continue

            # Egocentric offsets:
            # row_diff: negative is "forward", positive is "behind"
            # col_diff: negative is "left", positive is "right"
            row_diff = row - agent_row
            col_diff = col - agent_col

            # --- Start of Cardinal Transformation ---
            # Convert egocentric (row_diff, col_diff) to cardinal (north/south, east/west)
            north_south_diff = 0
            east_west_diff = 0

            if direction_int == 0:  # Facing North
                north_south_diff = row_diff
                east_west_diff = col_diff
            elif direction_int == 1:  # Facing East
                north_south_diff = col_diff
                east_west_diff = -row_diff
            elif direction_int == 2:  # Facing South
                north_south_diff = -row_diff
                east_west_diff = -col_diff
            elif direction_int == 3:  # Facing West
                north_south_diff = -col_diff
                east_west_diff = row_diff
            # --- End of Cardinal Transformation ---

            # Handle walls specially - track the closest wall in each cardinal direction
            if tile_type == 2:  # WALL
                # Check for walls on cardinal axes relative to the agent
                if east_west_diff == 0:  # Wall is on the North-South axis
                    dist = abs(north_south_diff)
                    if north_south_diff < 0:  # North
                        if (
                            wall_distances["north"] is None
                            or dist < wall_distances["north"]
                        ):
                            wall_distances["north"] = dist
                    else:  # South
                        if (
                            wall_distances["south"] is None
                            or dist < wall_distances["south"]
                        ):
                            wall_distances["south"] = dist
                elif north_south_diff == 0:  # Wall is on the East-West axis
                    dist = abs(east_west_diff)
                    if east_west_diff > 0:  # East
                        if (
                            wall_distances["east"] is None
                            or dist < wall_distances["east"]
                        ):
                            wall_distances["east"] = dist
                    else:  # West
                        if (
                            wall_distances["west"] is None
                            or dist < wall_distances["west"]
                        ):
                            wall_distances["west"] = dist
                continue  # Skip adding walls to regular object descriptions

            # Get object description (color and type)
            color_name = COLOR_NAMES.get(color, "unknown color")
            tile_name = TILE_NAMES.get(tile_type, "unknown object")

            if color == 0 or color_name == "white":
                obj_description = tile_name
            else:
                obj_description = f"{color_name} {tile_name}"

            # Build cardinal position description
            position_parts = []
            if north_south_diff < 0:
                dist = abs(north_south_diff)
                position_parts.append(f"{dist} tile{'s' if dist > 1 else ''} north")
            elif north_south_diff > 0:
                dist = north_south_diff
                position_parts.append(f"{dist} tile{'s' if dist > 1 else ''} south")

            if east_west_diff < 0:
                dist = abs(east_west_diff)
                position_parts.append(f"{dist} tile{'s' if dist > 1 else ''} west")
            elif east_west_diff > 0:
                dist = east_west_diff
                position_parts.append(f"{dist} tile{'s' if dist > 1 else ''} east")

            # Combine position description
            if position_parts:
                position_desc = " and ".join(position_parts)
                descriptions.append(f"There is a {obj_description} {position_desc}")

    # Create carrying description
    carrying_description = ""
    if carrying is not None and carrying.size > 0:
        tile_type = int(carrying[0])
        color = int(carrying[1])

        if tile_type != 0:  # Not carrying an empty item
            color_name = COLOR_NAMES.get(color, "unknown color")
            tile_name = TILE_NAMES.get(tile_type, "unknown object")
            item_description = (
                tile_name
                if color == 0 or color_name == "white"
                else f"{color_name} {tile_name}"
            )
            carrying_description = f"You are carrying a {item_description}"

    # Add wall descriptions from the cardinal perspective
    wall_descriptions = []
    for direction_name, dist in wall_distances.items():
        if dist is not None:
            wall_descriptions.append(
                f"There is a wall {dist} tile{'s' if dist > 1 else ''} to the {direction_name}"
            )

    # Combine all descriptions for the final output
    direction_description = f"You are facing {facing_direction}"
    all_descriptions = [direction_description]
    if carrying_description:
        all_descriptions.append(carrying_description)
    all_descriptions.extend(wall_descriptions)
    all_descriptions.extend(descriptions)

    # Handle the case where nothing is visible
    if len(all_descriptions) == 1:
        return f"{direction_description}. You see nothing but empty space."

    return ". ".join(all_descriptions) + "."


def obs_to_text_cardinal_fully_observable(obs, state):
    """
    Converts a fully observable gridworld to descriptive text using cardinal directions.

    This function describes the entire grid, with all object locations given in
    absolute cardinal directions (N, S, E, W) relative to the agent, as well as
    their absolute grid coordinates. The surrounding walls are described by their
    distance in each cardinal direction.

    Args:
        obs: This parameter is unused but maintained for API consistency.
        state: An object containing the full environment state, including:
            - agent: An object with the agent's state:
                - position: A tuple (row, col) of the agent's absolute position.
                - direction: An int (0-3) where 0=N, 1=E, 2=S, 3=W.
                - pocket: A tuple (tile_type, color) of the carried item.
            - grid: A H x W x 2 numpy array representing the full grid.

    Returns:
        A string describing the full environment using cardinal directions and coordinates.
    """
    agent_state = state.agent
    grid = state.grid
    agent_pos = agent_state.position
    direction = agent_state.direction
    carrying = agent_state.pocket

    # --- Agent's Status ---
    direction_names = {0: "north", 1: "east", 2: "south", 3: "west"}
    facing_direction = direction_names.get(int(direction), "unknown")

    initial_description = f"You are at location {int(agent_pos[0]), int(agent_pos[1])} facing {facing_direction}"

    carrying_description = ""
    if carrying is not None and int(carrying[0]) != 0:
        tile_type, color = int(carrying[0]), int(carrying[1])
        if tile_type != 0:  # Don't describe carrying an empty item
            color_name = COLOR_NAMES.get(color, "unknown color")
            tile_name = TILE_NAMES.get(tile_type, "unknown object")

            if color in (0, 9) or color_name == "white":
                item_description = tile_name
            else:
                item_description = f"{color_name} {tile_name}"
            carrying_description = f"You are carrying a {item_description}"

    # --- Wall Descriptions ---
    grid_height, grid_width = grid.shape[:2]
    dist_north = agent_pos[0]
    dist_south = (grid_height - 1) - agent_pos[0]
    dist_west = agent_pos[1]
    dist_east = (grid_width - 1) - agent_pos[1]

    wall_descriptions = [
        f"There is a wall {dist_north} tile{'s' if dist_north != 1 else ''} to the north",
        f"There is a wall {dist_south} tile{'s' if dist_south != 1 else ''} to the south",
        f"There is a wall {dist_west} tile{'s' if dist_west != 1 else ''} to the west",
        f"There is a wall {dist_east} tile{'s' if dist_east != 1 else ''} to the east",
    ]

    # --- Grid Object Descriptions ---
    object_descriptions = []
    for r in range(grid_height):
        for c in range(grid_width):
            if (r, c) == tuple(agent_pos):
                continue

            tile_type = int(grid[r, c, 0])
            color = int(grid[r, c, 1])

            # Skip empty, floor, or wall tiles
            if tile_type in (0, 1, 2):
                continue

            # --- Calculate Cardinal Position ---
            north_south_diff = agent_pos[0] - r
            east_west_diff = c - agent_pos[1]

            # --- Build Description String ---
            color_name = COLOR_NAMES.get(color, "unknown color")
            tile_name = TILE_NAMES.get(tile_type, "unknown object")

            if color in (0, 9) or color_name == "white":
                obj_description = tile_name
            else:
                obj_description = f"{color_name} {tile_name}"

            position_parts = []
            if north_south_diff > 0:
                position_parts.append(
                    f"{north_south_diff} tile{'s' if north_south_diff > 1 else ''} north"
                )
            elif north_south_diff < 0:
                dist = abs(north_south_diff)
                position_parts.append(f"{dist} tile{'s' if dist > 1 else ''} south")

            if east_west_diff > 0:
                position_parts.append(
                    f"{east_west_diff} tile{'s' if east_west_diff > 1 else ''} east"
                )
            elif east_west_diff < 0:
                dist = abs(east_west_diff)
                position_parts.append(f"{dist} tile{'s' if dist > 1 else ''} west")

            if not position_parts:
                continue

            position_desc = " and ".join(position_parts)
            # Add the object's absolute coordinates to its description
            object_descriptions.append(
                f"There is a {obj_description} at location {(r, c)}, which is {position_desc}"
            )

    # --- Combine all parts into a final description ---
    all_descriptions = [initial_description]
    if carrying_description:
        all_descriptions.append(carrying_description)

    all_descriptions.extend(wall_descriptions)
    all_descriptions.extend(object_descriptions)

    if not object_descriptions and not carrying_description:
        # If only position and walls are described
        base_desc = f"{initial_description}. " + ". ".join(wall_descriptions)
        return f"{base_desc}. You see nothing else of interest."

    return ". ".join(all_descriptions) + "."


def generate_valid_actions_description(obs, state):
    """
    Generate a compact description of valid and invalid actions based on the current observation and agent state.

    Args:
        obs: n x n x 2 JAX uint8 array where:
                - obs[:,:,0] contains tile types (using Tiles enum values)
                - obs[:,:,1] contains colors (using Colors enum values)
                - Agent is at position (n-1, n//2) facing forward (toward row 0)
        state: object containing agent information including:
                - direction: int JAX array (0-3) where 0=north, 1=east, 2=south, 3=west
                - pocket: 2D array of shape (n, 2) with tile types and colors of carried items

    Returns:
        str: String describing valid and invalid actions with their reasons in a compact format.
    """

    agent_state = state.agent
    carrying = agent_state.pocket  # 2D array of shape (n, 2)

    n = obs.shape[0]
    agent_row = n - 1
    agent_col = n // 2

    # The tile directly in front of the agent
    front_row = agent_row - 1  # Agent faces toward row 0
    front_col = agent_col

    valid_actions = {}
    invalid_actions = {}

    # Check if the agent is carrying something
    is_carrying = carrying is not None and carrying.size > 0 and int(carrying[0]) != 0

    # Get the tile type and color in front of the agent
    if front_row >= 0:  # Make sure we're not going out of bounds
        front_tile_type = int(obs[front_row, front_col, 0])
        front_tile_color = int(obs[front_row, front_col, 1])
    else:
        front_tile_type = 2  # Treat out of bounds as wall
        front_tile_color = 0

    # Action 0: "go forward"
    if front_tile_type in [0, 1, 10]:  # empty, floor, or open door
        valid_actions[0] = {
            "action": "go forward",
            "reason": "path is clear",
        }
    elif front_tile_type == 2:  # wall
        invalid_actions[0] = {
            "action": "go forward",
            "reason": "blocked by wall",
        }
    elif front_tile_type == 9:  # closed door
        invalid_actions[0] = {
            "action": "go forward",
            "reason": "blocked by closed door",
        }
    else:  # other objects
        color_name = COLOR_NAMES.get(front_tile_color, "unknown color")
        tile_name = TILE_NAMES.get(front_tile_type, "unknown object")
        if front_tile_color in (0, 9) or color_name == "white":
            obj_description = tile_name
        else:
            obj_description = f"{color_name} {tile_name}"
        invalid_actions[0] = {
            "action": "go forward",
            "reason": f"blocked by {obj_description}",
        }

    # Action 1: "turn right" - always valid
    valid_actions[1] = {
        "action": "turn right",
        "reason": "always possible",
    }

    # Action 2: "turn left" - always valid
    valid_actions[2] = {
        "action": "turn left",
        "reason": "always possible",
    }

    # Action 3: "pick up"
    if is_carrying:
        invalid_actions[3] = {
            "action": "pick up",
            "reason": "already carrying something",
        }
    elif front_tile_type in PICKABLE:
        color_name = COLOR_NAMES.get(front_tile_color, "unknown color")
        tile_name = TILE_NAMES.get(front_tile_type, "unknown object")
        if front_tile_color in (0, 9) or color_name == "white":
            obj_description = tile_name
        else:
            obj_description = f"{color_name} {tile_name}"
        valid_actions[3] = {
            "action": "pick up",
            "reason": f"can pick up the {obj_description}",
        }
    else:
        base_reason = ""
        if front_tile_type in [0, 1]:  # empty or floor
            base_reason = "nothing to pick up"
        elif front_tile_type == 2:  # wall
            base_reason = "cannot pick up wall"
        elif front_tile_type in [8, 9, 10]:  # doors
            base_reason = "cannot pick up door"
        else:
            color_name = COLOR_NAMES.get(front_tile_color, "unknown color")
            tile_name = TILE_NAMES.get(front_tile_type, "unknown object")
            if front_tile_color in (0, 9) or color_name == "white":
                obj_description = tile_name
            else:
                obj_description = f"{color_name} {tile_name}"
            base_reason = f"cannot pick up {obj_description}"

        invalid_actions[3] = {
            "action": "pick up",
            "reason": base_reason,
        }

    # Action 4: "drop"
    if is_carrying:
        if front_tile_type in [0, 1]:  # empty or floor
            tile_type = int(carrying[0])
            color = int(carrying[1])
            color_name = COLOR_NAMES.get(color, "unknown color")
            tile_name = TILE_NAMES.get(tile_type, "unknown object")
            if color in (0, 9) or color_name == "white":
                item_description = tile_name
            else:
                item_description = f"{color_name} {tile_name}"
            valid_actions[4] = {
                "action": "drop",
                "reason": f"can drop the {item_description}",
            }
        else:
            invalid_actions[4] = {
                "action": "drop",
                "reason": "front square is occupied",
            }
    else:
        invalid_actions[4] = {
            "action": "drop",
            "reason": "not carrying anything",
        }

    # Action 5: "open door"
    if front_tile_type == 9:  # closed door
        color_name = COLOR_NAMES.get(front_tile_color, "unknown color")
        if front_tile_color in (0, 9) or color_name == "white":
            door_description = "door"
        else:
            door_description = f"{color_name} door"
        valid_actions[5] = {
            "action": "open door",
            "reason": f"can open the {door_description}",
        }
    else:
        if front_tile_type in [0, 1]:  # empty or floor
            invalid_actions[5] = {
                "action": "open door",
                "reason": "no door in front",
            }
        elif front_tile_type == 10:  # open door
            invalid_actions[5] = {
                "action": "open door",
                "reason": "door is already open",
            }
        elif front_tile_type == 8:  # locked door
            invalid_actions[5] = {
                "action": "open door",
                "reason": "door is locked",
            }
        else:
            invalid_actions[5] = {
                "action": "open door",
                "reason": "no door in front",
            }

    # Format the output
    result = "VALID_ACTIONS={\n"
    for action_id, action_info in valid_actions.items():
        result += f"    {action_id}: {json.dumps(action_info)},\n"
    result += "}\n"

    result += "INVALID_ACTIONS={\n"
    for action_id, action_info in invalid_actions.items():
        result += f"    {action_id}: {json.dumps(action_info)},\n"
    result += "}"

    return result


def generate_valid_actions_simplified(obs, state):
    """
    Generate a simple JSON of valid actions based on the current observation and agent state.

    Args:
        obs: n x n x 2 JAX uint8 array where:
                - obs[:,:,0] contains tile types (using Tiles enum values)
                - obs[:,:,1] contains colors (using Colors enum values)
                - Agent is at position (n-1, n//2) facing forward (toward row 0)
        state: object containing agent information including:
                - direction: int JAX array (0-3) where 0=north, 1=east, 2=south, 3=west
                - pocket: 2D array of shape (n, 2) with tile types and colors of carried items

    Returns:
        str: JSON string mapping action IDs to action names for valid actions only.
    """

    agent_state = state.agent
    carrying = agent_state.pocket  # 2D array of shape (n, 2)

    n = obs.shape[0]
    agent_row = n - 1
    agent_col = n // 2

    # The tile directly in front of the agent
    front_row = agent_row - 1  # Agent faces toward row 0
    front_col = agent_col

    # Action names mapping
    action_names = ACTION_LIST

    valid_actions = {}

    # Check if the agent is carrying something
    is_carrying = carrying is not None and carrying.size > 0 and int(carrying[0]) != 0

    # Get the tile type in front of the agent
    if front_row >= 0:  # Make sure we're not going out of bounds
        front_tile_type = int(obs[front_row, front_col, 0])
    else:
        front_tile_type = 2  # Treat out of bounds as wall

    # Action 0: "go forward"
    if front_tile_type in [0, 1, 10]:  # empty, floor, or open door
        valid_actions[0] = action_names[0]

    # Action 1: "turn right" - always valid
    valid_actions[1] = action_names[1]

    # Action 2: "turn left" - always valid
    valid_actions[2] = action_names[2]

    # Action 3: "pick up"
    if front_tile_type in PICKABLE:
        valid_actions[3] = action_names[3]

    # Action 4: "drop"
    if is_carrying and front_tile_type in [0, 1]:  # empty or floor
        valid_actions[4] = action_names[4]

    # Action 5: "open door"
    if front_tile_type == 9:  # closed door
        valid_actions[5] = action_names[5]

    return json.dumps(valid_actions)
