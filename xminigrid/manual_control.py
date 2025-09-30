import argparse
import xminigrid
from xminigrid.rendering.text_render import print_ruleset
from xminigrid.manual_control import ManualControl
from utils import generate_new_goal, _text_encode_goal, obs_to_text
import copy

import jax


class PersistentManualControl(ManualControl):
    def __init__(self, initial_timestep, only_pickup_goals=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_timestep = initial_timestep
        self.only_pickup_goals = only_pickup_goals

    def reset(self, env_params=None):
        goal_encoding = generate_new_goal(
            ruleset, only_pickup_goals=self.only_pickup_goals
        )
        timestep = copy.deepcopy(self.initial_timestep)
        timestep = timestep.replace(
            state=timestep.state.replace(goal_encoding=goal_encoding)
        )
        print(_text_encode_goal(goal_encoding.tolist()))

        infos = obs_to_text(timestep.observation, timestep.state)
        print(infos)

        self.timestep = timestep
        self.render()
        print(
            f"Step: {self.timestep.state.step_num} |",
            f"StepType: {self.timestep.step_type} |",
            f"Discount: {self.timestep.discount} |",
            f"Reward: {self.timestep.reward}",
        )

    def step(self, action):
        super().step(action)
        infos = obs_to_text(self.timestep.observation, self.timestep.state)
        print(infos)

        print(
            f"Step: {self.timestep.state.step_num} |",
            f"StepType: {self.timestep.step_type} |",
            f"Discount: {self.timestep.discount} |",
            f"Reward: {self.timestep.reward}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        type=str,
        default="XLand-MiniGrid-R1-9x9",
        choices=xminigrid.registered_environments(),
    )
    parser.add_argument(
        "--benchmark-id",
        type=str,
        default="trivial-1m",
        choices=xminigrid.registered_benchmarks(),
    )
    parser.add_argument("--ruleset-id", type=int, default=0)
    parser.add_argument("--agent-view", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-path", type=str, default=".")
    parser.add_argument(
        "--video-format", type=str, default=".mp4", choices=(".mp4", ".gif")
    )
    parser.add_argument("--video-fps", type=int, default=5)
    parser.add_argument(
        "--only-pickup-goals",
        action="store_true",
        default=False,
        help="If set, only 'pick up' goals will be generated.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the environment. Set to 0 for a random seed.",
    )

    args = parser.parse_args()
    env, env_params = xminigrid.make(args.env_id)

    if args.agent_view:
        from xminigrid.experimental.img_obs import RGBImgObservationWrapper

        env = RGBImgObservationWrapper(env)

    assert "XLand-MiniGrid" in args.env_id, "This script is for XLand-MiniGrid only."
    bench = xminigrid.load_benchmark(args.benchmark_id)
    ruleset = bench.get_ruleset(args.ruleset_id)
    env_params = env_params.replace(ruleset=ruleset)

    goal_encoding = generate_new_goal(ruleset, only_pickup_goals=args.only_pickup_goals)
    new_ruleset = ruleset.replace(goal=goal_encoding)
    env_params = env_params.replace(ruleset=new_ruleset)

    # change env_params.view_size to 10
    env_params = env_params.replace(view_size=13)
    assert env_params.view_size % 2 == 1, "view_size must be odd for the agent view."
    timestep = env.reset(env_params, jax.random.key(args.seed))

    control = PersistentManualControl(
        initial_timestep=timestep,
        env=env,
        env_params=env_params,
        agent_view=args.agent_view,
        save_video=args.save_video,
        video_path=args.video_path,
        video_format=args.video_format,
        video_fps=args.video_fps,
        only_pickup_goals=args.only_pickup_goals,
    )
    control.start()
