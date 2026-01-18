"""
Replay LIBERO demos from an HDF5 file and visualize with a GUI, while printing
subtask completions.

Usage:
  python scripts/replay_with_subtasks.py --dataset path/to/demo.hdf5 --demo 0
  python scripts/replay_with_subtasks.py --dataset-dir path/to/folder --demo 0
"""

import argparse
import os

import h5py
import numpy as np

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import TASK_MAPPING
import label_subtask_utils as utils


def build_env(problem_name, task_bddl_path, data_group, render_camera="agentview"):
    env_kwargs = utils.load_env_kwargs(data_group)
    if "robots" not in env_kwargs:
        env_kwargs["robots"] = ["Panda"]
    if "controller_configs" not in env_kwargs:
        env_kwargs["controller_configs"] = utils.load_controller_config(
            default_controller="OSC_POSE", robots=env_kwargs["robots"]
        )
    env_kwargs["controller_configs"] = utils.normalize_controller_configs(
        env_kwargs.get("controller_configs"), env_kwargs["robots"]
    )

    env_kwargs.setdefault("has_renderer", True)
    env_kwargs.setdefault("render_camera", render_camera)
    env_kwargs.setdefault("has_offscreen_renderer", False)
    env_kwargs.setdefault("ignore_done", True)
    env_kwargs.setdefault("use_camera_obs", False)
    env_kwargs.setdefault("reward_shaping", False)
    env_kwargs.setdefault("control_freq", 20)
    env_kwargs["bddl_file_name"] = task_bddl_path

    env_kwargs = utils.filter_env_kwargs(env_kwargs, TASK_MAPPING[problem_name])
    return TASK_MAPPING[problem_name](**env_kwargs)


def replay_demo(env, actions, states, goal_state):
    subtask_history = []
    active_subtasks = set()

    env.reset()
    if len(states) > 0:
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()

    for t, action in enumerate(actions):
        env.step(action)

        if len(states) > 0:
            if len(states) == len(actions) + 1:
                state_index = min(t + 1, len(states) - 1)
            else:
                state_index = min(t, len(states) - 1)
            env.sim.set_state_from_flattened(states[state_index])
            env.sim.forward()

        for predicate in goal_state:
            predicate_str = str(predicate)
            if predicate_str in active_subtasks:
                continue
            try:
                if env._eval_predicate(predicate):
                    print(f"[Step {t}] Subtask Completed: {predicate_str}")
                    subtask_history.append((t, predicate_str))
                    active_subtasks.add(predicate_str)
            except Exception:
                continue
    return subtask_history


def process_demo(dataset_path, demo_key, render_camera="agentview"):
    benchmark_dict = benchmark.get_benchmark_dict()
    # try to infer benchmark name from path
    benchmark_name = "libero_10"
    task_suite = benchmark_dict[benchmark_name]()

    with h5py.File(dataset_path, "r") as f:
        data_group = f["data"]
        demo_keys = utils.iter_demo_keys(data_group, demo_key)
        if not demo_keys:
            print("No demos found.")
            return

        demo_to_process = demo_keys[0]
        if demo_to_process not in data_group:
            print(f"Demo {demo_to_process} not found.")
            return

        bddl_file_name = utils.get_demo_bddl_name(data_group, demo_to_process)
        task_bddl_path = utils.resolve_bddl_path(
            bddl_file_name, dataset_path, benchmark_name, task_suite, get_libero_path
        )
        if task_bddl_path is None or not os.path.exists(task_bddl_path):
            print("Could not resolve BDDL path.")
            return

        problem_info = BDDLUtils.get_problem_info(task_bddl_path)
        problem_name = problem_info["problem_name"]
        language_instruction = problem_info.get("language_instruction", "")

        actions = data_group[f"{demo_to_process}/actions"][()]
        states = data_group[f"{demo_to_process}/states"][()] if "states" in data_group[demo_to_process] else np.empty((0,))

        env = build_env(problem_name, task_bddl_path, data_group, render_camera=render_camera)
        try:
            goal_state = env.parsed_problem["goal_state"]
            print(f"Replaying demo {demo_to_process} with {len(actions)} steps")
            subtask_history = replay_demo(env, actions, states, goal_state)
        finally:
            env.close()

        print("\n--- Summary ---")
        print(f"Language Instruction: {language_instruction}")
        for step, name in subtask_history:
            print(f"Step {step}: {name}")


def process_directory(dataset_dir, demo_key, render_camera="agentview"):
    dataset_dir = os.path.abspath(dataset_dir)
    h5_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith((".hdf5", ".h5"))]
    h5_files.sort()
    if not h5_files:
        print(f"No HDF5 files found under {dataset_dir}")
        return

    for path in h5_files:
        print(f"\n=== {path} ===")
        process_demo(path, demo_key, render_camera=render_camera)


def main():
    parser = argparse.ArgumentParser(description="Replay LIBERO demos with GUI and print subtask completions.")
    parser.add_argument("--dataset", type=str, help="Path to a single HDF5 dataset.")
    parser.add_argument("--dataset-dir", type=str, help="Directory with HDF5 datasets.")
    parser.add_argument("--demo", type=str, default="demo_0", help="Demo key or index (e.g., demo_0 or 0).")
    parser.add_argument("--camera", type=str, default="agentview", help="Render camera for GUI.")
    args = parser.parse_args()

    if args.dataset_dir:
        process_directory(args.dataset_dir, args.demo, render_camera=args.camera)
    elif args.dataset:
        process_demo(args.dataset, args.demo, render_camera=args.camera)
    else:
        print("Please provide --dataset or --dataset-dir")


if __name__ == "__main__":
    main()
