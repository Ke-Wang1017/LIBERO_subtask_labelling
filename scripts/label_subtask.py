import argparse
import os

import h5py
import numpy as np

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import TASK_MAPPING

import label_subtask_utils as utils

# --- CONFIGURATION ---
DATASET_PATH = "/home/ke/Documents/LIBERO/libero/datasets/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5"


def build_env(problem_name, task_bddl_path, data_group):
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

    env_kwargs.setdefault("has_renderer", False)
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


def process_demo(
    demo_to_process,
    data_group,
    dataset_path,
    task_suite,
    benchmark_name,
    output_dir,
    save_to_hdf5,
):
    print(f"--- Processing {demo_to_process} ---")

    bddl_file_name = utils.get_demo_bddl_name(data_group, demo_to_process)
    task_bddl_path = utils.resolve_bddl_path(
        bddl_file_name, dataset_path, benchmark_name, task_suite, get_libero_path
    )
    if task_bddl_path is None or not os.path.exists(task_bddl_path):
        print("Error: BDDL file could not be resolved for this demo.")
        return

    print(f"Task BDDL: {task_bddl_path}")
    problem_info = BDDLUtils.get_problem_info(task_bddl_path)
    problem_name = problem_info["problem_name"]
    language_instruction = problem_info.get("language_instruction", "")

    if "actions" not in data_group[demo_to_process]:
        print("Error: Actions not found for this demo.")
        return

    actions = data_group[f"{demo_to_process}/actions"][()]
    states = data_group[f"{demo_to_process}/states"][()] if "states" in data_group[demo_to_process] else np.empty((0,))

    env = build_env(problem_name, task_bddl_path, data_group)
    try:
        if not language_instruction:
            language_instruction = env.parsed_problem.get("language_instruction", "")
        goal_state = env.parsed_problem["goal_state"]

        print(f"Replaying {len(actions)} steps...")
        subtask_history = replay_demo(env, actions, states, goal_state)
    finally:
        env.close()

    print("\n--- Summary ---")
    print(f"Language Instruction: {language_instruction}")
    for step, name in subtask_history:
        print(f"Step {step}: {name}")

    natural_language_history = {
        (step, predicate_str): utils.predicate_to_text(predicate_str, language_instruction)
        for step, predicate_str in subtask_history
    }

    output_json_path = os.path.join(output_dir, f"{demo_to_process}_subtasks.json")
    utils.save_subtask_history(
        subtask_history=subtask_history,
        language_instruction=language_instruction,
        demo_key=demo_to_process,
        output_path=output_json_path,
        save_to_hdf5=save_to_hdf5,
        hdf5_file=data_group.file if save_to_hdf5 else None,
        demo_group=data_group[demo_to_process] if save_to_hdf5 else None,
        natural_language_history=natural_language_history,
    )


def replay_and_label(
    dataset_path,
    benchmark_name="libero_10",
    demo_key=None,
    max_demos=None,
    output_dir=None,
    save_to_hdf5=False,
):
    max_demos = None if (max_demos is not None and max_demos < 0) else max_demos
    benchmark_dict = benchmark.get_benchmark_dict()
    if benchmark_name not in benchmark_dict:
        print(f"Error: Unknown benchmark '{benchmark_name}'.")
        return
    task_suite = benchmark_dict[benchmark_name]()

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(dataset_path), "subtask_labels")
    os.makedirs(output_dir, exist_ok=True)

    file_mode = "r+" if save_to_hdf5 else "r"
    with h5py.File(dataset_path, file_mode) as f:
        data_group = f["data"]
        demo_keys = utils.iter_demo_keys(data_group, demo_key)
        if not demo_keys:
            print("Error: No demonstrations found in the dataset.")
            return
        if max_demos is not None and max_demos >= 0:
            demo_keys = demo_keys[:max_demos]

        print(f"  Found {len(demo_keys)} demos to process")
        processed = 0

        for demo_to_process in demo_keys:
            if demo_to_process not in data_group:
                print(f"Error: Demo '{demo_to_process}' not found in dataset.")
                continue
            process_demo(
                demo_to_process,
                data_group,
                dataset_path,
                task_suite,
                benchmark_name,
                output_dir,
                save_to_hdf5,
            )
            processed += 1

        print(f"  Completed {processed} demos for dataset: {dataset_path}")


def replay_directory(
    dataset_dir,
    benchmark_name="libero_10",
    demo_key=None,
    max_demos=None,
    output_dir=None,
    save_to_hdf5=False,
):
    max_demos = None if (max_demos is not None and max_demos < 0) else max_demos
    dataset_dir = os.path.abspath(dataset_dir)
    all_datasets = []
    for root, _, files in os.walk(dataset_dir):
        for fname in files:
            if fname.endswith(".hdf5") or fname.endswith(".h5"):
                all_datasets.append(os.path.join(root, fname))

    if not all_datasets:
        print(f"No dataset files (.hdf5 / .h5) found under {dataset_dir}")
        return

    all_datasets.sort()
    print(f"Found {len(all_datasets)} dataset files under {dataset_dir}")

    for path in all_datasets:
        relative = os.path.relpath(path, dataset_dir)
        per_file_output = output_dir
        if per_file_output is None:
            per_file_output = os.path.join(
                dataset_dir, "subtask_labels", os.path.splitext(relative)[0]
            )
        print(f"\nProcessing dataset: {path}")
        replay_and_label(
            dataset_path=path,
            benchmark_name=benchmark_name,
            demo_key=demo_key,
            max_demos=max_demos,
            output_dir=per_file_output,
            save_to_hdf5=save_to_hdf5,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_PATH, help="Path to HDF5 dataset.")
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Directory containing one or more HDF5 datasets to process.",
    )
    parser.add_argument(
        "--benchmark",
        default="libero_10",
        help="Benchmark name (e.g., libero_10, libero_90).",
    )
    parser.add_argument(
        "--demo",
        default=None,
        help="Demo key (e.g., demo_0) or index (e.g., 0).",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=None,
        help="Maximum number of demos to process when --demo is not set. Use -1 for all.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save subtask history JSON files. Default: <dataset_dir>/subtask_labels",
    )
    parser.add_argument(
        "--save-to-hdf5",
        action="store_true",
        help="Also save subtask history as attributes in the HDF5 file.",
    )
    args = parser.parse_args()
    if args.dataset_dir:
        replay_directory(
            args.dataset_dir,
            benchmark_name=args.benchmark,
            demo_key=args.demo,
            max_demos=args.max_demos,
            output_dir=args.output_dir,
            save_to_hdf5=args.save_to_hdf5,
        )
    else:
        replay_and_label(
            args.dataset,
            benchmark_name=args.benchmark,
            demo_key=args.demo,
            max_demos=args.max_demos,
            output_dir=args.output_dir,
            save_to_hdf5=args.save_to_hdf5,
        )
