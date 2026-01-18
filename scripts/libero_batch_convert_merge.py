"""
Convert LIBERO HDF5 demonstrations (with subtask labels saved by label_subtask.py)
into LeRobot datasets, attaching a subtask label for every frame.

Unlike the original merging script, this version converts each HDF5 file
independently and does not merge across tasks.

Example:
python scripts/libero_batch_convert_merge.py \\
  --input-dir libero/datasets/libero_10 \\
  --output-root ./lerobot_exports \\
  --repo-prefix libero_10_subtasks
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from tqdm import tqdm

# Ensure local lerobot package is discoverable
import sys
sys.path.append("/home/ke/Documents/lerobot_neuracore/lerobot/src")

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as exc:
    raise SystemExit(
        "LeRobot dependency missing. Ensure `datasets` (HF) and `lerobot` are installed in this env. "
        f"Original error: {exc}"
    )


def load_demo(h5: h5py.File, demo_key: str):
    data = h5["data"][demo_key]
    actions = np.array(data["actions"])
    # states = np.array(data["states"])
    obs = data["obs"]

    agentview = np.array(obs["agentview_rgb"])
    eye_in_hand = np.array(obs["eye_in_hand_rgb"])
    # Transpose images to CHW
    agentview = agentview.transpose(0, 3, 1, 2)
    eye_in_hand = eye_in_hand.transpose(0, 3, 1, 2)

    # Optional additional states (ee, joints, gripper)
    extras = {
        "ee_pos": np.array(obs["ee_pos"]),
        "ee_ori": np.array(obs["ee_ori"]),
        "ee_states": np.array(obs["ee_states"]),
        "joint_states": np.array(obs["joint_states"]),
        "gripper_states": np.array(obs["gripper_states"]),
    }
    states = np.concatenate([extras["ee_states"], extras["gripper_states"]], axis=1)
    # Subtask history saved by label_subtask.py
    subtask_meta = data.attrs.get("subtask_history")
    language_instruction = data.attrs.get("language_instruction", "").decode("utf-8") if isinstance(
        data.attrs.get("language_instruction"), bytes
    ) else data.attrs.get("language_instruction", "")
    subtask_history = []
    if subtask_meta:
        try:
            meta = json.loads(subtask_meta)
            subtask_history = meta.get("subtask_history", [])
            if not language_instruction:
                language_instruction = meta.get("language_instruction", "")
        except Exception:
            pass

    return {
        "actions": actions,
        "states": states,
        "agentview": agentview,
        "eye_in_hand": eye_in_hand,
        "extras": extras,
        "language_instruction": language_instruction,
        "subtask_history": subtask_history,
    }


def subtask_per_frame(num_steps: int, subtask_history: List[Dict]) -> Tuple[List[str], List[str]]:
    """
    Returns lists of predicate and natural language labels per frame.
    A subtask entry with step=k applies from its start (0 for the first subtask,
    previous_step + 1 for subsequent ones) through step k inclusive. After the
    last entry, keep using that subtask.
    """
    predicates = []
    naturals = []
    history = sorted(subtask_history, key=lambda x: x.get("step", 0))

    if not history:
        return ["" for _ in range(num_steps)], ["" for _ in range(num_steps)]

    # Initialize with the first subtask, starting at t=0
    current_pred = history[0].get("predicate", "")
    current_nl = history[0].get("natural_language", current_pred)
    next_idx = 1

    for t in range(num_steps):
        # Switch when we've passed the completion step of the current subtask
        while next_idx < len(history) and t > history[next_idx - 1].get("step", 0):
            current_pred = history[next_idx].get("predicate", "")
            current_nl = history[next_idx].get("natural_language", current_pred)
            next_idx += 1

        predicates.append(current_pred)
        naturals.append(current_nl)
    return predicates, naturals


def create_features(states: np.ndarray, actions: np.ndarray, agentview: np.ndarray, eye: np.ndarray):
    return {
        "state": {
            "dtype": "float32",
            "shape": (states.shape[1],),
            "names": [f"state_dim_{i}" for i in range(states.shape[1])],
        },
        "actions": {
            "dtype": "float32",
            "shape": (actions.shape[1],),
            "names": [f"action_dim_{i}" for i in range(actions.shape[1])],
        },
        "images.agentview_rgb": {
            "dtype": "image",
            "shape": agentview.shape[1:],
            "names": ["channel", "height", "width"],
        },
        "images.wrist_rgb": {
            "dtype": "image",
            "shape": eye.shape[1:],
            "names": ["channel", "height", "width"],
        },
        "subtask": {"dtype": "string", "shape": (1,)},
    }


def convert_all(hdf5_files: list[Path], output_root: Path, repo_id: str, fps: float):
    if not hdf5_files:
        print("No HDF5 files to convert.")
        return None

    dataset_path = output_root / repo_id
    if dataset_path.exists():
        import shutil

        shutil.rmtree(dataset_path)

    # Define features from first file / demo
    with h5py.File(hdf5_files[0], "r") as f0:
        demo_keys0 = sorted(list(f0["data"].keys()))
        if not demo_keys0:
            print(f"[skip] no demos in {hdf5_files[0].name}")
            return None
        sample = load_demo(f0, demo_keys0[0])
        features = create_features(sample["states"], sample["actions"], sample["agentview"], sample["eye_in_hand"])

    dataset = LeRobotDataset.create(
        repo_id=f"KeWangRobotics/{repo_id}",
        root=dataset_path,
        fps=fps,
        robot_type="libero_panda",
        features=features,
        use_videos=False,
        image_writer_threads=8,
        image_writer_processes=4,
    )
    # Prevent cleanup from deleting saved images; keep buffer reset only
    dataset.clear_episode_buffer = (
        lambda delete_images=True, _orig=dataset.clear_episode_buffer: _orig(delete_images=False)
    )

    for hdf5_path in hdf5_files:
        with h5py.File(hdf5_path, "r") as f:
            demo_keys = sorted(list(f["data"].keys()))
            if not demo_keys:
                print(f"[skip] no demos in {hdf5_path.name}")
                continue
            for demo_key in demo_keys:
                d = load_demo(f, demo_key)
                preds, texts = subtask_per_frame(len(d["actions"]), d["subtask_history"])
                task_text = d["language_instruction"]
                for i in tqdm(range(len(d["actions"])), desc=f"{hdf5_path.name}:{demo_key}", leave=False):
                    subtask_txt = str(texts[i]) if texts[i] is not None else ""
                    frame = {
                        "state": d["states"][i].astype(np.float32),
                        "actions": d["actions"][i].astype(np.float32),
                        "images.agentview_rgb": d["agentview"][i],
                        "images.wrist_rgb": d["eye_in_hand"][i],
                        "subtask": subtask_txt,
                        "task": task_text,
                    }
                    dataset.add_frame(frame)
                dataset.save_episode()

    try:
        dataset._wait_image_writer()
        dataset.stop_image_writer()
    except Exception:
        pass

    print(
        f"[done] merged {len(hdf5_files)} files -> {dataset_path} | episodes {dataset.meta.total_episodes} | frames {dataset.meta.total_frames}"
    )
    dataset.finalize()
    dataset.push_to_hub()
    return dataset_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert LIBERO HDF5 (with subtask labels) into a single merged LeRobot dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing LIBERO HDF5 files.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="./lerobot_exports",
        help="Root directory to store converted datasets.",
    )
    parser.add_argument(
        "--repo-prefix",
        type=str,
        default="libero_subtasks",
        help="Repo id suffix; final repo id is KeWangRobotics/<repo_prefix>.",
    )
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    hdf5_files = sorted([p for p in input_dir.glob("*.hdf5")] + [p for p in input_dir.glob("*.h5")])
    if not hdf5_files:
        print(f"No HDF5 files found in {input_dir}")
        return

    print(f"Found {len(hdf5_files)} files")
    convert_all(hdf5_files, output_root, args.repo_prefix, fps=args.fps)


if __name__ == "__main__":
    main()
