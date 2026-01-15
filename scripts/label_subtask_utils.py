import ast
import json
import os

import h5py


def decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def sorted_demo_keys(data_group):
    demo_keys = list(data_group.keys())

    def demo_index(key):
        try:
            return int(key.split("_")[-1])
        except (ValueError, AttributeError):
            return key

    return sorted(demo_keys, key=demo_index)


def iter_demo_keys(data_group, demo_key):
    if demo_key is not None:
        if str(demo_key).isdigit():
            demo_key = f"demo_{demo_key}"
        return [demo_key]
    return sorted_demo_keys(data_group)


def load_env_kwargs(data_group):
    env_kwargs = {}
    if "env_args" in data_group.attrs:
        try:
            env_args = json.loads(decode_attr(data_group.attrs["env_args"]))
            if isinstance(env_args, dict) and "env_kwargs" in env_args:
                env_kwargs = env_args["env_kwargs"] or {}
        except (TypeError, ValueError):
            env_kwargs = {}
    elif "env_info" in data_group.attrs:
        try:
            env_kwargs = json.loads(decode_attr(data_group.attrs["env_info"])) or {}
        except (TypeError, ValueError):
            env_kwargs = {}
    return env_kwargs


def load_controller_config(default_controller, robots):
    controller_config = None
    try:
        from robosuite import load_controller_config as legacy_loader

        controller_config = legacy_loader(default_controller=default_controller)
    except Exception:
        try:
            from robosuite.controllers import load_composite_controller_config

            robot_name = robots[0] if robots else "Panda"
            controller_config = load_composite_controller_config(
                controller=None, robot=robot_name
            )
        except Exception:
            try:
                from robosuite.controllers import load_part_controller_config

                controller_config = load_part_controller_config(
                    default_controller=default_controller
                )
            except Exception as exc:
                print(f"Warning: Failed to load controller config ({exc}).")

    try:
        from robosuite.controllers.composite.composite_controller_factory import (
            is_part_controller_config,
            refactor_composite_controller_config,
        )

        if controller_config and is_part_controller_config(controller_config):
            robot_name = robots[0] if robots else "Panda"
            controller_config = refactor_composite_controller_config(
                controller_config, robot_name, ["right"]
            )
    except Exception:
        pass

    return controller_config


def normalize_controller_configs(controller_configs, robots):
    if controller_configs is None:
        return None

    controller_configs_list = (
        controller_configs if isinstance(controller_configs, list) else [controller_configs]
    )

    try:
        from robosuite.controllers.composite.composite_controller_factory import (
            is_part_controller_config,
            refactor_composite_controller_config,
        )
    except Exception:
        return controller_configs

    robot_name = robots[0] if robots else "Panda"
    normalized = []
    for config in controller_configs_list:
        if is_part_controller_config(config):
            normalized.append(
                refactor_composite_controller_config(config, robot_name, ["right"])
            )
        else:
            normalized.append(config)

    return normalized if isinstance(controller_configs, list) else normalized[0]


def filter_env_kwargs(env_kwargs, env_class):
    allowed = set()
    for cls in env_class.mro():
        if cls is object:
            continue
        try:
            import inspect

            sig = inspect.signature(cls.__init__)
        except (TypeError, ValueError):
            continue
        for name, param in sig.parameters.items():
            if name == "self" or param.kind == param.VAR_KEYWORD:
                continue
            allowed.add(name)

    if not allowed:
        return env_kwargs

    filtered = {key: value for key, value in env_kwargs.items() if key in allowed}
    dropped = sorted(set(env_kwargs) - set(filtered))
    if dropped:
        print(f"Warning: Dropping unsupported env args: {', '.join(dropped)}")
    return filtered


def resolve_bddl_path(bddl_file_name, dataset_path, benchmark_name, task_suite, get_libero_path):
    if bddl_file_name:
        bddl_file_name = decode_attr(bddl_file_name)
        if os.path.isabs(bddl_file_name) and os.path.exists(bddl_file_name):
            return bddl_file_name

        candidates = []
        dataset_dir = os.path.dirname(os.path.abspath(dataset_path))
        candidates.append(os.path.join(dataset_dir, bddl_file_name))
        candidates.append(os.path.join(get_libero_path("bddl_files"), bddl_file_name))

        bddl_marker = f"bddl_files{os.sep}"
        if bddl_marker in bddl_file_name:
            suffix = bddl_file_name.split(bddl_marker, 1)[-1]
            candidates.append(os.path.join(get_libero_path("bddl_files"), suffix))

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

    if task_suite is not None:
        return task_suite.get_task_bddl_file_path(0)
    return None


def get_demo_bddl_name(data_group, demo_key):
    demo_attrs = data_group[demo_key].attrs
    for key in ["bddl_file_name", "bddl_file"]:
        if key in demo_attrs:
            return demo_attrs.get(key)
    for key in ["bddl_file_name", "bddl_file"]:
        if key in data_group.attrs:
            return data_group.attrs.get(key)
    return None


def predicate_to_text(predicate_str, language_instruction=""):
    try:
        parts = ast.literal_eval(predicate_str)
        if not isinstance(parts, (list, tuple)) or not parts:
            raise ValueError
    except Exception:
        return predicate_str

    verb = str(parts[0]).lower()
    args = [str(p) for p in parts[1:]]

    def join_args(items):
        return " and ".join(items)

    if verb in {"on", "ontop"} and len(args) >= 2:
        return f"Place {args[0]} on {join_args(args[1:])}"
    if verb in {"in", "inside"} and len(args) >= 2:
        return f"Put {args[0]} inside {join_args(args[1:])}"
    if verb in {"under"} and len(args) >= 2:
        return f"Put {args[0]} under {join_args(args[1:])}"
    if verb in {"near", "nextto"} and len(args) >= 2:
        return f"Place {args[0]} near {join_args(args[1:])}"
    if verb in {"close"} and args:
        return f"Close {join_args(args)}"
    if verb in {"open"} and args:
        return f"Open {join_args(args)}"
    if verb in {"turnon", "switchon"} and args:
        return f"Turn on {join_args(args)}"
    if verb in {"turnoff", "switchoff"} and args:
        return f"Turn off {join_args(args)}"

    return f"{verb} " + " ".join(args) if args else verb


def save_subtask_history(
    subtask_history,
    language_instruction,
    demo_key,
    output_path=None,
    save_to_hdf5=False,
    hdf5_file=None,
    demo_group=None,
    natural_language_history=None,
):
    subtask_data = {
        "demo_key": demo_key,
        "language_instruction": language_instruction,
        "subtask_history": [
            {
                "step": step,
                "predicate": predicate_str,
                "natural_language": natural_language_history.get((step, predicate_str))
                if natural_language_history
                else None,
            }
            for step, predicate_str in subtask_history
        ],
        "num_subtasks": len(subtask_history),
    }

    if save_to_hdf5 and hdf5_file is not None and demo_group is not None:
        try:
            if "subtask_history" in demo_group.attrs:
                del demo_group.attrs["subtask_history"]
            demo_group.attrs["subtask_history"] = json.dumps(subtask_data)
            demo_group.attrs["language_instruction"] = language_instruction
            demo_group.attrs["num_subtasks"] = len(subtask_history)
            print(f"  Saved subtask history to HDF5 for {demo_key}")
        except Exception as e:
            print(f"  Warning: Could not save to HDF5: {e}")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(subtask_data, f, indent=2)
        print(f"  Saved subtask history to: {output_path}")

