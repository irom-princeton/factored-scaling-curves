import os
import re


def sort_path(traj_path):
    return sorted(
        traj_path, key=lambda x: int(re.search(r"(\d+)", x).group())
    )  # Ensure ordering before sampling


def get_data_paths(input_paths):
    # concatenate all paths
    real_traj_paths = []
    sim_traj_paths = []
    num_traj_available = 0
    for path in input_paths:
        if "sim" in path:
            sim_traj_paths.append(
                [
                    os.path.join(path, traj_name)
                    for traj_name in os.listdir(path)
                    # if traj_name.endswith(".h5") and "failed" not in traj_name
                    if traj_name.endswith(".h5")
                ]
            )
            num_traj_available += len(sim_traj_paths[-1])
        else:
            real_traj_paths.append(
                [
                    os.path.join(path, traj_name)
                    for traj_name in os.listdir(path)
                    if traj_name.endswith(".h5")
                ]
            )
            num_traj_available += len(real_traj_paths[-1])
    return real_traj_paths, sim_traj_paths, num_traj_available


def sample_paths(
    traj_paths,
    num_trajs,
    ratio=None,
    sample_strat="uniform",
    delta=20,
    num_instances=5,
    num_per_instance=30,
):
    if sample_strat in ("uniform", "val"):
        output_paths = []
        for i, traj_path in enumerate(traj_paths):
            sorted_paths = sort_path(traj_path)
            num_samples = num_trajs[i]

            # Define variation indices based on the variation type
            if "lighting" in sorted_paths[0]:
                idx_per_variation = [16, 30, 44, 58]
            elif any(
                k in sorted_paths[0]
                for k in ["camera_pose", "distractor", "background"]
            ):
                if "camera_pose_new_new" in sorted_paths[0]:
                    idx_per_variation = [20, 40, 60, 80]
                else:
                    idx_per_variation = [25, 50, 75, 100]
            elif "table_texture" in sorted_paths[0]:
                idx_per_variation = [20, 40, 60, 80]
            else:
                raise ValueError(f"Unknown variation type in {sorted_paths[0]}")

            paths = []
            for j in range(num_samples):
                variation_idx = j % len(idx_per_variation)
                sample_idx = j // len(idx_per_variation)

                if sample_strat == "val":
                    index = (
                        idx_per_variation[variation_idx]
                        + idx_per_variation[1]
                        - idx_per_variation[0]
                        - 1
                        - sample_idx
                    )
                else:
                    index = idx_per_variation[variation_idx] + sample_idx
                if index < len(sorted_paths):  # Ensure the index is valid
                    paths.append(sorted_paths[index])
                else:
                    print(
                        f"Warning: Index {index} out of bounds for variation {variation_idx}, skipping."
                    )

            output_paths += paths

    elif sample_strat == "sim_uniform":
        output_paths = []
        for i, traj_path in enumerate(traj_paths):
            sorted_paths = sort_path(traj_path)
            num_samples = num_trajs[i]
            idx_per_variation = {num_per_instance * i: 0 for i in range(num_instances)}
            variation_keys = list(idx_per_variation.keys())
            paths = []
            j = 0
            while len(paths) < num_samples:
                variation_idx = j % len(idx_per_variation)
                sample_idx = idx_per_variation[variation_keys[variation_idx]]
                index = variation_keys[variation_idx] + sample_idx
                if index < len(sorted_paths):  # Ensure the index is valid
                    if "failed" not in sorted_paths[index]:
                        paths.append(sorted_paths[index])
                        idx_per_variation[variation_keys[variation_idx]] += 1
                    else:
                        add_path = True
                        while "failed" in sorted_paths[index]:
                            index += 1
                            if (
                                index
                                >= variation_keys[variation_idx] + num_per_instance
                            ):
                                # raise ValueError(
                                #     f"Too many failed paths for variation {variation_idx}. {sorted_paths}"
                                # )
                                add_path = False
                                break
                        if add_path:
                            paths.append(sorted_paths[index])
                            idx_per_variation[variation_keys[variation_idx]] = (
                                index - variation_keys[variation_idx] + 1
                            )
                else:
                    raise ValueError(
                        f"Warning: Index {index} out of bounds for variation {variation_idx}, skipping."
                    )
                j += 1
            output_paths += paths

    elif sample_strat == "sim_factor":
        factor_paths = traj_paths.pop(-1)
        num_samples = num_trajs.pop(-1)
        sorted_paths = sort_path(factor_paths)

        output_paths = sample_paths(
            traj_paths,
            num_trajs,
            ratio=None,
            sample_strat="sim_uniform",
            delta=None,
            num_instances=num_instances,
            num_per_instance=num_per_instance,
        )

        num_instances = (num_samples + delta - 1) // delta
        samples_per_instance = [delta] * (num_instances - 1) + [
            num_samples - delta * (num_instances - 1)
        ]
        for instance, num in zip(range(num_instances), samples_per_instance):
            count = 0
            i = instance * num_per_instance
            while count < num and i < len(sorted_paths):
                if "failed" not in sorted_paths[i]:
                    output_paths.append(sorted_paths[i])
                    count += 1
                i += 1  # Move to the next path

    elif sample_strat == "real_factor":
        factor_paths = traj_paths.pop(-1)
        num_samples = num_trajs.pop(-1)
        sorted_paths = sort_path(factor_paths)

        output_paths = sample_paths(
            traj_paths,
            num_trajs,
            ratio=None,
            sample_strat="uniform",
            delta=None,
            num_instances=None,
            num_per_instance=None,
        )

        # Define variation indices based on the variation type
        if "lighting" in sorted_paths[0]:
            num_per_instance = 20  # TODO: actual num_per_instance=14 < delta
        elif any(
            k in sorted_paths[0] for k in ["camera_pose", "distractor", "background"]
        ):
            if "camera_pose_new_new" in sorted_paths[0]:
                num_per_instance = 20
            else:
                num_per_instance = 25
        elif "table_texture" in sorted_paths[0]:
            num_per_instance = 20
        else:
            raise ValueError(f"Unknown variation type in {sorted_paths[0]}")

        num_instances = (num_samples + delta - 1) // delta
        samples_per_instance = [delta] * (num_instances - 1) + [
            num_samples - delta * (num_instances - 1)
        ]
        for instance, num in zip(range(num_instances), samples_per_instance):
            output_paths += sorted_paths[
                instance * num_per_instance : instance * num_per_instance + num
            ]

    elif sample_strat == "ordered":
        output_paths = []
        for i, traj_path in enumerate(traj_paths):
            sorted_paths = sort_path(traj_path)  # Ensure ordering before sampling
            num_samples = num_trajs[i]
            samples_added = 0
            for path in sorted_paths:
                if "failed" not in path:
                    output_paths.append(path)
                    samples_added += 1
                if samples_added >= num_samples:
                    break

    elif sample_strat == "sim_val":
        output_paths = []
        for i, traj_path in enumerate(traj_paths):
            sorted_paths = sort_path(traj_path)  # Ensure ordering before sampling
            num_samples = num_trajs[i]
            samples_added = 0
            for path in sorted_paths[::-1]:
                if "failed" not in path:
                    output_paths.append(path)
                    samples_added += 1
                if samples_added >= num_samples:
                    break

    else:
        raise NotImplementedError(f"Unknown sampling strategy: {sample_strat}")

    return output_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--args.sim_data_ratio",
        default=None,
    )
    parser.add_argument(
        "--real_data_ratio",
        default=None,
    )
    parser.add_argument(
        "--args.num_sim_traj",
        type=int,
        nargs="+",  # Accepts one or more integers
        default=None,
    )
    parser.add_argument(
        "--num_real_traj",
        type=int,
        nargs="+",  # Accepts one or more integers
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--sample_real_strat",
        type=str,
        default="uniform",
    )
    parser.add_argument(
        "--keep_only_first",
        action="store_true",
    )
    parser.add_argument(
        "--sample_sim_strat",
        type=str,
        default="sim_uniform",
    )
    parser.add_argument(
        "--delta",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--num_per_instance",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    assert not (args.num_sim_traj is not None and args.sim_data_ratio is not None), (
        "Only one of args.sim_data_ratio and args.num_sim_traj should be specified"
    )
    assert not (args.num_real_traj is not None and args.real_data_ratio is not None), (
        "Only one of real_data_ratio and num_real_traj should be specified"
    )

    real_traj_paths, sim_traj_paths, num_traj_available = get_data_paths(
        args.input_paths
    )

    # sim_traj_paths = sample_paths(
    #     sim_traj_paths,
    #     args.num_sim_traj,
    #     args.sim_data_ratio,
    #     args.sample_sim_strat,
    #     delta=args.delta,
    #     num_instances=args.num_instances,
    #     num_per_instance=args.num_per_instance,
    # )
    real_traj_paths = sample_paths(
        real_traj_paths,
        args.num_real_traj,
        args.real_data_ratio,
        args.sample_real_strat,
        delta=args.delta,
    )

    print(real_traj_paths)
    # return real_traj_paths
    # #print(sim_traj_paths)
