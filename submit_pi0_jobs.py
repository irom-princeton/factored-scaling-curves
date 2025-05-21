import itertools
import os
import subprocess
from pathlib import Path

import hydra

from plot_curve import PlotCurve

DATA_DIR = os.getenv("GDC_DATA_DIR")
RESULTS_DIR = os.getenv("GDC_RESULTS_DIR", os.path.join("./results"))


def parse(k):
    ks = []
    if "Camera Pose" in k or "camera_pose" in k:
        ks.append("cp")
    if "Background" in k or "background" in k:
        ks.append("bg")
    if "Directional" in k or "Lighting" in k or "lighting" in k or "directional" in k:
        ks.append("lt")
    if "Distractor" in k or "distractor" in k:
        ks.append("dis")
    if "Table Texture" in k or "table_texture" in k:
        ks.append("tt")
    if "Delta qpos" in k or "delta_qpos" in k:
        ks.append("dq")
    if "Table Height" in k or "table_height" in k:
        ks.append("th")
    if "Object Pose" in k or "object_pose" in k or "obj_pose" in k:
        ks.append("op")
    return ks


class Pi0RealTowelVisual:
    seeds = [0]
    exp_type = "visual"
    instruction = "fold the towel"

    @property
    def path_groups(self) -> dict[str, list[str]]:
        return {
            "dis": [f"{DATA_DIR}/raw_data/fold_towel_distractor"],
            "tt": [f"{DATA_DIR}/raw_data/fold_towel_table_texture"],
            "lt": [f"{DATA_DIR}/raw_data/fold_towel_lighting"],
            "cp": [f"{DATA_DIR}/raw_data/fold_towel_camera_pose"],
        }

    @property
    def policy_name(self) -> str:
        return "pi0"

    @property
    def env_name(self) -> str:
        return "fold_towel"

    def __init__(self, cfg):
        self.real_data_nums = cfg.real_data_nums
        self.num_train_points = cfg.num_train_points
        self.sample_real_strat = cfg.sample_real_strat
        self.k_factor = cfg.k_factor
        self.data_ready = cfg.data_ready
        self.train_ready = cfg.train_ready
        self.run_baseline_equal = cfg.run_baseline_equal
        self.run_baseline_remix = cfg.run_baseline_remix
        self.dry_run = cfg.dry_run
        self.cfg_name = cfg.cfg_name

        self.slope_num_factors = cfg.slope_num_factors

        self.num_base_demos_per_factor = (
            cfg.real_data_nums[cfg.num_train_points - 1] // self.k_factor
        )
        self.num_per_instance = cfg.num_per_instance
        self.num_instances = cfg.num_instances
        self.delta_equal = [
            n - cfg.real_data_nums[cfg.num_train_points - 1]
            for n in cfg.real_data_nums[cfg.num_train_points :]
        ]
        self.delta = cfg.real_data_nums[1] - cfg.real_data_nums[0]
        self.combos = (
            cfg.combos
            if cfg.combos is not None
            else self.generate_comb_jobs(self.k_factor)
        )
        self.num_base_demos = self.num_base_demos_per_factor * (
            len(self.path_groups) - self.k_factor
        )
        self.exp_name = f"real_{self.env_name}_{self.exp_type}_{self.policy_name}_{cfg.x_strat}_{self.sample_real_strat}_base{self.num_base_demos}_vary{self.real_data_nums[-1]}"
        # print(self.delta_equal, self.num_base_demos_per_factor, self.num_base_demos, self.exp_name)

    def get_input_paths(self):
        input_paths = []
        for key in self.path_groups:
            input_paths += self.path_groups[key]
        return input_paths

    def generate_comb_jobs(self, k=1):
        keys = list(self.path_groups.keys())
        combs = list(itertools.combinations(keys, k))
        return combs

    def build_distributed_traj(self, vary_keys, real_data_num):
        dist_traj = []

        # Base (fixed) group allocations
        for key in self.path_groups:
            if key not in vary_keys:
                total = self.num_base_demos_per_factor
                count = len(self.path_groups[key])
                base = total // count
                rem = total % count
                dist_traj += [base] * (count - 1) + [base + rem]

        # Varying group allocations
        total = real_data_num
        vary_paths = sum([len(self.path_groups[k]) for k in vary_keys])
        base = total // vary_paths
        rem = total % vary_paths
        dist_traj += [base] * (vary_paths - 1) + [base + rem]

        return dist_traj

    def run_command(self, cmd):
        if self.dry_run:
            print(cmd)
        else:
            return subprocess.check_output(cmd, shell=True).decode().strip()

    def main(self):
        total_demos = []
        for vary_keys in self.combos:
            job_ids_exp = []
            vary_suffix = "".join(vary_keys)
            input_paths = " ".join(self.get_input_paths())
            print(vary_keys)
            for seed in self.seeds:
                for sim_data_num in self.real_data_nums[:]:
                    dist_traj = self.build_distributed_traj(vary_keys, sim_data_num)
                    dist_str = " ".join(map(str, dist_traj))
                    total = sum(dist_traj)
                    if len(total_demos) < len(self.real_data_nums):
                        total_demos.append(total)
                    print(dist_str)
                    job_id = self.submit_jobs(
                        input_paths,
                        dist_str,
                        seed,
                        self.exp_name,
                        total,
                        curve_name=vary_suffix,
                    )
                    job_ids_exp.append(job_id)
            print(job_ids_exp)
            self.echo_command(job_ids_exp, vary_suffix)
        if self.run_baseline_equal:
            self.baseline_equal()
        if self.run_baseline_remix:
            self.baseline_remix()

    def submit_jobs(
        self,
        input_string,
        sim_traj_string,
        seed,
        exp_name,
        total_demos,
        curve_name,
        eval_name=None,
        num_per_instance=None,
        train_job_id=None,
        num_instances=None,
    ):
        repo_id = f"{exp_name}{curve_name}_seed{seed}_sim0_real{total_demos}"
        job1 = self.submit_data_job(
            input_string,
            sim_traj_string,
            seed,
            repo_id,
            num_per_instance=num_per_instance,
            num_instances=num_instances,
        )
        job2 = self.submit_stats_job(job1, repo_id)
        job3 = self.submit_training_job(job2, repo_id, repo_id)

        return job3

    def submit_data_job(
        self,
        input_string,
        sim_traj_string,
        seed,
        repo_id,
        num_per_instance=None,
        num_instances=None,
    ):
        job1 = self.run_command(
            "cd pi-zero && "
            f"sbatch --parsable process_data.sh "
            f"-in {input_string} "
            f"--num_real_traj {sim_traj_string} "
            f"--seed {seed} "
            f"--dataset_name {repo_id} "
            f"--sample_real_strat {self.sample_real_strat} "
            f"--delta {self.delta} "
            f"--num_per_instance {self.num_per_instance if num_per_instance is None else num_per_instance} "
            f"--num_instances {self.num_instances if num_instances is None else num_instances} "
            f"--instruction '{self.instruction}' "
        )
        return job1

    def submit_stats_job(self, job1, repo_id):
        job2 = self.run_command(
            "cd pi-zero && "
            f"sbatch --parsable --dependency=afterok:{job1} compute_norm_stats.sh "
            f"--repo_id {repo_id} "
        )
        return job2

    def submit_training_job(self, job2, repo_id, exp_name):
        job3 = self.run_command(
            "cd pi-zero && "
            f"sbatch --parsable --dependency=afterok:{job2} train.sh "
            f"--exp_name {exp_name} "
            f"--repo_id {repo_id} "
            f"--use_droid 0 "
            f"--num_train_steps 5001 "
            f"--keep_period 100 "
            f"--batch_size 32 "
            f"--freeze_llm 1 "
            f"--freeze_img 1 "
            f"--fsdp_devices 2 "
            f"--ema_decay {None} "
            f"--overwrite "
        )
        return job3

    def baseline_equal(self):
        # Baseline equal
        job_ids_equal = []
        input_string = " ".join(sum(self.path_groups.values(), []))

        for delta in self.delta_equal:
            base_part = delta // len(self.path_groups)
            rem = delta % len(self.path_groups)
            split = [
                (base_part + 1 if i < rem else base_part)
                for i in range(len(self.path_groups))
            ]

            distributed = []
            total = 0
            for i, (_, _) in enumerate(self.path_groups.items()):
                val = self.num_base_demos_per_factor + split[i]
                distributed.append(str(val))
                total += val
            job_id = self.submit_jobs(
                input_string,
                " ".join(distributed),
                self.seeds[0],
                exp_name=self.exp_name,
                curve_name="baseline_equal",
                total_demos=total,
            )
            job_ids_equal.append(job_id)
        self.echo_command(job_ids_equal, "baseline_equal")

    def baseline_remix(self):
        pass

    def echo_command(self, job_ids, curve_name, file_name=None):
        # cmd = f"./get_sim_curves.sh {' '.join(job_ids)} grid sim_{self.env_name}_{self.additional_name}_{self.sample_real_strat}_base{self.num_base_demos}_vary{self.real_data_nums[-1]}{curve_name} {self.env_name}"
        # script_file = Path("exp.sh")
        # script_file.write_text(script_file.read_text() + f"\n{cmd}" if script_file.exists() else cmd)
        # script_file.chmod(0o755)
        # print("Warning: echo_command is not implemented for real tasks.")
        pass

    def echo_plot_curve_command(self, **kwargs):
        # cmd = f"python plot_curve.py --exp-name {self.exp_name} --show-power-law --num-points {self.num_train_points} --use-sim --env-name {self.env_name} --num-worst-factor {self.k_factor} --points {' '.join(map(str, self.total_demos))}"

        # cmd = f"python submit_sim_training.py --config-name {self.cfg_name} run_method=weighted_slope"
        # script_file = Path(os.path.join(RESULTS_DIR, self.exp_name, "get_train_result.sh"))
        # script_file.write_text(script_file.read_text() + f"\n{cmd}" if script_file.exists() else cmd)
        # script_file.chmod(0o755)
        pass

    def turn_weights_to_distribute(self, point, r, num_factors_to_collect):
        delta = (
            int(point)
            - self.num_base_demos
            - self.real_data_nums[self.num_train_points - 1]
        )

        # Aggregate factor weights from combinations
        factor_weights = {k: 0 for k in self.path_groups}
        for combo, weight in r[
            "success_rate_scaling_curve"
        ].items():  # Combo: "Camera Pose Table Texture"
            for f in parse(combo):
                factor_weights[f] += weight / len(parse(combo))

        # Pick top factors and normalize their weights
        top_factors = sorted(factor_weights.items(), key=lambda x: x[1], reverse=True)[
            :num_factors_to_collect
        ]
        total_weight = sum(w for _, w in top_factors)
        norm_weights = [(k, w / total_weight) for k, w in top_factors]

        # Allocate demonstrations proportional to normalized weights
        factor_num = {
            k: self.num_base_demos_per_factor + round(w * delta)
            for k, w in norm_weights
        }
        diff = delta - sum(
            x - self.num_base_demos_per_factor for x in factor_num.values()
        )
        keys = list(factor_num.keys())
        for i in range(diff):
            factor_num[keys[i % len(keys)]] += 1

        # Build distribution for all path groups (zeros for non-top factors)
        distributed = [
            str(factor_num.get(k, self.num_base_demos_per_factor))
            for k in self.path_groups
        ]
        print(f"delta={delta} | Allocations: {' '.join(distributed)}")

        return distributed, sum(int(x) for x in distributed)

    def weighted_slope(self):
        curve_plotter = PlotCurve(
            exp_name=self.exp_name,
            env_name=self.env_name,
            num_points=self.num_train_points,
            use_sim=True,
            show_power_law=True,
            move_folder=False,
            set_avg_final_sr=True,
            set_avg_final_metric=True,
            num_worst_factors=self.k_factor,
            points=self.total_demos,
            with_all_results=False,
        )
        _, slope_weights, worst_weights = curve_plotter.get_final_results()
        input_string = " ".join(sum(self.path_groups.values(), []))

        total_jobs_submitted = 0

        for num_factors_to_collect in self.slope_num_factors:
            job_ids = []
            for point, r in slope_weights.items():
                distributed, total = self.turn_weights_to_distribute(
                    point, r, num_factors_to_collect
                )
                job_id = self.submit_jobs(
                    input_string,
                    " ".join(distributed),
                    self.seeds[0],
                    self.exp_name,
                    total_demos=total,
                    curve_name=f"slope_{num_factors_to_collect}",
                )
                job_ids.append(job_id)
            total_jobs_submitted += len(job_ids)
            self.echo_command(
                job_ids,
                f"slope_{num_factors_to_collect}",
                file_name="get_test_result.sh",
            )

        num_factors_to_collects = [1, len(self.path_groups)]
        for num_factors_to_collect in num_factors_to_collects:
            job_ids = []
            for point, r in worst_weights.items():
                distributed, total = self.turn_weights_to_distribute(
                    point, r, num_factors_to_collect
                )
                job_id = self.submit_jobs(
                    input_string,
                    " ".join(distributed),
                    self.seeds[0],
                    self.exp_name,
                    total_demos=total,
                    curve_name=f"worst_{num_factors_to_collect}",
                )
                job_ids.append(job_id)
            total_jobs_submitted += len(job_ids)
            self.echo_command(
                job_ids,
                f"worst_{num_factors_to_collect}",
                file_name="get_test_result.sh",
            )

        print(total_jobs_submitted)


class Pi0RealTowelSpatial(Pi0RealTowelVisual):
    exp_type = "spatial"
    instruction = "fold the towel"

    @property
    def path_groups(self) -> dict[str, list[str]]:
        return {
            "dis": [f"{DATA_DIR}/raw_data/fold_towel_distractor"],
            "tt": [f"{DATA_DIR}/raw_data/fold_towel_table_texture"],
            "lt": [f"{DATA_DIR}/raw_data/fold_towel_lighting"],
            "cp": [f"{DATA_DIR}/raw_data/fold_towel_camera_pose"],
            "dq": [f"{DATA_DIR}/raw_data/fold_towel_qpos"],
            "op": [f"{DATA_DIR}/raw_data/fold_towel_object_pose"],
        }

    def __init__(self, cfg):
        super().__init__(cfg)


class Pi0RealDrawer:
    seeds = [0]
    exp_type = "spatial"
    instruction = (
        "open the drawer, pick up the mouse and put into the drawer, close the drawer"
    )

    @property
    def path_groups(self) -> dict[str, list[str]]:
        return {
            "dis": [f"{DATA_DIR}/raw_data/drawer_distractor"],
            "tt": [f"{DATA_DIR}/raw_data/drawer_table_texture"],
            "lt": [f"{DATA_DIR}/raw_data/drawer_lighting"],
            "cp": [f"{DATA_DIR}/raw_data/drawer_camera_pose"],
            "dq": [f"{DATA_DIR}/raw_data/drawer_qpos"],
            "op": [f"{DATA_DIR}/raw_data/drawer_object_pose"],
        }

    @property
    def policy_name(self) -> str:
        return "pi0"

    @property
    def env_name(self) -> str:
        return "drawer"

    def __init__(self, cfg):
        self.real_data_nums = cfg.real_data_nums
        self.num_train_points = cfg.num_train_points
        self.sample_real_strat = cfg.sample_real_strat
        self.k_factor = cfg.k_factor
        self.data_ready = cfg.data_ready
        self.train_ready = cfg.train_ready
        self.run_baseline_equal = cfg.run_baseline_equal
        self.run_baseline_remix = cfg.run_baseline_remix
        self.dry_run = cfg.dry_run
        self.cfg_name = cfg.cfg_name

        self.slope_num_factors = cfg.slope_num_factors

        self.num_base_demos_per_factor = (
            cfg.real_data_nums[cfg.num_train_points - 1] // self.k_factor
        )
        self.num_per_instance = cfg.num_per_instance
        self.num_instances = cfg.num_instances
        self.delta_equal = [
            n - cfg.real_data_nums[cfg.num_train_points - 1]
            for n in cfg.real_data_nums[cfg.num_train_points :]
        ]
        self.delta = cfg.real_data_nums[1] - cfg.real_data_nums[0]
        self.combos = (
            cfg.combos
            if cfg.combos is not None
            else self.generate_comb_jobs(self.k_factor)
        )
        self.num_base_demos = self.num_base_demos_per_factor * (
            len(self.path_groups) - self.k_factor
        )
        if cfg.get("exp_name", None) is None:
            self.exp_name = f"real_{self.env_name}_{self.exp_type}_{self.policy_name}_{cfg.x_strat}_{self.sample_real_strat}_base{self.num_base_demos}_vary{self.real_data_nums[-1]}_bugfix"
        else:
            self.exp_name = cfg.exp_name
        # print(self.delta_equal, self.num_base_demos_per_factor, self.num_base_demos, self.exp_name)

    def get_input_paths(self, vary_keys):
        input_paths = []
        for key in self.path_groups:
            if key not in vary_keys:
                input_paths += self.path_groups[key]
        for key in vary_keys:
            input_paths += self.path_groups[key]
        return input_paths

    def generate_comb_jobs(self, k=1):
        keys = list(self.path_groups.keys())
        combs = list(itertools.combinations(keys, k))
        return combs

    def build_distributed_traj(self, vary_keys, real_data_num):
        dist_traj = []

        # Base (fixed) group allocations
        for key in self.path_groups:
            if key not in vary_keys:
                total = self.num_base_demos_per_factor
                count = len(self.path_groups[key])
                base = total // count
                rem = total % count
                dist_traj += [base] * (count - 1) + [base + rem]

        # Varying group allocations
        total = real_data_num
        vary_paths = sum([len(self.path_groups[k]) for k in vary_keys])
        base = total // vary_paths
        rem = total % vary_paths
        dist_traj += [base] * (vary_paths - 1) + [base + rem]

        return dist_traj

    def run_command(self, cmd):
        if self.dry_run:
            print(cmd)
        else:
            return subprocess.check_output(cmd, shell=True).decode().strip()

    def main(self):
        total_demos = []
        num_jobs_submitted = 0
        for vary_keys in self.combos:
            job_ids_exp = []
            vary_suffix = "".join(vary_keys)
            input_paths = " ".join(self.get_input_paths(vary_keys))
            print(vary_keys)
            for seed in self.seeds:
                for sim_data_num in self.real_data_nums[: self.num_train_points]:
                    dist_traj = self.build_distributed_traj(vary_keys, sim_data_num)
                    dist_str = " ".join(map(str, dist_traj))
                    total = sum(dist_traj)
                    if len(total_demos) < len(self.real_data_nums):
                        total_demos.append(total)
                    print(dist_str)
                    job_id = self.submit_jobs(
                        input_paths,
                        dist_str,
                        seed,
                        self.exp_name,
                        total,
                        curve_name=vary_suffix,
                    )
                    num_jobs_submitted += 1
                    job_ids_exp.append(job_id)
            print(job_ids_exp)
            self.echo_command(job_ids_exp, vary_suffix)
        if self.run_baseline_equal:
            self.baseline_equal()

        num_jobs_submitted += 2
        print(num_jobs_submitted)

    def submit_jobs(
        self,
        input_string,
        sim_traj_string,
        seed,
        exp_name,
        total_demos,
        curve_name,
        eval_name=None,
        num_per_instance=None,
        train_job_id=None,
        num_instances=None,
    ):
        repo_id = f"{exp_name}{curve_name}_seed{seed}_sim0_real{total_demos}"
        # job1 = self.submit_data_job(input_string, sim_traj_string, seed,
        #                             repo_id, num_per_instance=num_per_instance, num_instances=num_instances)
        # job2 = self.submit_stats_job(job1, repo_id)
        job3 = self.submit_training_job(repo_id, repo_id)

        return job3

    def submit_data_job(
        self,
        input_string,
        sim_traj_string,
        seed,
        repo_id,
        num_per_instance=None,
        num_instances=None,
    ):
        job1 = self.run_command(
            "cd pi-zero && "
            f"sbatch --parsable process_data.sh "
            f"-in {input_string} "
            f"--num_real_traj {sim_traj_string} "
            f"--seed {seed} "
            f"--dataset_name {repo_id} "
            f"--sample_real_strat {self.sample_real_strat} "
            f"--delta {self.delta} "
            f"--num_per_instance {self.num_per_instance if num_per_instance is None else num_per_instance} "
            f"--num_instances {self.num_instances if num_instances is None else num_instances} "
            f"--instruction '{self.instruction}' "
        )
        return job1

    def submit_stats_job(self, job1, repo_id):
        job2 = self.run_command(
            "cd pi-zero && "
            f"sbatch --parsable --dependency=afterok:{job1} compute_norm_stats.sh "
            f"--repo_id {repo_id} "
        )
        return job2

    def submit_training_job(self, repo_id, exp_name):
        job3 = self.run_command(
            "cd pi-zero && "
            f"sbatch --parsable train.sh "
            f"--exp_name {exp_name} "
            f"--repo_id {repo_id} "
            f"--use_droid 0 "
            f"--num_train_steps 10001 "
            f"--keep_period 100 "
            f"--batch_size 32 "
            f"--freeze_llm 1 "
            f"--freeze_img 1 "
            f"--freeze_action 0 "
            f"--fsdp_devices 2 "
            f"--ema_decay {None} "
            f"--overwrite "
        )
        return job3

    def baseline_equal(self):
        # Baseline equal
        job_ids_equal = []
        input_string = " ".join(sum(self.path_groups.values(), []))

        for delta in self.delta_equal:
            base_part = delta // len(self.path_groups)
            rem = delta % len(self.path_groups)
            split = [
                (base_part + 1 if i < rem else base_part)
                for i in range(len(self.path_groups))
            ]

            distributed = []
            total = 0
            for i, (_, _) in enumerate(self.path_groups.items()):
                val = self.num_base_demos_per_factor + split[i]
                distributed.append(str(val))
                total += val
            job_id = self.submit_jobs(
                input_string,
                " ".join(distributed),
                self.seeds[0],
                exp_name=self.exp_name,
                curve_name="baseline_equal",
                total_demos=total,
            )
            job_ids_equal.append(job_id)
        self.echo_command(job_ids_equal, "baseline_equal")

    def baseline_remix(self):
        pass

    def turn_weights_to_distribute(self, point, r, num_factors_to_collect):
        delta = (
            int(point)
            - self.num_base_demos
            - self.real_data_nums[self.num_train_points - 1]
        )

        # Aggregate factor weights from combinations
        factor_weights = {k: 0 for k in self.path_groups}
        for combo, weight in r.items():  # Combo: "Camera Pose Table Texture"
            for f in parse(combo):
                factor_weights[f] += weight / len(parse(combo))

        # Pick top factors and normalize their weights
        top_factors = sorted(factor_weights.items(), key=lambda x: x[1], reverse=True)[
            :num_factors_to_collect
        ]
        total_weight = sum(w for _, w in top_factors)
        norm_weights = [(k, w / total_weight) for k, w in top_factors]

        # Allocate demonstrations proportional to normalized weights
        factor_num = {
            k: self.num_base_demos_per_factor + round(w * delta)
            for k, w in norm_weights
        }
        diff = delta - sum(
            x - self.num_base_demos_per_factor for x in factor_num.values()
        )
        keys = list(factor_num.keys())
        for i in range(diff):
            factor_num[keys[i % len(keys)]] += 1

        # Build distribution for all path groups (zeros for non-top factors)
        distributed = [
            str(factor_num.get(k, self.num_base_demos_per_factor))
            for k in self.path_groups
        ]
        print(f"delta={delta} | Allocations: {' '.join(distributed)}")

        return distributed, sum(int(x) for x in distributed)

    def echo_command(self, job_ids, curve_name, file_name=None):
        cmd = f"./get_sim_curves.sh {' '.join(map(str, job_ids))} {self.exp_name} {curve_name} {self.env_name} "

        os.makedirs(os.path.join(RESULTS_DIR, self.exp_name), exist_ok=True)
        script_file = Path(
            os.path.join(
                RESULTS_DIR,
                self.exp_name,
                "get_train_result.sh" if file_name is None else file_name,
            )
        )
        script_file.write_text(
            script_file.read_text() + f"\n{cmd}" if script_file.exists() else cmd
        )
        script_file.chmod(0o755)

    def echo_plot_curve_command(
        self, method="weighted_slope", dest="get_train_result.sh"
    ):
        # cmd = f"python plot_curve.py --exp-name {self.exp_name} --show-power-law --num-points {self.num_train_points} --use-sim --env-name {self.env_name} --num-worst-factor {self.k_factor} --points {' '.join(map(str, self.total_demos))}"

        cmd = f"python submit_pi0_jobs.py --config-name {self.cfg_name} run_method={method}"
        script_file = Path(os.path.join(RESULTS_DIR, self.exp_name, dest))
        script_file.write_text(
            script_file.read_text() + f"\n{cmd}" if script_file.exists() else cmd
        )
        script_file.chmod(0o755)

    def weighted_slope(self):
        self.total_demos = [self.num_base_demos + n for n in self.real_data_nums]

        curve_plotter = PlotCurve(
            exp_name=self.exp_name,
            env_name=self.env_name,
            num_points=self.num_train_points,
            use_sim=True,
            show_power_law=True,
            move_folder=False,
            set_avg_final_sr=False,
            set_avg_final_metric=False,
            num_worst_factors=self.k_factor,
            points=self.total_demos,
            with_all_results=False,
            get_baseline_worst_res=False,
            base_demos=self.num_base_demos,
        )
        slope_weights, worst_weights = curve_plotter.get_train_results()
        input_string = " ".join(sum(self.path_groups.values(), []))

        total_jobs_submitted = 0

        _keys = ["success_rate_scaling_curve"]

        # for key in next(iter(slope_weights.values())):
        for key in _keys:
            if "success_rate" in key:
                method_name = key.replace("success_rate", "slope")
            else:
                # not using other metrics to collect data for now
                continue
            if key == "success_rate_scaling_curve":
                slope_num_factors = self.slope_num_factors
            else:
                if key in ["initial_point", "mid_point"]:
                    slope_num_factors = []
                elif "taylor" in key:
                    slope_num_factors = []
                elif "linear" in key:
                    slope_num_factors = []
                else:
                    slope_num_factors = [1]
            for num_factors_to_collect in slope_num_factors:
                job_ids = []
                for point, r in slope_weights.items():
                    distributed, total = self.turn_weights_to_distribute(
                        point, r[key], num_factors_to_collect
                    )
                    print(f"{method_name}_{num_factors_to_collect}", distributed)
                    job_id = self.submit_jobs(
                        input_string,
                        " ".join(distributed),
                        self.seeds[0],
                        self.exp_name,
                        total_demos=total,
                        curve_name=f"{method_name}_{num_factors_to_collect}",
                    )
                    job_ids.append(job_id)
                total_jobs_submitted += len(job_ids)
                self.echo_command(
                    job_ids,
                    f"{method_name}_{num_factors_to_collect}",
                    file_name="get_test_result.sh",
                )
        print(total_jobs_submitted)

        self.echo_plot_curve_command(
            method="get_test_results", dest="get_test_result.sh"
        )


@hydra.main(
    config_path=os.path.join(os.getcwd(), "cfg"),
    config_name="real_towel_pi0_visual",
    version_base=None,
)
def main(cfg):
    cls = hydra.utils.get_class(cfg._target_)
    exp = cls(cfg)

    method_name = cfg.get("run_method", "main")
    method_args = cfg.get("run_args", {})

    # Call the method dynamically
    method = getattr(exp, method_name)
    method(**method_args)


if __name__ == "__main__":
    main()
