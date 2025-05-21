import itertools
import os
import subprocess
from pathlib import Path

import hydra

from plot_curve import PlotCurve
from scripts.dataset.process_dataset import get_sampled_paths

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


class DPSimBase:
    exp_type = "visual"
    seeds = [0]
    eval_factors = "-tt -dis -b -cp -dir"
    eval_script = "eval_pick_place.sh"
    anomaly_dataset_path = f"{DATA_DIR}/anomaly_dataset/sim_pick_and_place_anomaly"
    img_res = 192
    num_cam = 2
    eval_files = (
        "simulation.eval_base_goal_poses_file=data/eval_poses/eval_grid_pick_poses_60.npy "
        "simulation.eval_base_goal_poses_file=data/eval_poses/eval_grid_place_poses_60.npy "
    )

    @property
    def path_groups(self) -> dict[str, list[str]]:
        return {
            "bg": [f"{DATA_DIR}/raw_data/sim_new_variation/background"],
            "dis": [f"{DATA_DIR}/raw_data/sim_new_variation/distractor"],
            "tt": [f"{DATA_DIR}/raw_data/sim_new_variation/table_texture"],
            "lt": [f"{DATA_DIR}/raw_data/sim_new_variation/directional"],
            "cp": [f"{DATA_DIR}/raw_data/sim_new_variation/camera_pose"],
        }

    @property
    def env_name(self) -> str:
        return "tomato_plate"

    @property
    def policy_name(self) -> str:
        return "dp"

    @property
    def num_factors(self) -> int:
        return len(self.path_groups)

    @property
    def val_dataset_name(self) -> str:
        return f"jsg_jsg_{self.num_cam}cam_{self.img_res}_{self.env_name}_val_{self.num_factors}_factors_seed0_sim{self.num_factors * 5}_real0"

    def __init__(self, cfg):
        self.sim_data_nums = cfg.sim_data_nums
        self.num_train_points = cfg.num_train_points
        self.sample_sim_strat = cfg.sample_sim_strat
        self.k_factor = cfg.k_factor
        self.data_ready = cfg.data_ready
        self.train_ready = cfg.train_ready
        self.run_baseline_equal = cfg.run_baseline_equal
        self.run_baseline_remix = cfg.run_baseline_remix
        self.dry_run = cfg.dry_run
        self.cfg_name = cfg.cfg_name
        self.get_baseline_worst_res = cfg.get("get_baseline_worst_res", True)

        self.slope_num_factors = cfg.slope_num_factors

        self.num_base_demos_per_factor = (
            self.sim_data_nums[self.num_train_points - 1] // self.k_factor
        )
        self.delta = self.sim_data_nums[1] - self.sim_data_nums[0]
        self.delta_equal = [
            n - self.sim_data_nums[self.num_train_points - 1]
            for n in self.sim_data_nums[self.num_train_points :]
        ]
        self.num_per_instance = cfg.num_per_instance
        self.num_instances = cfg.num_instances
        self.combos = (
            cfg.combos
            if cfg.combos is not None
            else self.generate_comb_jobs(self.k_factor)
        )
        self.num_base_demos = self.num_base_demos_per_factor * (
            len(self.path_groups) - self.k_factor
        )

        self.total_demos = [n for n in self.sim_data_nums]

        if cfg.get("exp_name", None) is not None:
            self.exp_name = cfg.get("exp_name")
        else:
            self.exp_name = f"sim_{self.env_name}_{self.exp_type}_{self.policy_name}_{cfg.x_strat}_{self.sample_sim_strat}_base{self.num_base_demos}_vary{self.sim_data_nums[-1]}"
        # print(self.delta_equal, self.num_base_demos_per_factor, self.num_base_demos, self.exp_name)

    def run_command(self, cmd):
        if self.dry_run:
            print(cmd)
        else:
            return subprocess.check_output(cmd, shell=True).decode().strip()

    def turn_weights_to_distribute(self, point, r, num_factors_to_collect):
        delta = int(point) - self.sim_data_nums[self.num_train_points - 1]

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

    def build_distributed_traj(self, vary_keys, sim_data_num):
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
        total = sim_data_num
        vary_paths = sum([len(self.path_groups[k]) for k in vary_keys])
        base = total // vary_paths
        rem = total % vary_paths
        dist_traj += [base] * (vary_paths - 1) + [base + rem]

        return dist_traj

    def main(self):
        num_submitted_jobs = 0
        self.submit_val_data_job(
            " ".join(self.get_input_paths(self.combos[0])), self.seeds[0]
        )
        for vary_keys in self.combos:
            job_ids_exp = []
            vary_suffix = "".join(vary_keys)
            input_paths = " ".join(self.get_input_paths(vary_keys))
            print(vary_keys)
            for seed in self.seeds:
                for sim_data_num in self.sim_data_nums:
                    # sim_data_num = self.sim_data_nums[-1]
                    dist_traj = self.build_distributed_traj(vary_keys, sim_data_num)
                    dist_str = " ".join(map(str, dist_traj))
                    total = sum(dist_traj)
                    print(dist_str)
                    job_id = self.submit_jobs(
                        input_string=input_paths,
                        sim_traj_string=dist_str,
                        seed=seed,
                        exp_name=self.exp_name,
                        total_demos=total,
                        curve_name=vary_suffix,
                    )
                    num_submitted_jobs += 1
                    job_ids_exp.append(job_id)
            print(job_ids_exp)
            self.echo_command(job_ids_exp, vary_suffix)
        if self.run_baseline_equal:
            self.baseline_equal()
            num_submitted_jobs += len(self.delta_equal)
        if self.run_baseline_remix:
            self.baseline_remix()
            num_submitted_jobs += len(self.delta_equal)
        self.echo_plot_curve_command()
        print(f"Total jobs submitted: {num_submitted_jobs}")

    def submit_jobs(
        self,
        input_string,
        sim_traj_string,
        seed,
        exp_name,
        total_demos,
        curve_name,
        num_per_instance=None,
        train_job_id=None,
        num_instances=None,
    ):
        first_line = self.submit_data_job(
            input_string,
            sim_traj_string,
            seed,
            exp_name,
            curve_name,
            num_per_instance,
            num_instances,
        )
        eval_first_line, eval_second_line, job_id_to_return = self.submit_training_job(
            first_line, exp_name, seed, total_demos, curve_name, train_job_id
        )
        self.submit_eval_job(eval_first_line, eval_second_line, curve_name)
        return job_id_to_return

    def submit_data_job(
        self,
        input_string,
        sim_traj_string,
        seed,
        exp_name,
        curve_name,
        num_per_instance=None,
        num_instances=None,
    ):
        if not self.data_ready:
            if self.num_cam == 1:
                line = "-c 0"
            else:
                line = "-c 0 2"
            job1 = self.run_command(
                f"sbatch --parsable process_data.sh "
                f"-in {input_string} "
                f"-out data/processed_data "
                f"--num_thread 5 "
                f"--num_sim_traj {sim_traj_string} "
                f"--num_real_traj 0 "
                f"{line} "
                f"--seed {seed} "
                f"--additional_name {exp_name}{curve_name} "
                f"--visualize_image "
                f"--sample_sim_strat {self.sample_sim_strat} "
                f"--delta {self.delta} "
                f"--num_per_instance {self.num_per_instance if num_per_instance is None else num_per_instance} "
                f"--num_instances {self.num_instances if num_instances is None else num_instances} "
                f"-res {self.img_res} {self.img_res} "
            )
            first_line = f"sbatch --parsable --dependency=afterok:{job1} train.sh"
        else:
            first_line = "sbatch --parsable train.sh"

        return first_line

    def submit_val_data_job(
        self,
        input_string,
        seed,
        num_per_instance=None,
        num_instances=None,
    ):
        if self.num_cam == 1:
            line = "-c 0"
        else:
            line = "-c 0 2"
        sim_traj_string = "5 " * self.num_factors
        self.run_command(
            f"sbatch --parsable process_data.sh "
            f"-in {input_string} "
            f"-out data/processed_data "
            f"--num_thread 5 "
            f"--num_sim_traj {sim_traj_string} "
            f"--num_real_traj 0 "
            f"{line} "
            f"--seed {seed} "
            f"--additional_name {self.env_name}_val_{len(self.path_groups)}_factors "
            f"--visualize_image "
            f"--sample_sim_strat val "
            f"--num_per_instance {self.num_per_instance if num_per_instance is None else num_per_instance} "
            f"--num_instances {self.num_instances if num_instances is None else num_instances} "
            f"-res {self.img_res} {self.img_res} "
        )

    def submit_training_job(
        self, first_line, exp_name, seed, total_demos, curve_name, train_job_id=None
    ):
        if not self.train_ready:
            val_line = f"task.val_dataset_name={self.val_dataset_name}"
            anomaly_line = (
                ""
                if self.anomaly_dataset_path is None
                else f"task.anomaly_dataset.images_path={self.anomaly_dataset_path}"
            )
            do_anomaly_line = (
                "task.do_anomaly=false"
                if self.anomaly_dataset_path is None
                else "task.do_anomaly=true"
            )
            job2 = self.run_command(
                f"{first_line} "
                f"simulation={self.env_name} "
                f"task.env={self.env_name} "
                f"task.dataset_name=jsg_jsg_{self.num_cam}cam_{self.img_res}_{exp_name}{curve_name}_seed{seed}_sim{total_demos}_real0 "
                f"policy/model/obs_encoder=resnet "
                f"policy=diffusion "
                f"task.num_views={self.num_cam} "
                f"simulation.num_envs=30 "
                f"{do_anomaly_line} "
                f"task.dataset_type=guided_dc.agent.BaseSequenceDataset "
                f"{val_line} "
                f"{anomaly_line} "
            )

            first_line = (
                f"sbatch --parsable --dependency=afterok:{job2} {self.eval_script}"
            )
            second_line = f"-j {job2}"
            job_id_to_return = job2
        else:
            assert train_job_id is not None, (
                "train_job_id must be provided when train_ready is True"
            )
            first_line = f"sbatch --parsable {self.eval_script}"
            second_line = f"-j {train_job_id}"
            job_id_to_return = train_job_id

        return first_line, second_line, job_id_to_return

    def submit_eval_job(self, eval_first_line, eval_second_line, curve_name):
        job3 = self.run_command(
            f"{eval_first_line} "
            f"{eval_second_line} "
            f"--env_name {self.env_name} "
            f"{self.eval_factors} "
            f"--num_eval_instances 5 "
            f"simulation.num_envs=30 "
            f"simulation.record.output_dir={os.path.join(RESULTS_DIR, self.exp_name, curve_name)} "
            f"{self.eval_files}"
        )
        # if self.dry_run:
        #     os.makedirs(os.path.join(RESULTS_DIR, self.exp_name, curve_name, "2102123", "100"), exist_ok=True)
        #     os.makedirs(os.path.join(RESULTS_DIR, self.exp_name, curve_name, "2102123", "100", "background_0"), exist_ok=True)
        #     with open(os.path.join(RESULTS_DIR, self.exp_name, curve_name, "2102123", "100", "background_0", "10_30_0_0.txt"), 'w') as f:
        #         f.write("dummy txt file")
        #     with open(os.path.join(RESULTS_DIR, self.exp_name, curve_name, "2102123", "100", "22.1%.txt"), 'w') as f:
        #         f.write("dummy txt file")

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
                total_demos=total,
                curve_name="baseline_equal",
            )
            job_ids_equal.append(job_id)
        self.echo_command(job_ids_equal, "baseline_equal")

    def baseline_remix(self):
        # Baseline equal

        # for delta in self.delta_equal:
        #     base_part = delta // len(self.path_groups)
        #     rem = delta % len(self.path_groups)
        #     split = [(base_part + 1 if i < rem else base_part) for i in range(len(self.path_groups))]

        #     distributed = []
        #     total = 0
        #     for i, (_, _) in enumerate(self.path_groups.items()):
        #         val = self.num_base_demos_per_factor + split[i]
        #         distributed.append(val)
        #         total += val

        # 1. Get data paths
        train_paths_list = get_sampled_paths(
            input_paths=[v[0] for v in self.path_groups.values()],
            num_sim_traj=[self.num_base_demos_per_factor] * len(self.path_groups),
            num_real_traj=0,
            sample_sim_strat=self.sample_sim_strat,
            num_instances=self.num_instances,
            num_per_instance=self.num_per_instance,
            delta=self.delta,
        )

        # Save train_paths_list to a txt file
        train_paths_list_file = "./train_paths.txt"
        os.makedirs(os.path.dirname(train_paths_list_file), exist_ok=True)
        with open(train_paths_list_file, "w") as f:
            for path in train_paths_list:
                f.write(f"{path}\n")

        self.dry_run = False
        self.run_command("python split.py --list_file ./train_paths.txt")
        self.dry_run = True

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

        cmd = f"python submit_dp_jobs.py --config-name {self.cfg_name} run_method={method}"
        script_file = Path(os.path.join(RESULTS_DIR, self.exp_name, dest))
        script_file.write_text(
            script_file.read_text() + f"\n{cmd}" if script_file.exists() else cmd
        )
        script_file.chmod(0o755)

    def weighted_slope(self):
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
            base_demos=self.num_base_demos,
            with_all_results=False,
            get_baseline_worst_res=True,
        )
        slope_weights, worst_weights = curve_plotter.get_train_results()
        input_string = " ".join(sum(self.path_groups.values(), []))

        total_jobs_submitted = 0

        _keys = ["success_rate_scaling_curve"]
        # _keys = ["1_distances_scaling_curve", "1_normalized_distances_scaling_curve",
        #          "5_normalized_distances_scaling_curve", "5_distances_scaling_curve",
        #          "10_normalized_distances_scaling_curve", "10_distances_scaling_curve"]
        #  "val_loss_scaling_curve"]

        # for key in next(iter(slope_weights.values())):
        for key in _keys:
            # if "success_rate" in key:
            #     method_name = key.replace("success_rate", "slope")
            # else:
            #     # not using other metrics to collect data for now
            #     continue
            # if key == "success_rate_scaling_curve":
            #     slope_num_factors = self.slope_num_factors
            # else:
            #     if key in ["initial_point", "mid_point"]:
            #         slope_num_factors = []
            #     elif "taylor" in key:
            #         slope_num_factors = []
            #     elif "linear" in key:
            #         slope_num_factors = []
            #     else:
            #         slope_num_factors = [1]
            method_name = key
            for num_factors_to_collect in self.slope_num_factors:
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

    def get_test_results(self):
        curve_plotter = PlotCurve(
            exp_name=self.exp_name,
            env_name=self.env_name,
            num_points=self.num_train_points,
            use_sim=True,
            show_power_law=True,
            move_folder=False,
            set_avg_final_sr=True,
            set_avg_final_metric=False,
            num_worst_factors=self.k_factor,
            points=self.total_demos,
            with_all_results=False,
            base_demos=self.num_base_demos,
        )
        curve_plotter.get_test_results()


class DPSimPegVisual(DPSimBase):
    exp_type = "visual"
    seeds = [0]
    eval_factors = "-tt -dis -b -cp -dir"
    eval_script = "eval_peg_pull.sh"
    anomaly_dataset_path = f"{DATA_DIR}/anomaly_dataset/sim_peg_visual_anomaly"
    img_res = 256
    num_cam = 2
    eval_files = (
        "simulation.eval_base_manip_poses_file=data/eval_poses/eval_base_manip_poses_file_peg_60.npy "
        "simulation.eval_base_goal_poses_file=data/eval_poses/eval_base_goal_poses_file_peg_60.npy "
    )

    @property
    def path_groups(self):
        return {
            "bg": [f"{DATA_DIR}/raw_data/sim_peg_one_factor_130/background"],
            "dis": [f"{DATA_DIR}/raw_data/sim_peg_one_factor_130/distractor"],
            "tt": [f"{DATA_DIR}/raw_data/sim_peg_one_factor_130/table_texture"],
            "lt": [f"{DATA_DIR}/raw_data/sim_peg_one_factor_130/directional"],
            "cp": [f"{DATA_DIR}/raw_data/sim_peg_one_factor_130/camera_pose"],
        }

    @property
    def env_name(self):
        return "peg_insertion"


class DPSimPegSpatial(DPSimBase):
    exp_type = "spatial"
    seeds = [0]
    eval_factors = "-tt -dis -b -cp -dir -op -th -dq"
    eval_script = "eval_peg_pull.sh"
    anomaly_dataset_path = f"{DATA_DIR}/anomaly_dataset/sim_peg_spatial_anomaly"
    img_res = 256
    num_cam = 2
    eval_files = (
        "simulation.eval_base_manip_poses_file=data/eval_poses/eval_base_manip_poses_file_peg_60.npy "
        "simulation.eval_base_goal_poses_file=data/eval_poses/eval_base_goal_poses_file_peg_60.npy "
        "simulation.eval_factor_manip_poses_file=data/eval_poses/eval_factor_manip_poses_file_peg_600.npy "
        "simulation.eval_factor_goal_poses_file=data/eval_poses/eval_factor_goal_poses_file_peg_600.npy "
        "simulation.eval_base_delta_qpos_file=data/eval_poses/eval_base_delta_qpos_file_peg_60.npy "
        "simulation.eval_factor_delta_qpos_file=data/eval_poses/eval_factor_delta_qpos_file_peg_600.npy"
    )

    @property
    def env_name(self):
        return "peg_insertion"

    @property
    def path_groups(self):
        return {
            "bg": [f"{DATA_DIR}/raw_data/sim_peg/background"],
            "dis": [f"{DATA_DIR}/raw_data/sim_peg/distractor"],
            "tt": [f"{DATA_DIR}/raw_data/sim_peg/table_texture"],
            "lt": [f"{DATA_DIR}/raw_data/sim_peg/directional"],
            "cp": [f"{DATA_DIR}/raw_data/sim_peg/camera_pose"],
            "dq": [f"{DATA_DIR}/raw_data/sim_peg/delta_qpos"],
            "th": [f"{DATA_DIR}/raw_data/sim_peg/table_height"],
            "op": [f"{DATA_DIR}/raw_data/sim_peg/obj_pose"],
        }


class DPSimPegSpatialAblation(DPSimBase):
    exp_type = "spatial"
    seeds = [0]
    eval_factors = "-tt -dis -b -cp -dir -op -th -dq"
    eval_script = "eval_peg_pull.sh"
    anomaly_dataset_path = f"{DATA_DIR}/anomaly_dataset/sim_peg_spatial_anomaly"
    img_res = 256
    num_cam = 2
    eval_files = (
        "simulation.eval_base_manip_poses_file=data/eval_poses/eval_base_manip_poses_file_peg_60.npy "
        "simulation.eval_base_goal_poses_file=data/eval_poses/eval_base_goal_poses_file_peg_60.npy "
        "simulation.eval_factor_manip_poses_file=data/eval_poses/eval_factor_manip_poses_file_peg_600.npy "
        "simulation.eval_factor_goal_poses_file=data/eval_poses/eval_factor_goal_poses_file_peg_600.npy "
        "simulation.eval_base_delta_qpos_file=data/eval_poses/eval_base_delta_qpos_file_peg_60.npy "
        "simulation.eval_factor_delta_qpos_file=data/eval_poses/eval_factor_delta_qpos_file_peg_600.npy"
    )

    @property
    def env_name(self):
        return "peg_insertion"

    @property
    def path_groups(self):
        return {
            "bg": [f"{DATA_DIR}/raw_data/sim_peg_ablation/background"],
            "dis": [f"{DATA_DIR}/raw_data/sim_peg_ablation/distractor"],
            "tt": [f"{DATA_DIR}/raw_data/sim_peg_ablation/table_texture"],
            "lt": [f"{DATA_DIR}/raw_data/sim_peg_ablation/directional"],
            "cp": [f"{DATA_DIR}/raw_data/sim_peg_ablation/camera_pose"],
            "dq": [f"{DATA_DIR}/raw_data/sim_peg_ablation/delta_qpos"],
            "th": [f"{DATA_DIR}/raw_data/sim_peg_ablation/table_height"],
            "op": [f"{DATA_DIR}/raw_data/sim_peg_ablation/obj_pose"],
        }


class DPSimPullSpatial(DPSimBase):
    exp_type = "spatial"
    seeds = [0]
    eval_factors = "-tt -dis -b -cp -dir -op -th -dq"
    eval_script = "eval_peg_pull.sh"
    anomaly_dataset_path = f"{DATA_DIR}/anomaly_dataset/sim_pull_spatial_anomaly"
    img_res = 192
    num_cam = 1
    eval_files = (
        "simulation.eval_base_manip_poses_file=data/eval_poses/eval_base_manip_poses_file_pull_60.npy "
        "simulation.eval_base_goal_poses_file=data/eval_poses/eval_base_goal_poses_file_pull_60.npy "
        "simulation.eval_factor_manip_poses_file=data/eval_poses/eval_factor_manip_poses_file_pull_600.npy "
        "simulation.eval_factor_goal_poses_file=data/eval_poses/eval_factor_goal_poses_file_pull_600.npy "
        "simulation.eval_base_delta_qpos_file=data/eval_poses/eval_base_delta_qpos_file_pull_60.npy "
        "simulation.eval_factor_delta_qpos_file=data/eval_poses/eval_factor_delta_qpos_file_pull_600.npy"
    )

    @property
    def env_name(self):
        return "pull_cube_tool"

    @property
    def path_groups(self):
        return {
            "dis": [f"{DATA_DIR}/raw_data/sim_pull_cube_tool_45/distractor"],
            "tt": [f"{DATA_DIR}/raw_data/sim_pull_cube_tool_45/table_texture"],
            "lt": [f"{DATA_DIR}/raw_data/sim_pull_cube_tool_45/directional"],
            "cp": [f"{DATA_DIR}/raw_data/sim_pull_cube_tool_45/camera_pose"],
            "bg": [f"{DATA_DIR}/raw_data/sim_pull_cube_tool_45/background"],
            "dq": [f"{DATA_DIR}/raw_data/sim_pull_cube_tool_45/delta_qpos"],
            "th": [f"{DATA_DIR}/raw_data/sim_pull_cube_tool_45/table_height"],
            "op": [f"{DATA_DIR}/raw_data/sim_pull_cube_tool_45/obj_pose"],
        }


class DPSimPullVisual(DPSimBase):
    exp_type = "visual"
    seeds = [0]
    eval_factors = "-tt -dis -cp -dir -b"
    eval_script = "eval_peg_pull.sh"
    anomaly_dataset_path = f"{DATA_DIR}/anomaly_dataset/sim_pull_visual_anomaly"
    img_res = 192
    num_cam = 1
    eval_files = (
        "simulation.eval_base_manip_poses_file=data/eval_poses/eval_base_manip_poses_file_pull_60.npy "
        "simulation.eval_base_goal_poses_file=data/eval_poses/eval_base_goal_poses_file_pull_60.npy "
        "simulation.eval_factor_manip_poses_file=data/eval_poses/eval_factor_manip_poses_file_pull_600.npy "
        "simulation.eval_factor_goal_poses_file=data/eval_poses/eval_factor_goal_poses_file_pull_600.npy "
        "simulation.eval_base_delta_qpos_file=data/eval_poses/eval_base_delta_qpos_file_pull_60.npy "
        "simulation.eval_factor_delta_qpos_file=data/eval_poses/eval_factor_delta_qpos_file_pull_600.npy"
    )

    @property
    def env_name(self):
        return "pull_cube_tool"

    @property
    def path_groups(self):
        return {
            "dis": [f"{DATA_DIR}/raw_data/sim_pull_cube_one_factor_130/distractor"],
            "tt": [f"{DATA_DIR}/raw_data/sim_pull_cube_one_factor_130/table_texture"],
            "lt": [f"{DATA_DIR}/raw_data/sim_pull_cube_one_factor_130/lighting"],
            "cp": [f"{DATA_DIR}/raw_data/sim_pull_cube_one_factor_130/camera_pose"],
            "bg": [f"{DATA_DIR}/raw_data/sim_pull_cube_one_factor_130/background"],
        }


class DPReal(DPSimBase):
    exp_type = "visual"
    seeds = [0]
    eval_factors = "-tt -dis -b -cp -dir"
    eval_script = "eval_pick_place.sh"
    anomaly_dataset_path = f"{DATA_DIR}/anomaly_dataset/real_pick_and_place_anomaly"
    img_res = 192
    num_cam = 2
    eval_files = (
        "simulation.eval_base_goal_poses_file=data/eval_poses/eval_grid_pick_poses_60.npy "
        "simulation.eval_base_goal_poses_file=data/eval_poses/eval_grid_place_poses_60.npy "
    )

    @property
    def path_groups(self) -> dict[str, list[str]]:
        return {
            "bg": [f"{DATA_DIR}/raw_data/background_new"],
            "dis": [f"{DATA_DIR}/raw_data/distractor_new"],
            "tt": [f"{DATA_DIR}/raw_data/table_texture_new"],
            "lt": [f"{DATA_DIR}/raw_data/lighting_new"],
            "cp": [f"{DATA_DIR}/raw_data/camera_pose_new_new"],
        }

    @property
    def env_name(self) -> str:
        return "tomato_plate"

    @property
    def policy_name(self) -> str:
        return "dp"

    @property
    def val_dataset_name(self) -> str:
        return f"jsg_jsg_{self.num_cam}cam_{self.img_res}_real_{self.env_name}_val_{self.num_factors}_factors_seed0_sim{self.num_factors * 5}_real0"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.exp_name = cfg.get("exp_name", None)

    def submit_val_data_job(
        self,
        input_string,
        seed,
        num_per_instance=None,
        num_instances=None,
    ):
        if self.num_cam == 1:
            line = "-c 0"
        else:
            line = "-c 0 2"
        sim_traj_string = "5 " * self.num_factors
        self.run_command(
            f"sbatch --parsable process_data.sh "
            f"-in {input_string} "
            f"-out data/processed_data "
            f"--num_thread 5 "
            f"--num_sim_traj 0 "
            f"--num_real_traj {sim_traj_string} "
            f"{line} "
            f"--seed {seed} "
            f"--additional_name real_{self.env_name}_val_{len(self.path_groups)}_factors "
            f"--visualize_image "
            f"--sample_sim_strat val "
            f"--num_per_instance {self.num_per_instance if num_per_instance is None else num_per_instance} "
            f"--num_instances {self.num_instances if num_instances is None else num_instances} "
            f"-res {self.img_res} {self.img_res} "
        )


@hydra.main(
    config_path=os.path.join(os.getcwd(), "cfg"),
    config_name="sim_peg_insertion_dp_visual",
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
