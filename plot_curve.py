import argparse
import glob
import json
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

RESULTS_DIR = os.getenv(
    "GDC_RESULTS_DIR", "/n/fs/robot-data/factored-scaling-curves/results"
)

# rcParams["font.family"] = "serif"
# rcParams["font.serif"] = ["Charter"]
# if you use math text and want it in Charter as well:
# rcParams["mathtext.fontset"] = "custom"
# rcParams["mathtext.rm"] = "Charter"
rcParams["mathtext.it"] = "Charter:italic"
rcParams["mathtext.bf"] = "Charter:bold"


def convert_key(key):
    """Convert key to int if possible, otherwise to float."""
    num = float(key)
    return int(num) if num.is_integer() else num


# def power_law_from_origin_log(x, a, b):
#     # return 1 - a * ((x + a**(-1/b))**b)
#     return np.log(a) + b*np.log(x+a**(-1/b))

# def power_law_from_origin(x, a, b):
#     # return 1 - a * ((x + a**(-1/b))**b)
#     return 1 - a * (x+a**(-1/b))**b


def power_law_log(x, a, b, num_base):
    # return 1 - a * ((x + a**(-1/b))**b)
    return np.log(a) + b * np.log(x + num_base)


def power_law(x, a, b, num_base):
    # return 1 - a * ((x + a**(-1/b))**b)
    return 1 - a * (x + num_base) ** b


def taylor_expansion_power_law_flipped(x, a, b, x0):
    term0 = 1 - a * x0**b
    term1 = -a * b * x0 ** (b - 1) * (x - x0)
    term2 = -0.5 * a * b * (b - 1) * x0 ** (b - 2) * (x - x0) ** 2
    term3 = -(1 / 6) * a * b * (b - 1) * (b - 2) * x0 ** (b - 3) * (x - x0) ** 3
    # term4 = -(1/24) * a * b * (b - 1) * (b - 2) * (b - 3) * x0**(b - 4) * (x-x0)**4
    return term0 + term1 + term2 + term3


# def power_law(x, a, b):
#     return a * (x**b)


def taylor_expansion_power_law(x, a, b, x0):
    term0 = a * x0**b
    term1 = a * b * x0 ** (b - 1) * (x - x0)
    term2 = 0.5 * a * b * (b - 1) * x0 ** (b - 2) * (x - x0) ** 2
    term3 = (1 / 6) * a * b * (b - 1) * (b - 2) * x0 ** (b - 3) * (x - x0) ** 3
    # term4 = (1/24) * a * b * (b - 1) * (b - 2) * (b - 3) * x0**(b - 4) * (x-x0)**4
    return term0 + term1 + term2 + term3


def linear(x, a, b):
    return a * x + b


def extract_first_two_numbers(filename):
    match = re.match(r"(\d+)_(\d+)_(\d+)_\d+\.txt", filename)
    if match:
        first_number = int(match.group(1))
        second_number = int(match.group(2))
        return first_number, second_number
    else:
        raise ValueError("Filename does not match the expected format")


def extract_first_factor(filename):
    match = re.match(r"(\w+_\d+)", filename)
    if match:
        return match.group(0)  # Returns the first factor_num like "table_texture_10"
    else:
        raise ValueError("Filename does not match the expected format")


def extract_single_number(s):
    numbers = re.findall(r"\d+", s)
    if len(numbers) != 1:
        raise ValueError(
            f"Expected exactly one numeric sequence in '{s}', found {len(numbers)}."
        )
    return int(numbers[0])


def get_results_from_predictions(
    title, points, predictions, baseline, anticipated_baseline
):
    factor_name = title
    results = {int(point): {} for point in points}

    for point, prediction in zip(points, predictions):
        anticipated_value = prediction[1]
        true_value = prediction[2]

        delta_anticipated = anticipated_value - anticipated_baseline

        if true_value is None:
            delta = None
            delta_ratio = None
            real_ratio = None
        else:
            delta = true_value - baseline
            delta_ratio = delta / baseline
            real_ratio = true_value / baseline

        results[int(point)][factor_name] = (
            anticipated_value,
            delta_anticipated,
            true_value,
            delta,
            delta_ratio,
            real_ratio,
        )

    return results


def better_plot(
    ax,
    x_fit,
    y_fit,
    curve_label,
    points,
    num_points,
    fit_function,
    params,
    values,
    show_point_label,
    color_fit,
    label,
    y_min=0,
    y_max=1,
    title=None,
    x_ext=None,
    y_ext=None,
    color_history="#87CEFA",  # Ground-truth points (orange)
    color_pred="#FFA500",  # Predicted points (light blue)
):
    if "distance" in label or "anomaly" in label:
        color_history = "#D4A3D9"  # Ground-truth points (orange)
        color_pred = "#936099"  # Predicted points (light blue)
    # Plot historical ground-truth
    ax.scatter(
        points[:num_points],
        values[:num_points],
        label="Reference Points",
        color=color_history,
        marker="o",
        s=150,
    )  # Increased marker size

    # Offset amount to avoid text clipping

    # for x, y in zip(points[:num_points], values[:num_points]):
    #     txt = ax.text(x, y + offset, f"{y:.2f}", ha="center", va="bottom", fontsize=16, color=color_history)  # Increased font size
    #     texts.append(txt)

    # Plot fit curve
    ax.plot(
        x_fit, y_fit, "-", label=curve_label, color=color_history, linewidth=4
    )  # Increased line width
    ax.plot(
        x_ext, y_ext, "--", color=color_history, linewidth=4
    )  # Increased line width

    ## Calculate R²
    # y_actual = values[:num_points]
    # y_predicted = [fit_function(x, *params) for x in points[:num_points]]
    # ss_res = sum((y_actual - y_predicted) ** 2)
    # ss_tot = sum((y_actual - sum(y_actual) / len(y_actual)) ** 2)
    # r_squared = 1 - (ss_res / ss_tot)

    # Add R² as text in the top-left corner
    # ax.text(
    #     # 0.7, 0.1,  # Relative position in axes coordinates
    #     0.02, 0.98,
    #     f"R² = {r_squared:.2f}",
    #     transform=ax.transAxes,  # Use axes coordinates
    #     fontsize=16,  # Font size
    #     verticalalignment="top",
    #     horizontalalignment="left",
    #     color="black"
    # )

    # Predicted values
    predictions = []
    for idx, x in enumerate(points[num_points:]):
        y_pred = fit_function(x, *params)
        if len(values) == len(points):
            y_gt = values[num_points + idx]
        else:
            y_gt = None
        predictions.append((x, y_pred, y_gt))

        sc = ax.scatter(
            x,
            y_pred,
            marker="o",
            s=150,  # Increased marker size
            facecolor=color_history,  # Interior color
            edgecolor="black",  # Border color
            linewidths=1.5,
            zorder=3,
            label=("Predicted Points" if idx == 0 and show_point_label else None),
        )
        sc.set_linestyle("--")

        # txt = ax.text(x, y_pred + offset, f"{y_pred:.2f}", ha="center", va="bottom", fontsize=16, color=color_pred)  # Increased font size
        # texts.append(txt)

    if len(values) == len(points):
        # Plot extrapolated ground-truth
        ax.scatter(
            points[num_points:],
            values[num_points:],
            label="Test Points" if show_point_label else None,
            color=color_pred,
            marker="D",
            s=150,
            edgecolors=color_pred,
            linewidths=0.5,
            zorder=3,
        )  # Increased marker size

        # for x, y in zip(points[num_points:], values[num_points:]):
        #     txt = ax.text(x, y - 2*offset, f"{y:.2f}", ha="center", va="bottom", fontsize=16, color=color_pred)  # Increased font size
        #     texts.append(txt)

    # Axis and layout
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Number of Factor Data", fontsize=18)  # Increased font size
    ax.set_ylabel("Success Rate", fontsize=18)  # Increased font size
    if title:
        ax.set_title(title, fontsize=20)  # Increased font size

    ax.tick_params(
        axis="both", which="major", labelsize=16
    )  # Increased tick label font size
    # ax.legend(
    # fontsize=20, loc="upper left", frameon=False
    # )  # Increased legend font size
    ax.grid(True, linestyle="--", alpha=0.2)

    return predictions


def get_predictions_by_strat(
    ax,
    points,
    values,
    strat,
    num_points,
    show_power_law,
    label,
    color_predicted="orange",
    color_gt="orange",
    color_fit="purple",
    show_point_label=True,
    y_min=0,
    y_max=1,
    title=None,
    num_base_demos=None,
):
    assert show_power_law

    print(label, strat)

    valid_indices = (
        ~np.isnan(values[:num_points])
        & ~np.isinf(values[:num_points])
        & ~np.isnan(points[:num_points])
        & ~np.isinf(points[:num_points])
    )

    # fit curve in log scale
    filtered_points = points[:num_points][valid_indices]
    filtered_values = np.log(1 - values[:num_points][valid_indices])

    num_base = num_base_demos

    if "three_points_first" in strat and len(filtered_points) > 3:
        filtered_points = filtered_points[[0, 1, -1]]
        filtered_values = filtered_values[[0, 1, -1]]

    elif "three_points_last" in strat and len(filtered_points) > 3:
        filtered_points = filtered_points[1:]
        filtered_values = filtered_values[1:]

    if "scaling_curve" in strat and "taylor" not in strat:
        fit_function = lambda x, a, b: power_law_log(x, a, b, num_base=num_base)  # noqa: E731
        try:
            params, _ = curve_fit(
                fit_function,
                filtered_points,
                filtered_values,
                p0=[1, -1],
                bounds=((10e-10, -1000), (10e4, -0.01)),
            )
        except RuntimeError:
            print(f"Curve fitting failed for {label} with strategy {strat}.")
            params = [1, -1]
        # Fit curve
        a_fit, b_fit = params
        fit_function = lambda x, a, b: power_law(x, a, b, num_base=num_base)  # noqa: E731
        x_fit = np.linspace(points[0], points[num_points - 1], 100)
        y_fit = fit_function(x_fit, *params)
        x_ext = np.linspace(points[num_points - 1], points[-1], 100)
        y_ext = fit_function(x_ext, *params)
        # if label == "SR":
        # curve_label = f"{label}: 1 - {a_fit:.6f} * (x + {a_fit**(-1/b_fit):.3f})^{b_fit:.2f}"
        curve_label = rf"$\Phi_{{i,j}} = 1 - {a_fit:.2f}\,\cdot\,(x + {num_base_demos})^{{{b_fit:.2f}}}$"
        # else:
        # curve_label = f"{label}: {a_fit:.6f} * x^{b_fit:.2f}"

    elif "scaling_curve_taylor" in strat:
        x0 = np.mean(filtered_points)
        if label == "SR":
            fit_function = lambda x, a, b: taylor_expansion_power_law_flipped(
                x, a, b, x0
            )
        else:
            fit_function = lambda x, a, b: taylor_expansion_power_law(
                x=x, a=a, b=b, x0=x0
            )

        params, _ = curve_fit(
            fit_function,
            filtered_points,
            filtered_values,
            p0=[1, -1],
            bounds=((0.0001, -1000), (1000, -0.01)),
        )
        # Fit curve
        a_fit, b_fit = params
        # Generate smooth curve for plotting
        x_fit = np.linspace(min(points), max(points), 100)
        y_fit = fit_function(x_fit, *params)
        if label == "SR":
            curve_label = (
                f"{label} (Taylor, flipped): "
                f"1 - {a_fit:.6f} * x0^{b_fit:.2f} "
                f"- {a_fit:.6f} * {b_fit:.2f} * x0^{b_fit - 1:.2f} * (x - x0) "
                f"- 0.5 * {a_fit:.6f} * {b_fit:.2f} * ({b_fit - 1:.2f}) * x0^{b_fit - 2:.2f} * (x - x0)^2 "
                f"- 1/6 * {a_fit:.6f} * {b_fit:.2f} * ({b_fit - 1:.2f}) * ({b_fit - 2:.2f}) * x0^{b_fit - 3:.2f} * (x - x0)^3"
            )
        else:
            curve_label = (
                f"{label} (Taylor): "
                f"{a_fit:.6f} * x0^{b_fit:.2f} "
                f"+ {a_fit:.6f} * {b_fit:.2f} * x0^{b_fit - 1:.2f} * (x - x0) "
                f"+ 0.5 * {a_fit:.6f} * {b_fit:.2f} * ({b_fit - 1:.2f}) * x0^{b_fit - 2:.2f} * (x - x0)^2 "
                f"+ 1/6 * {a_fit:.6f} * {b_fit:.2f} * ({b_fit - 1:.2f}) * ({b_fit - 2:.2f}) * x0^{b_fit - 3:.2f} * (x - x0)^3"
            )

    elif "linear" in strat:
        params, _ = curve_fit(linear, filtered_points, filtered_values)
        # Fit curve
        a_fit, b_fit = params
        # Generate smooth curve for plotting
        x_fit = np.linspace(min(points), max(points), 100)
        y_fit = linear(x_fit, *params)
        curve_label = f"{label}: {a_fit:.3f} * x + {b_fit:.3f}"
        fit_function = linear

    elif strat == "initial_point":
        x_fit = np.linspace(min(points), max(points), 100)
        y_fit = values[0] + (values[num_points - 1] - values[0]) * (
            x_fit - points[0]
        ) / (points[num_points - 1] - points[0])
        curve_label = f"{label}: {values[0]:.3f} + ({values[num_points - 1] - values[0]:.3f}) * (x - {points[0]:.3f}) / ({points[num_points - 1] - points[0]:.3f})"
        params = [
            values[0],
            values[num_points - 1] - values[0],
            points[0],
            points[num_points - 1],
        ]
        fit_function = lambda x, a, b, c, d: a + b * (x - c) / (d - c)

    elif strat == "mid_point":
        x_fit = np.linspace(min(points), max(points), 100)
        y_fit = values[1] + (values[num_points - 1] - values[1]) * (
            x_fit - points[1]
        ) / (points[num_points - 1] - points[1])
        curve_label = f"{label}: {values[1]:.3f} + ({values[num_points - 1] - values[1]:.3f}) * (x - {points[1]:.3f}) / ({points[num_points - 1] - points[1]:.3f})"
        params = [
            values[1],
            values[num_points - 1] - values[1],
            points[1],
            points[num_points - 1],
        ]
        fit_function = lambda x, a, b, c, d: a + b * (x - c) / (d - c)

    else:
        raise ValueError(f"Unknown strategy: {strat}")

    prediction_for_current_pt = fit_function(filtered_points[-1], *params)

    predictions = better_plot(
        ax=ax,
        x_fit=x_fit,
        y_fit=y_fit,
        x_ext=x_ext,
        y_ext=y_ext,
        curve_label=curve_label,
        label=label,
        points=points,
        num_points=num_points,
        fit_function=fit_function,
        params=params,
        values=values,
        show_point_label=show_point_label,
        color_fit=color_fit,
        y_min=y_min,
        y_max=y_max,
        title=title,
    )

    return predictions, prediction_for_current_pt


def calculate_factor_sr_per_x(
    jobs=f"{RESULTS_DIR}/ct_1s_base120_vary60tt_clusterpick/1864042_grid",
    name="tt_cluster_pick_120",
    save_fig=False,
):
    # success_total = {f"background_{idx}": [] for idx in range(10)}
    success_total = {}
    success_total.update({f"camera_pose_{idx}": [] for idx in range(10)})
    success_total.update({f"directional_{idx}": [] for idx in range(10)})
    success_total.update({f"table_texture_{idx}": [] for idx in range(10)})
    success_total.update({f"background_{idx}": [] for idx in range(10)})
    success_total.update({f"distractor_{idx}": [] for idx in range(10)})
    success_total.update({f"delta_qpos_{idx}": [] for idx in range(10)})
    success_total.update({f"table_height_{idx}": [] for idx in range(10)})
    success_total.update({f"obj_pose_{idx}": [] for idx in range(10)})

    for ckpts in os.listdir(jobs):
        # print(ckpts)
        _ckpts = os.path.join(jobs, ckpts)
        for evals in os.listdir(_ckpts):
            if not os.path.isdir(os.path.join(_ckpts, evals)):
                continue
            first_factor = extract_first_factor(evals)
            _evals = os.path.join(_ckpts, evals)
            for files in os.listdir(_evals):
                if files.endswith(".txt"):
                    first_number, second_number = extract_first_two_numbers(files)
                    success_total[first_factor].append([first_number, second_number])

    keys_to_pop = []
    for factor_name, suc_tot in success_total.items():
        if len(suc_tot) == 0:
            keys_to_pop.append(factor_name)

    for factor_name in keys_to_pop:
        success_total.pop(factor_name)

    success_rate = {}
    for factor_name, suc_tot in success_total.items():
        # print(len(suc_tot))
        sucs = 0
        tots = 0
        for suc, tot in suc_tot:
            sucs += suc
            tots += tot
        if tots != 0:
            success_rate[factor_name] = sucs / tots
        else:
            continue
    # Plot a histogram of the success rates, with the x-axis labeled by the factor names
    if save_fig:
        plt.figure(figsize=(20, 8))  # Make figure wider
        plt.bar(success_rate.keys(), success_rate.values())

        # Rotate and adjust labels
        plt.xticks(rotation=60, fontsize=10, ha="right")  # 'ha' ensures right alignment

        # Labels and title
        plt.ylabel("Success Rate", fontsize=14)
        plt.xlabel("Factor", fontsize=14)
        plt.title("Success Rates by Factor", fontsize=16)

        plt.tight_layout()  # Prevent labels from being cut off
        plt.savefig(f"success_rates_{name}.png", dpi=300)

    return success_rate


def get_factor_sr(job_folder):
    success_rate = calculate_factor_sr_per_x(jobs=job_folder, save_fig=False)
    success_rate_per_factor = {
        "background": [],
        "camera_pose": [],
        "distractor": [],
        "directional": [],
        "table_texture": [],
        "table_height": [],
        "delta_qpos": [],
        "obj_pose": [],
    }

    for k, v in success_rate.items():
        if "background" in k:
            success_rate_per_factor["background"].append(v)
        elif "camera_pose" in k:
            success_rate_per_factor["camera_pose"].append(v)
        elif "distractor" in k:
            success_rate_per_factor["distractor"].append(v)
        elif "directional" in k:
            success_rate_per_factor["directional"].append(v)
        elif "table_texture" in k:
            success_rate_per_factor["table_texture"].append(v)
        elif "table_height" in k:
            success_rate_per_factor["table_height"].append(v)
        elif "delta_qpos" in k:
            success_rate_per_factor["delta_qpos"].append(v)
        elif "obj_pose" in k:
            success_rate_per_factor["obj_pose"].append(v)

    ks = list(success_rate_per_factor.keys())
    for k in ks:
        if success_rate_per_factor[k] == []:
            success_rate_per_factor.pop(k)

    success_rate_per_factor = {
        k: [float(np.mean(v))] for k, v in success_rate_per_factor.items()
    }

    return success_rate_per_factor


def find_correlation_between_metrics(result_dict, corr_metric="norm1"):
    gt_rank = {}
    metric_rank = {}
    for point, results in result_dict.items():
        metric_rank[point] = {}
        for strat, result_list in results.items():
            if strat == "success_rate_scaling_curve":
                gt_rank[point] = result_list[1]
            else:
                metric_rank[point].update({strat: result_list[1]})
    # Calculate the correlation coefficient
    correlation_dict = {}
    if corr_metric == "norm1":
        for point, metric_results in metric_rank.items():
            correlation_dict[point] = {}
            for strat, rank in metric_results.items():
                if "baseline" in strat:
                    continue
                gt_rank_value = gt_rank[point]
                correlation = np.abs(gt_rank_value - rank).item()

                correlation_dict[point][strat] = correlation
                # print(f"Point: {point}, Strat: {strat}, Correlation: {correlation:.3f}")
    elif corr_metric == "rank_violation":
        for point_idx, (point, metric_results) in enumerate(metric_rank.items()):
            for strat, rank in metric_results.items():
                if "baseline" in strat:
                    continue
                if point_idx == 0:
                    correlation_dict[strat] = []

                gt_rank_value_i = gt_rank[point]
                rank_i = rank
                rank_violations = []
                for point, metric_results in metric_rank.items():
                    gt_rank_value_j = gt_rank[point]
                    rank_j = metric_results[strat]
                    rank_violation = np.abs(gt_rank_value_i - gt_rank_value_j) * (
                        (gt_rank_value_i < gt_rank_value_j) != (rank_i < rank_j)
                    )
                    rank_violations.append(rank_violation)

                max_rank_violation_i = max(rank_violations)
                correlation_dict[strat].append(max_rank_violation_i)
        for strat, max_rank_violation_for_all_points in correlation_dict.items():
            correlation_dict[strat] = np.mean(max_rank_violation_for_all_points)
    return correlation_dict


class PlotCurve:
    def __init__(
        self,
        exp_name,
        env_name,
        num_points=4,
        use_sim=True,
        set_avg_final_sr=False,
        set_avg_final_metric=False,
        num_worst_factors=1,
        show_power_law=True,
        move_folder=True,
        points=[0, 10, 20, 30],
        baseline_equal_results=None,
        baseline_remix_results=None,
        baseline_worst_results=None,
        with_all_results=True,
        get_baseline_worst_res=True,
        base_demos=None,
    ):
        if "*" in exp_name:
            self.top_exp_folder = self._parse_exp_name(exp_name)
            self.med_curve_folders = glob.glob(f"{RESULTS_DIR}/{exp_name}")
        else:
            assert os.path.isdir(f"{RESULTS_DIR}/{exp_name}"), (
                f"Folder {RESULTS_DIR}/{exp_name} does not exist"
            )
            assert len(os.listdir(f"{RESULTS_DIR}/{exp_name}")) > 0, (
                f"Folder {RESULTS_DIR}/{exp_name} is empty"
            )
            self.top_exp_folder = exp_name
            self.med_curve_folders = [
                os.path.join(f"{RESULTS_DIR}/{exp_name}", f)
                for f in os.listdir(f"{RESULTS_DIR}/{exp_name}")
                if os.path.isdir(os.path.join(f"{RESULTS_DIR}/{exp_name}", f))
                and "baseline" not in f
                and not f[0].isdigit()
                and "bad" not in f
                and "slope" not in f
                and "worst" not in f
            ]

        self.exp_name = exp_name
        self.points = points
        self.env_name = env_name
        self.num_points = num_points
        self.use_sim = use_sim
        self.set_avg_final_sr = set_avg_final_sr
        self.set_avg_final_metric = set_avg_final_metric
        self.num_worst_factors = num_worst_factors
        self.show_power_law = show_power_law
        self.move_folder = move_folder
        self.baseline_equal_results = baseline_equal_results
        self.baseline_worst_results = baseline_worst_results
        self.baseline_remix_results = baseline_remix_results
        self.with_all_results = with_all_results
        self.get_baseline_worst_res = get_baseline_worst_res
        self.base_demos = base_demos

        if self.env_name in ("tomato_plate", "peg_insertion", "pull_cube_tool"):
            self.metrics = [
                "1_normalized_distances",
                "1_distances",
                "1_anomaly_rate",
                "5_normalized_distances",
                "5_distances",
                "5_anomaly_rate",
                "10_normalized_distances",
                "10_distances",
                "10_anomaly_rate",
                # "val_loss",
                "success_rate",
            ]
        else:
            self.metrics = ["success_rate"]

        self.strats = [
            "scaling_curve",
            #    "scaling_curve_origin",
            #    "scaling_curve_taylor",
            #    "initial_point",
            #    "mid_point",
            #    "three_points_first_scaling_curve",
            #    "three_points_last_scaling_curve",
            #    "three_points_first_linear",
            #    "three_points_last_linear",
            #    "three_points_first_scaling_curve_taylor",
            #    "three_points_last_scaling_curve_taylor",
        ]

    def _parse_exp_name(self, exp_name):
        # Match pattern and capture necessary components
        match = re.match(r"(ro|so)_1s_base(\d+)_vary(\d+)([^_]*)_?(.*)?", exp_name)

        if match:
            prefix = "real_only" if match.group(1) == "ro" else "sim_only"
            base = match.group(2)
            vary = match.group(3)
            suffix = match.group(5)  # Everything after `vary*_`

            # If there's no suffix, exclude the underscore
            if suffix:
                return f"{prefix}_base{base}_vary{vary}_{suffix}"
            else:
                return f"{prefix}_base{base}_vary{vary}"

        return None  # Return None if pattern doesn't match

    def _extract_factor_name(self, curve_folder):
        # Adjust the regex to capture the string after "vary" and before next underscore (if exists)
        # match = re.search(r"vary([^_]+)", curve_folder)

        # raw_name = (
        #     match.group(1) if match else ""
        # )  # Return extracted value or empty string if not

        factor_name = ""
        raw_name = curve_folder.split("/")[-1]

        if "dis" in raw_name:
            factor_name += "Distractor "
        if "bg" in raw_name:
            factor_name += "Background "
        if "cp" in raw_name:
            factor_name += "Camera Pose "
        if "lt" in raw_name:
            factor_name += "Lighting "
        if "tt" in raw_name:
            factor_name += "Table Texture "
        if "th" in raw_name:
            factor_name += "Table Height "
        if "op" in raw_name:
            factor_name += "Object Pose "
        if "dq" in raw_name:
            factor_name += "Delta qpos "
        return factor_name.strip()

    def _plot_multiple_power_law(
        self, metric_name, strat, baseline_name=None, baseline_values=None
    ):
        COLOR_HISTORY = "#7FC97F"
        COLOR_FIT = "#BEAED4"
        COLOR_PRED = "#FDC086"
        COLOR_BASELINE = "#4682B4"

        def _load_results_from_folder(folder):
            with open(f"{folder}/grid_data.json", "r") as f:
                result_dict = json.load(f)
            if "0" in result_dict:
                result_dict = result_dict["0"]
            result_dict = {convert_key(k): v for k, v in result_dict.items()}
            values = np.array([np.mean(result_dict[rd]) for rd in result_dict])
            return values

        def _process_metric_values(folders):
            metric_values_list = []
            for folder in folders:
                subfolders = [
                    f
                    for f in os.listdir(folder)
                    if os.path.isdir(os.path.join(folder, f))
                ]
                sorted_subfolders = sorted(subfolders, key=extract_single_number)
                job_ids = [extract_single_number(f) for f in sorted_subfolders]

                means, _, _ = self._calculate_metric(
                    job_ids,
                    metric_points_list[0],
                    metric=metric_name,
                )
                metric_values_list.append(np.array(means))
            return metric_values_list

        folders = self.med_curve_folders
        num_plots = len(folders)
        # Determine subplot grid layout
        if num_plots <= 3:
            num_cols = num_plots
            num_rows = 1
        else:
            num_cols = math.ceil(math.sqrt(num_plots))
            num_rows = math.ceil(num_plots / num_cols)
        figsize = (num_cols * 6, num_rows * 4.5)

        fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
        axs = np.ravel(axs)

        points_list, values_list = [], []
        min_y, max_y = 1.0, 0.0

        for folder in folders:
            values = _load_results_from_folder(folder)
            if max(values) > 1:
                values /= 100
            points = np.array(self.points)
            points_list.append(points)
            values_list.append(values)
            min_y, max_y = min(min_y, np.min(values)), max(max_y, np.max(values))

        avg_final_sr = np.mean([values[self.num_points - 1] for values in values_list])
        if self.set_avg_final_sr:
            for values in values_list:
                values[self.num_points - 1] = avg_final_sr

        if metric_name is not None:
            metric_points_list = list(points_list)
            metric_values_list = _process_metric_values(folders)
            min_y = min(min_y, *(np.min(v) for v in metric_values_list))
            max_y = max(max_y, *(np.max(v) for v in metric_values_list))

            avg_final_metric = np.mean(
                [v[self.num_points - 1] for v in metric_values_list]
            )
            if self.set_avg_final_metric:
                for values in metric_values_list:
                    values[self.num_points - 1] = avg_final_metric
        else:
            metric_points_list, metric_values_list = [], []
            avg_final_metric = None

        dict_to_save = {
            f"success_rate_{strat}": {
                int(p): {} for p in points_list[0][self.num_points :]
            }
        }
        if metric_name:
            dict_to_save[f"{metric_name}_{strat}"] = {
                int(p): {} for p in points_list[0][self.num_points :]
            }

        os.makedirs(f"{RESULTS_DIR}/{self.top_exp_folder}", exist_ok=True)
        baseline_results = {int(p): None for p in points_list[0][self.num_points :]}

        for idx, (folder, points, values) in enumerate(
            zip(folders, points_list, values_list)
        ):
            ax = axs[idx]
            title = self._extract_factor_name(folder)

            # Plot success rate
            predictions, anticipated_baseline = get_predictions_by_strat(
                ax,
                points,
                values,
                strat,
                self.num_points,
                self.show_power_law,
                label="SR",
                y_min=0.95 * min_y,
                y_max=1.05 * max_y,
                title=title,
                color_predicted=COLOR_PRED,
                color_gt=COLOR_HISTORY,
                color_fit=COLOR_FIT,
                num_base_demos=self.base_demos,
            )

            for point in points[self.num_points :]:
                point = int(point)
                dict_to_save[f"success_rate_{strat}"][point].update(
                    get_results_from_predictions(
                        title,
                        points[self.num_points :],
                        predictions,
                        avg_final_sr,
                        anticipated_baseline,
                    )[point]
                )

            # Plot baseline
            if baseline_values is not None:
                for i, point in enumerate(points[self.num_points :]):
                    point = int(point)
                    if max(baseline_values[i]) < 1:
                        baseline_value = baseline_values[i] / 100
                    else:
                        baseline_value = baseline_values[i]
                    baseline_results[point] = (
                        baseline_value,
                        None,
                        baseline_value,
                        baseline_value - avg_final_sr,
                        (baseline_value - avg_final_sr) / avg_final_sr,
                        baseline_value / avg_final_sr,
                    )
                    ax.scatter(
                        point,
                        baseline_value,
                        label=f"Predicted SR for {baseline_name}" if i == 0 else None,
                        zorder=3,
                        color=COLOR_BASELINE,
                    )
                    ax.text(
                        point,
                        baseline_value,
                        f"{baseline_value:.2f}",
                        ha="left",
                        va="bottom",
                        fontsize=10,
                        color=COLOR_BASELINE,
                    )

            # Plot metric
            if metric_name is not None:
                predictions, anticipated_baseline = get_predictions_by_strat(
                    ax,
                    metric_points_list[idx],
                    metric_values_list[idx],
                    strat,
                    self.num_points,
                    self.show_power_law,
                    label=metric_name,
                    y_min=0.95 * min_y,
                    y_max=1.05 * max_y,
                    color_predicted=COLOR_BASELINE,
                    color_gt="blue",
                    color_fit=COLOR_FIT,
                    num_base_demos=self.base_demos,
                )
                for point in points[self.num_points :]:
                    point = int(point)
                    dict_to_save[f"{metric_name}_{strat}"][point].update(
                        get_results_from_predictions(
                            title,
                            points[self.num_points :],
                            predictions,
                            avg_final_metric,
                            anticipated_baseline,
                        )[point]
                    )

            ax.legend(fontsize=15)

        # Cleanup unused subplots
        for i in range(len(folders), len(axs)):
            fig.delaxes(axs[i])

        # Final labeling
        plot_name = f"{self.top_exp_folder}_{metric_name or baseline_name or strat}"
        # fig.suptitle(plot_name, fontsize=14)
        fig.tight_layout()
        plot_path = f"{RESULTS_DIR}/{self.top_exp_folder}/{metric_name or baseline_name or strat}.png"
        plt.savefig(plot_path, dpi=600)
        png_path = f"{RESULTS_DIR}/{self.top_exp_folder}/{metric_name or baseline_name or strat}.png"
        pdf_path = png_path.replace(".png", ".pdf")
        # save PNG as before
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
        print(f"Plot saved to {png_path} and {pdf_path}")
        plt.close()
        plt.close()

        return dict_to_save, baseline_results

    # Validation loss and embedding distance

    def _calculate_metric(self, job_ids, data_keys, metric):
        metric_values = {i: [] for i in data_keys}
        for job_id in job_ids:
            # find the corresponding folder in log/tomato_plate that starts with the job_id
            job_folder = next(
                f
                for f in os.listdir(f"log/{self.env_name}")
                if f.startswith(f"{job_id!s}_")
            )
            data_num = None
            for part in job_folder.split("_")[-2:]:
                if part.startswith("sim" if self.use_sim else "real"):
                    data_num = int(part.replace("sim" if self.use_sim else "real", ""))
                    break  # Stop after finding the real number
            if data_num is None:
                raise ValueError(f"Data number not found in {job_folder}")

            match = re.match(r"(\d+)_(.+)", metric)
            if match:
                k = int(match.group(1))
                metric_name = match.group(2)
                file_path = glob.glob(
                    f"log/{self.env_name}/{job_folder}/embedding_distance_*.npz"
                )[0]
                emb_distance = np.load(file_path, allow_pickle=True)[
                    metric_name
                ].item()[k]
                emb_distance = 1 - emb_distance
                metric_values[data_num - self.base_demos].append(np.mean(emb_distance))
            else:
                assert metric == "val_loss"
                file_path = glob.glob(
                    f"log/{self.env_name}/{job_folder}/val_loss_*.npz"
                )[0]
                val_loss = np.load(file_path)["losses"]
                val_loss = 1 - val_loss
                metric_values[data_num - self.base_demos].append(np.mean(val_loss))

        if not any(metric_values.values()):
            return [], [], []
        # Filter out non-float terms and calculate means and stds
        means = [
            np.mean(
                [
                    x
                    for x in metric_values[rd]
                    if isinstance(x, (np.float32, np.float64))
                ]
            ).item()
            for rd in metric_values
        ]
        stds = [
            np.std(
                [
                    x
                    for x in metric_values[rd]
                    if isinstance(x, (np.float32, np.float64))
                ]
            )
            for rd in metric_values
        ]
        return means, stds, list(metric_values.keys())

    def _postprocess_results(
        self, result_dict: dict[str, dict[str, dict[int, dict[str, tuple]]]]
    ):
        baseline_equal_results = self._get_baseline_equal_results()
        baseline_remix_results = self._get_baseline_equal_results()

        have_gt_points = len(self.points) > self.num_points

        if have_gt_points:
            true_factor_success_rate = {}

            for metric, strats in result_dict.items():
                for strat, pts in strats.items():
                    for point, factors in pts.items():
                        for factor, rl in factors.items():
                            (
                                anticipated_value,
                                delta_anticipated,
                                true_value,
                                delta,
                                delta_ratio,
                                real_ratio,
                            ) = rl
                            # print(
                            #     f"Factor: {factor}, Point: {point}, Anticipated Value: {anticipated_value:.2f}, True Value: {true_value:.2f}, Delta: {delta:.2f}, Delta Ratio: {delta_ratio:.2f}, Real Ratio: {real_ratio:.2f}"
                            # )
                            if metric == "success_rate":
                                if point not in true_factor_success_rate:
                                    true_factor_success_rate[point] = {}
                                true_factor_success_rate[point][factor] = true_value

            points = list(true_factor_success_rate.keys())
            true_rank = {point: [] for point in points}
            for point in points:
                true_rank[point] = sorted(
                    true_factor_success_rate[point].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                print(f"Point: {point}, Rank: {true_rank[point]}")

        else:
            points = self.points
            true_rank = {point: [] for point in points}

        final_results = {point: {} for point in points}
        slopes = {point: {} for point in points}
        for metric, strats in result_dict.items():
            for strat, pts in strats.items():
                for point, factors in pts.items():
                    slopes[point][f"{metric}_{strat}"] = {}
                    max_anticipated_value = {None: (-1, None, None, None, None, None)}
                    # min_anticipated_value = {None: (100, None, None)}
                    max_drop = {None: (None, 100, None, None, None, None)}
                    for factor, rl in factors.items():
                        (
                            anticipated_value,
                            delta_anticipated,
                            true_value,
                            delta,
                            delta_ratio,
                            real_ratio,
                        ) = rl
                        slopes[point][f"{metric}_{strat}"][factor] = (
                            delta_anticipated if delta_anticipated > 0 else 0
                        )
                        if metric == "success_rate" and strat == "scaling_curve":
                            print(delta_anticipated)
                        if metric == "success_rate":
                            if (
                                anticipated_value
                                > list(max_anticipated_value.values())[0][0]
                            ):
                                max_anticipated_value = {
                                    factor: (
                                        anticipated_value,
                                        delta_anticipated,
                                        true_value,
                                        delta,
                                        delta_ratio,
                                        real_ratio,
                                    )
                                }
                        else:
                            # if anticipated_value < list(min_anticipated_value.values())[0][0]:
                            #     min_anticipated_value = {factor: (anticipated_value, real_value, delta, delta_ratio, real_ratio)}
                            if (
                                delta_anticipated
                                < max_drop[list(max_drop.keys())[0]][1]
                            ):
                                max_drop = {
                                    factor: (
                                        anticipated_value,
                                        delta_anticipated,
                                        true_value,
                                        delta,
                                        delta_ratio,
                                        real_ratio,
                                    )
                                }
                    if metric == "success_rate":
                        factor_picked = list(max_anticipated_value.keys())[0]
                    else:
                        # factor_picked = list(min_anticipated_value.keys())[0]
                        factor_picked = list(max_drop.keys())[0]
                    if have_gt_points:
                        rank_picked = [f for f, _ in true_rank[point]].index(
                            factor_picked
                        )
                    else:
                        rank_picked = None
                    if metric == "success_rate":
                        anticipated_value = max_anticipated_value[factor_picked][0]
                        delta_anticipated = max_anticipated_value[factor_picked][1]
                        real_value = max_anticipated_value[factor_picked][2]
                        delta = max_anticipated_value[factor_picked][3]
                        delta_ratio = max_anticipated_value[factor_picked][4]
                        real_ratio = max_anticipated_value[factor_picked][5]
                    else:
                        anticipated_value = max_drop[factor_picked][0]
                        delta_anticipated = max_drop[factor_picked][1]
                        real_value = max_drop[factor_picked][2]
                        delta = max_drop[factor_picked][3]
                        delta_ratio = max_drop[factor_picked][4]
                        real_ratio = max_drop[factor_picked][5]
                    final_results[point][f"{metric}_{strat}"] = (
                        factor_picked,
                        rank_picked,
                        real_value,
                        delta,
                        delta_ratio,
                        real_ratio,
                    )
                    print(
                        f"Point: {point}, Metric: {metric}, Strat: {strat}, Factor Picked: {factor_picked}, Rank Picked: {rank_picked}, Real value: {real_value:.3f}, Delta: {delta:.3f}, Delta Ratio: {delta_ratio:.3f}, Real Ratio: {real_ratio:.3f}"
                    )

                    # For each point, metric, strat, get the weight of each factor based on it's delta improvement, and save as a dict slopes[point][f"{metric}_{strat}"] = {factor1: weight1, ...}

                    delta_sum = np.sum(
                        [r for r in slopes[point][f"{metric}_{strat}"].values()]
                    ).item()
                    for factor, delta_anticipated in slopes[point][
                        f"{metric}_{strat}"
                    ].items():
                        if delta_sum != 0:
                            slopes[point][f"{metric}_{strat}"][factor] = (
                                delta_anticipated / delta_sum
                            )
                        else:
                            slopes[point][f"{metric}_{strat}"][factor] = 1 / len(
                                slopes[point][f"{metric}_{strat}"]
                            )

        if self.baseline_worst_results is not None:
            factor_picked_worst = self.baseline_worst_results.pop("worst_factor")
            worst_weights = self.baseline_worst_results.pop("worst_weights")
        else:
            factor_picked_worst, worst_weights = self._get_baseline_worst_results()

        worst_weights_parsed = {}
        for point in points:
            worst_weights_parsed[point] = {"success_rate_scaling_curve": worst_weights}

        if self.with_all_results:
            metric = "success_rate"
            strat = "scaling_curve"
            # final_results[point]["baseline_worst"] = (factor_picked_worst)
            for point, factor_results in result_dict[metric][strat].items():
                for factor, rl in factor_results.items():
                    this_factor = True
                    for f in factor_picked_worst:
                        if f not in factor:
                            this_factor = False
                    if this_factor:
                        (
                            anticipated_value,
                            delta_anticipated,
                            real_value,
                            delta,
                            delta_ratio,
                            real_ratio,
                        ) = rl
                        if len(self.points) > self.num_points:
                            rank_picked = [f for f, _ in true_rank[point]].index(factor)
                        else:
                            rank_picked = None
                        final_results[point]["baseline_worst"] = (
                            factor_picked_worst,
                            rank_picked,
                            real_value,
                            delta,
                            delta_ratio,
                            real_ratio,
                        )
                        print(
                            f"Point: {point}, Metric: success_rate, Strat: baseline_worst, Factor Picked: {factor}, Rank Picked: {rank_picked}, Real value: {real_value:.3f}, Delta: {delta:.3f}, Delta Ratio: {delta_ratio:.3f}, Real Ratio: {real_ratio:.3f}"
                        )
                        break

            # For baseline equal
            if baseline_equal_results is not None:
                for point, result in baseline_equal_results.items():
                    point = str(int(point))
                    real_value, delta, delta_ratio, real_ratio = result
                    print(
                        f"Point: {point}, Metric: success_rate, Strat: baseline_equal, Factor Picked: N/A, Rank Picked: N/A, Real value: {real_value:.3f}, Delta: {delta:.3f}, Delta Ratio: {delta_ratio:.3f}, Real Ratio: {real_ratio:.3f}"
                    )
                    final_results[point]["baseline_equal"] = (
                        None,
                        None,
                        real_value,
                        delta,
                        delta_ratio,
                        real_ratio,
                    )

            for point, results in final_results.items():
                baseline_equal_result = (
                    results.get("baseline_equal")[2]
                    if "baseline_equal" in results
                    else None
                )
                baseline_worst_result = (
                    results.get("baseline_worst")[2]
                    if "baseline_worst" in results
                    else None
                )
                scaling_curve_result = results.get("success_rate_scaling_curve")[2]

                # Rank above three
                result = {
                    "baseline_equal": baseline_equal_result,
                    "baseline_worst": baseline_worst_result,
                    "scaling_curve": scaling_curve_result,
                }
                result = {k: v for k, v in result.items() if v is not None}
                sorted_result = sorted(result.items(), key=lambda x: x[1], reverse=True)
                print(f"Point: {point}, Rank: {sorted_result}")

        return final_results, slopes, worst_weights_parsed

    def _get_baseline_worst_results(self):
        folders = self.med_curve_folders

        success_rate_per_factor = {}
        for _, exp_folder in enumerate(folders):
            if (
                "baseline" in exp_folder
                or "slope" in exp_folder
                or "worst" in exp_folder
            ):
                continue
            sorted_job_folders = [
                f for f in os.listdir(exp_folder) if re.search(r"\d+", f)
            ]
            job_folder = sorted(sorted_job_folders, key=extract_single_number)[
                self.num_points - 1
            ]
            if success_rate_per_factor == {}:
                success_rate_per_factor = get_factor_sr(
                    os.path.join(exp_folder, job_folder)
                )
            else:
                for k, v in get_factor_sr(os.path.join(exp_folder, job_folder)).items():
                    success_rate_per_factor[k].extend(v)
        for k, v in success_rate_per_factor.items():
            success_rate_per_factor[k] = np.mean(v)

        def parse_factor_name(factor):
            if "background" in factor:
                return "Background"
            elif "camera_pose" in factor:
                return "Camera Pose"
            elif "distractor" in factor:
                return "Distractor"
            elif "directional" in factor:
                return "Lighting"
            elif "table_texture" in factor:
                return "Table Texture"
            elif "table_height" in factor:
                return "Table Height"
            elif "obj_pose" in factor:
                return "Object Pose"
            elif "delta_qpos" in factor:
                return "Delta qpos"
            else:
                raise ValueError(f"Unknown factor name: {factor}")

        # Sort the factors by success rate
        sorted_success_rate = sorted(
            success_rate_per_factor.items(), key=lambda x: x[1]
        )
        print(sorted_success_rate)
        factor_names = []
        for i in range(self.num_worst_factors):
            factor_name = parse_factor_name(sorted_success_rate[i][0])
            factor_names.append(factor_name)
        # print(factor_names)
        # print(success_rate_per_factor)
        # worst1_weights = {sorted_success_rate[i][0]: 1 if i==0 else 1 for i in range(len(sorted_success_rate))}

        total_weights = 0
        for l in sorted_success_rate:
            factor_name, success_rate = l
            total_weights += 1 - float(success_rate)

        worst_weighted_weights = {}
        for i in range(len(sorted_success_rate)):
            worst_weighted_weights[sorted_success_rate[i][0]] = (
                1 - float(sorted_success_rate[i][1])
            ) / total_weights
        return factor_names, worst_weighted_weights

    def _get_baseline_equal_results(self):
        if self.baseline_equal_results is not None:
            return self._plot_multiple_power_law(
                metric_name=None,
                baseline_name="Equal",
                baseline_values=self.baseline_equal_results,
                strat="scaling_curve",
            )[1]
        else:
            return None
        # if self.top_exp_folder == "sim_only_base80_vary40_factor_fixed_sample_3_instances":
        #     baseline_values = [54.266, 60.533, 59.6]
        # elif self.top_exp_folder == "sim_only_base315_vary75_peg":
        #     baseline_values = [22.29, 25.1]
        # elif self.top_exp_folder == "sim_only_base90_vary75_peg":
        #     baseline_values = [37.5, 25.11]
        # else:
        #     baseline_values = None
        # if baseline_values is not None:
        #     return self._plot_multiple_power_law(metric_name=None, baseline_name="Equal", baseline_values=self.baseline_equal_results, strats=["scaling_curve"])[1]
        # else:
        #     return None

    def _get_baseline_remix_results(self):
        if self.baseline_remix_results is not None:
            return self._plot_multiple_power_law(
                metric_name=None,
                baseline_name="Re-Mix",
                baseline_values=self.baseline_remix_results,
                strat="scaling_curve",
            )[1]
        else:
            return None

    def _get_metric_results(self):
        metric_results = {}
        for metric in self.metrics:
            metric_result, _ = self._plot_multiple_power_law(
                metric_name=metric, strat=self.strats[0]
            )
            metric_results.update(metric_result)

        with open(f"{RESULTS_DIR}/{self.top_exp_folder}/predictions.json", "w") as f:
            json.dump(metric_results, f, indent=4)

    def get_final_results(self, override=False):
        if (
            os.path.exists(f"{RESULTS_DIR}/{self.top_exp_folder}/final_results.json")
            and not override
        ):
            with open(
                f"{RESULTS_DIR}/{self.top_exp_folder}/final_results.json", "r"
            ) as f:
                final_results = json.load(f)

        # else:
        if (
            not os.path.exists(f"{RESULTS_DIR}/{self.top_exp_folder}/predictions.json")
            or override
        ):
            self._get_metric_results()

        with open(f"{RESULTS_DIR}/{self.top_exp_folder}/predictions.json", "r") as f:
            result_dict = json.load(f)

        final_results, slope_weights, worst_weights = self._postprocess_results(
            result_dict
        )
        with open(f"{RESULTS_DIR}/{self.top_exp_folder}/final_results.json", "w") as f:
            json.dump(final_results, f, indent=4)

        if self.move_folder:
            folders = self.med_curve_folders
            for folder in folders:
                os.rename(
                    folder,
                    f"{RESULTS_DIR}/{self.top_exp_folder}/{folder.split('/')[-1]}",
                )

        return final_results, slope_weights, worst_weights

    def get_train_results(self):
        # 1. Get predictions for each metric_strat, and slope, and rank of each factor.
        pred_dict = {}
        points = self.points[self.num_points :]
        slope_dict = {point: {} for point in points}
        rank_dict = {point: {} for point in points}
        for metric in self.metrics:
            for strat in self.strats:
                res, _ = self._plot_multiple_power_law(
                    metric_name=metric if metric != "success_rate" else None,
                    strat=strat,
                )
                pred_dict.update(res)
                # Get the slopes for each metric_strat.
                for point, factor_res in res[f"{metric}_{strat}"].items():
                    slope_dict[point][f"{metric}_{strat}"] = {}
                    rank_dict[point][f"{metric}_{strat}"] = {}
                    for factor, (
                        anticipated_value,
                        delta_anticipated,
                        true_value,
                        delta,
                        delta_ratio,
                        real_ratio,
                    ) in factor_res.items():
                        slope_dict[point][f"{metric}_{strat}"][factor] = (
                            delta_anticipated if delta_anticipated > 0 else 0
                        )
                        rank_dict[point][f"{metric}_{strat}"][factor] = (
                            delta_anticipated
                        )
                        if metric == "success_rate" and strat == "scaling_curve":
                            print(delta_anticipated)
                    # For each point, metric, strat, get the weight of each factor based on it's delta improvement, and save as a dict slopes[point][f"{metric}_{strat}"] = {factor1: weight1, ...}
                    delta_sum = np.sum(
                        [r for r in slope_dict[point][f"{metric}_{strat}"].values()]
                    ).item()
                    for factor, delta_anticipated in slope_dict[point][
                        f"{metric}_{strat}"
                    ].items():
                        if delta_sum != 0:
                            slope_dict[point][f"{metric}_{strat}"][factor] = (
                                delta_anticipated / delta_sum
                            )
                        else:
                            slope_dict[point][f"{metric}_{strat}"][factor] = 1 / len(
                                slope_dict[point][f"{metric}_{strat}"]
                            )
                    # substitute slope_dict[point][f"{metric}_{strat}"][factor] with rank of the factor
                    values_dict = rank_dict[point][f"{metric}_{strat}"]
                    sorted_items = sorted(
                        values_dict.items(), key=lambda item: item[1], reverse=True
                    )
                    ranks = {
                        key: rank + 1 for rank, (key, _) in enumerate(sorted_items)
                    }
                    rank_dict[point][f"{metric}_{strat}"] = {
                        key: ranks[key] for key in values_dict
                    }

        with open(f"{RESULTS_DIR}/{self.top_exp_folder}/predictions.json", "w") as f:
            json.dump(pred_dict, f, indent=4)
        with open(f"{RESULTS_DIR}/{self.top_exp_folder}/slope_weights.json", "w") as f:
            json.dump(slope_dict, f, indent=4)
        with open(f"{RESULTS_DIR}/{self.top_exp_folder}/ranks.json", "w") as f:
            json.dump(rank_dict, f, indent=4)

        # 2. Get per factor success rate for each point in each curve.
        folders = self.med_curve_folders

        success_rate_per_point_per_curve = {}
        for _, exp_folder in enumerate(folders):
            if (
                "baseline" in exp_folder
                or "slope" in exp_folder
                or "worst" in exp_folder
            ):
                continue
            curve_name = self._extract_factor_name(exp_folder)
            success_rate_per_point_per_curve[curve_name] = {}

            job_folders = [f for f in os.listdir(exp_folder) if re.search(r"\d+", f)]
            sorted_job_folders = sorted(job_folders, key=extract_single_number)

            for job_folder, point in zip(sorted_job_folders, self.points):
                success_rate_per_point_per_curve[curve_name][point] = get_factor_sr(
                    os.path.join(exp_folder, job_folder)
                )

        with open(
            f"{RESULTS_DIR}/{self.top_exp_folder}/success_rate_per_point.json", "w"
        ) as f:
            json.dump(success_rate_per_point_per_curve, f, indent=4)

        has_gt_points = True

        # 3. Get the correlation between metric_strat and gt success_rate.
        true_factor_success_rate = {}
        true_factor_success_rate_rank = {}
        corr_with_gt_dict = {point: {} for point in points}
        for point, factor_res in pred_dict["success_rate_scaling_curve"].items():
            for factor, (_, _, true_value, _, _, _) in factor_res.items():
                if true_value is None:
                    has_gt_points = False
                    continue
                if point not in true_factor_success_rate:
                    true_factor_success_rate[point] = {}
                if point not in true_factor_success_rate_rank:
                    true_factor_success_rate_rank[point] = {}
                true_factor_success_rate[point][factor] = true_value
                true_factor_success_rate_rank[point][factor] = true_value

            if point not in true_factor_success_rate:
                continue
            values_dict = true_factor_success_rate[point]
            sorted_items = sorted(
                values_dict.items(), key=lambda item: item[1], reverse=True
            )
            ranks = {key: rank + 1 for rank, (key, _) in enumerate(sorted_items)}
            true_factor_success_rate_rank[point] = {
                key: ranks[key] for key in values_dict
            }
            true_factor_rank = list(true_factor_success_rate_rank[point].values())

            for corr_metric in ["pearson", "mmrv"]:
                corr_with_gt_dict[point][corr_metric] = {}
                for ms, ms_factor_rank_dict in rank_dict[point].items():
                    ms_factor_rank = list(ms_factor_rank_dict.values())
                    corr = calculate_corr(true_factor_rank, ms_factor_rank, corr_metric)
                    corr_with_gt_dict[point][corr_metric][ms] = corr

        with open(f"{RESULTS_DIR}/{self.top_exp_folder}/corr_with_gt.json", "w") as f:
            json.dump(corr_with_gt_dict, f, indent=4)

        with open(
            f"{RESULTS_DIR}/{self.top_exp_folder}/true_factor_success_rate.json", "w"
        ) as f:
            json.dump(true_factor_success_rate, f, indent=4)
        with open(
            f"{RESULTS_DIR}/{self.top_exp_folder}/true_factor_success_rate_rank.json",
            "w",
        ) as f:
            json.dump(true_factor_success_rate_rank, f, indent=4)

        # 4. Get the correlation between metric_strat and success rate strategies

        success_rate_ms = [k for k in list(pred_dict.keys()) if "success_rate" in k]
        corr_with_success_rate_dict = {point: {} for point in points}
        for point in points:
            for ms in success_rate_ms:
                corr_with_success_rate_dict[point][ms] = {}
                success_rate_rank = list(rank_dict[point][ms].values())
                for corr_metric in ["pearson", "mmrv"]:
                    corr_with_success_rate_dict[point][ms][corr_metric] = {}
                    for other_ms, other_ms_factor_rank_dict in rank_dict[point].items():
                        if ms == other_ms:
                            continue
                        other_ms_factor_rank = list(other_ms_factor_rank_dict.values())
                        corr = calculate_corr(
                            success_rate_rank, other_ms_factor_rank, corr_metric
                        )
                        corr_with_success_rate_dict[point][ms][corr_metric][
                            other_ms
                        ] = corr
        with open(
            f"{RESULTS_DIR}/{self.top_exp_folder}/corr_with_success_rate.json", "w"
        ) as f:
            json.dump(corr_with_success_rate_dict, f, indent=4)

        # 5. Get weight for the baseline worst
        if self.get_baseline_worst_res:
            _, worst_weights = self._get_baseline_worst_results()
            with open(
                f"{RESULTS_DIR}/{self.top_exp_folder}/worst_weights.json", "w"
            ) as f:
                json.dump(worst_weights, f, indent=4)
            worst_weights_parsed = {}
            for point in points:
                worst_weights_parsed[point] = {
                    "success_rate_scaling_curve": worst_weights
                }
        else:
            worst_weights_parsed = None

        # 6. Get the factor picked for each point, and the rank of the factor.
        if has_gt_points:
            factor_picked_results = {point: {} for point in points}
            for metric_strat, point_factor in pred_dict.items():
                for point, factor_res in point_factor.items():
                    max_anticipated_value = {None: (-1, None, None, None, None, None)}
                    max_drop = {None: (None, 100, None, None, None, None)}
                    for factor, (
                        anticipated_value,
                        delta_anticipated,
                        true_value,
                        delta,
                        delta_ratio,
                        real_ratio,
                    ) in factor_res.items():
                        if "success_rate" in metric_strat:
                            if (
                                anticipated_value
                                > list(max_anticipated_value.values())[0][0]
                            ):
                                max_anticipated_value = {
                                    factor: (
                                        anticipated_value,
                                        delta_anticipated,
                                        true_value,
                                        delta,
                                        delta_ratio,
                                        real_ratio,
                                    )
                                }
                        else:
                            if (
                                delta_anticipated
                                < max_drop[list(max_drop.keys())[0]][1]
                            ):
                                max_drop = {
                                    factor: (
                                        anticipated_value,
                                        delta_anticipated,
                                        true_value,
                                        delta,
                                        delta_ratio,
                                        real_ratio,
                                    )
                                }
                    if "success_rate" in metric_strat:
                        factor_picked = list(max_anticipated_value.keys())[0]
                    else:
                        factor_picked = list(max_drop.keys())[0]
                    rank_picked = true_factor_success_rate_rank[point][factor_picked]
                    if "success_rate" in metric_strat:
                        anticipated_value = max_anticipated_value[factor_picked][0]
                        delta_anticipated = max_anticipated_value[factor_picked][1]
                        real_value = max_anticipated_value[factor_picked][2]
                        delta = max_anticipated_value[factor_picked][3]
                        delta_ratio = max_anticipated_value[factor_picked][4]
                        real_ratio = max_anticipated_value[factor_picked][5]
                    else:
                        anticipated_value = max_drop[factor_picked][0]
                        delta_anticipated = max_drop[factor_picked][1]
                        real_value = max_drop[factor_picked][2]
                        delta = max_drop[factor_picked][3]
                        delta_ratio = max_drop[factor_picked][4]
                        real_ratio = max_drop[factor_picked][5]
                    factor_picked_results[point][metric_strat] = (
                        factor_picked,
                        rank_picked,
                        real_value,
                        delta,
                        delta_ratio,
                        real_ratio,
                    )
                    print(
                        f"Point: {point}, Metric and strat: {metric_strat}, Factor Picked: {factor_picked}, Rank Picked: {rank_picked}, Real value: {real_value:.3f}, Delta: {delta:.3f}, Delta Ratio: {delta_ratio:.3f}, Real Ratio: {real_ratio:.3f}"
                    )
            with open(
                f"{RESULTS_DIR}/{self.top_exp_folder}/factor_picked_results.json", "w"
            ) as f:
                json.dump(factor_picked_results, f, indent=4)

        return slope_dict, worst_weights_parsed

    def get_test_results(self):
        folders = [
            f
            for f in os.listdir(f"{RESULTS_DIR}/{self.exp_name}")
            if os.path.isdir(os.path.join(f"{RESULTS_DIR}/{self.exp_name}", f))
            and ("baseline" in f or "slope" in f or "worst" in f or "distance" in f)
            # and not f[0].isdigit()
            and "bad" not in f
        ]

        points = self.points[self.num_points :]
        results_dict = {point: {} for point in points}

        for point in points:
            for _, folder in enumerate(folders):
                with open(
                    os.path.join(
                        f"{RESULTS_DIR}/{self.exp_name}", folder, "grid_data.json"
                    ),
                    "r",
                ) as f:
                    res_dict = json.load(f)

                if "0" in res_dict:
                    res_dict = res_dict["0"]
                res_dict = {
                    convert_key(k) - self.base_demos: np.mean(v)
                    for k, v in res_dict.items()
                }
                if point not in res_dict:
                    continue
                results_dict[point][folder] = res_dict[point]

            # Sort the results for each point
            sorted_results = sorted(
                results_dict[point].items(), key=lambda x: x[1], reverse=True
            )
            results_dict[point] = {k: v for k, v in sorted_results}

        with open(f"{RESULTS_DIR}/{self.exp_name}/test_results.json", "w") as f:
            json.dump(results_dict, f, indent=4)


class PlotCurvePi0(PlotCurve):
    def __init__(
        self,
        exp_name,
        env_name,
        num_points=4,
        use_sim=True,
        set_avg_final_sr=False,
        set_avg_final_metric=False,
        num_worst_factors=1,
        show_power_law=True,
        move_folder=True,
    ):
        super().__init__(
            exp_name,
            env_name,
            num_points,
            use_sim,
            set_avg_final_sr,
            set_avg_final_metric,
            num_worst_factors,
            show_power_law,
            move_folder,
        )
        self.metrics = [None]

    def _parse_exp_name(self, exp_name):
        return exp_name.replace("*", "pi0")

    def _extract_factor_name(self, curve_folder):
        factor_name = ""
        raw_name = curve_folder

        if "dis" in raw_name:
            factor_name += "Distractor "
        if "bg" in raw_name:
            factor_name += "Background "
        if "cp" in raw_name:
            factor_name += "Camera Pose "
        if "lt" in raw_name:
            factor_name += "Lighting "
        if "tt" in raw_name:
            factor_name += "Table Texture "
        if "th" in raw_name:
            factor_name += "Table Height "
        if "op" in raw_name:
            factor_name += "Object Pose "
        if "dq" in raw_name:
            factor_name += "Delta qpos "
        return factor_name.strip()


def calculate_corr(rank1, rank2, corr_metric="pearson"):
    if corr_metric == "pearson":
        return pearsonr(rank1, rank2)[0]
    elif corr_metric == "mmrv":
        N = len(rank1)
        mmrv = 0
        for i in range(N):
            rank_violations = []
            for j in range(N):
                violation = int((rank2[i] < rank2[j]) != (rank1[i] < rank1[j]))
                rank_violations.append(np.abs(rank1[i] - rank1[j]) * violation)
            max_rank_violation = max(rank_violations)
            mmrv += max_rank_violation
        return mmrv / N


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--show-power-law", action="store_true")
    parser.add_argument("--move-folder", action="store_true")
    parser.add_argument("--num-points", type=int, default=4)
    parser.add_argument("--env-name", type=str)
    parser.add_argument("--use-sim", action="store_true")
    parser.add_argument("--num-worst-factor", type=int, default=1)
    parser.add_argument("--points", nargs="+", type=int)

    args = parser.parse_args()

    # num_worst_factor_dict = {
    #     "sim_only_base315_vary75_peg": 1,
    #     "sim_only_base90_vary75_peg": 1,
    #     "sim_only_base90_vary60_factor": 2,
    #     "sim_only_base80_vary40_factor_fixed_sample_3_instances": 1,
    #     "sim_only_base60_vary40_uniform": 2,
    # }

    with open(
        os.path.join(RESULTS_DIR, args.exp_name, "baseline_equal", "grid_data.json"),
        "r",
    ) as f:
        baseline_equal_results = json.load(f)
        if "0" in baseline_equal_results:
            baseline_equal_results = baseline_equal_results["0"]
        # Convert key
        baseline_equal_results = list(baseline_equal_results.values())
        print(baseline_equal_results)
        if isinstance(baseline_equal_results[0], list):
            baseline_equal_results = [int(v[0]) for v in baseline_equal_results]
        else:
            baseline_equal_results = [int(v) for v in baseline_equal_results]

    if os.path.exists(
        os.path.join(RESULTS_DIR, args.exp_name, "baseline_worst", "grid_data.json")
    ):
        with open(
            os.path.join(
                RESULTS_DIR, args.exp_name, "baseline_worst", "grid_data.json"
            ),
            "r",
        ) as f:
            baseline_worst_results = json.load(f)
            # baseline_worst_results = list(baseline_worst_results.values())
            print(baseline_worst_results)
            # baseline_worst_results = [int(v[0]) for v in baseline_worst_results]
    else:
        baseline_worst_results = None

    curve_plotter = PlotCurve(
        exp_name=args.exp_name,
        env_name=args.env_name,
        num_points=args.num_points,
        use_sim=args.use_sim,
        show_power_law=args.show_power_law,
        move_folder=args.move_folder,
        set_avg_final_sr=True,
        set_avg_final_metric=False,
        num_worst_factors=args.num_worst_factor,
        points=args.points,
        baseline_equal_results=baseline_equal_results,
        baseline_worst_results=baseline_worst_results,
    )
    final_results = curve_plotter.get_train_results()

    # curve_plotters = []
    # for exp_name in  ["sim_only_base315_vary75_peg", "sim_only_base90_vary75_peg",
    #                     "sim_only_base90_vary60_factor", "sim_only_base80_vary40_factor_fixed_sample_3_instances",
    #                     "sim_only_base60_vary40_uniform"]:
    #     curve_plotter = PlotCurve(exp_name=exp_name, env_name=args.env_name, num_points=args.num_points,
    #                           use_sim=args.use_sim, show_power_law=args.show_power_law, move_folder=args.move_folder, set_avg_final_sr=True, set_avg_final_metric=True,
    #                           num_worst_factors=args.num_worst_factor)
    #     curve_plotters.append(curve_plotter)

    # correlation = get_correlation_between_metrics(curve_plotters, corr_metric="rank_violation")
