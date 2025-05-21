#!/bin/bash

# All previous arguments are job IDs
job_ids=("${@:1:$(($# - 3))}")

# Get the last five fixed arguments
env_name="${@: -1}"
curve_name="${@: -2:1}"
exp_name="${@: -3:1}"

# Create a folder for the experiment results
mkdir -p "${GDC_RESULTS_DIR}/${exp_name}/${curve_name}"

# Automatically calculate the number of folders
num_folders=${#job_ids[@]}

# Declare indexed arrays to store success rates and checkpoint epochs
declare -a success_rates
declare -a ckpt_epochs

group1_names=()
group2_names=()

# Initialize arrays dynamically based on num_folders
for (( i=0; i<num_folders; i++ )); do
    success_rates[i]=""
    ckpt_epochs[i]=""
    # echo "${i}"
done

folder_cnt=0
# Iterate over the job IDs and assign results to the correct folder
for job_id in "${job_ids[@]}"; do
    folder_name=$(ls log/$env_name | grep -E "^${job_id}_" | head -n 1)

    # Extract sim and real numbers from folder name
    if [[ $folder_name =~ sim([0-9]+)_real([0-9]+) ]]; then
        sim_number=${BASH_REMATCH[1]}
        real_number=${BASH_REMATCH[2]}
        
        if [[ ! " ${group1_names[@]} " =~ " $real_number " ]]; then
            group1_names+=($real_number)
        fi
        if [[ ! " ${group2_names[@]} " =~ " $sim_number " ]]; then
            group2_names+=($sim_number)
        fi
    fi

    # Capture both success rate and checkpoint epoch
    read success_rate ckpt_epoch < <(./get_best_ckpt.sh "$GDC_RESULTS_DIR/${exp_name}/${curve_name}/${job_id}")
    # Determine the folder index
    folder_index=$((folder_cnt % num_folders))
    folder_cnt=$((folder_cnt + 1))
    
    # Append results to the corresponding arrays
    success_rates[$folder_index]+="$success_rate "
    ckpt_epochs[$folder_index]+="$ckpt_epoch "
    
    # Move checked folders
    # mv "$GDC_RESULTS_DIR/${exp_name}/${curve_name}/${point}" "$GDC_RESULTS_DIR/${exp_name}/"
done

# Sort group names
IFS=$'\n' group1_names=($(sort -n <<<"${group1_names[*]}"))
IFS=$'\n' group2_names=($(sort -n <<<"${group2_names[*]}"))
unset IFS

# Capture grid data output
grid_data=""
for (( i=0; i<num_folders; i++ )); do
    output="${i} = [${success_rates[i]}]"
    grid_data+="$output\n"
done

# -e handles \n
echo -e "$grid_data"

# Save grid data to a file
echo -e "$grid_data" > "$GDC_RESULTS_DIR/${exp_name}/${curve_name}/grid_data.txt"

echo ""

# Run the Python script with grid_data
python3 - <<EOF
import re
import matplotlib.pyplot as plt
import numpy as np
import json

def parse_grid_data(grid_data, group1, group2):
    data_dict = {}
    grid_lines = grid_data.strip().split("\n")
    
    grid_values = []
    for line in grid_lines:
        values = re.findall(r"[-+]?\d*\.\d+|\d+", line.split("=", 1)[-1])
        print(values)
        grid_values.append(list(map(float, values)))
    index = 0
    for g1 in group1:
        data_dict[g1] = {}
        for g2 in group2:
            data_dict[g1][g2] = grid_values[index]
            index += 1
    
    return data_dict

# Read grid data from bash variable
grid_data = """$grid_data"""

group1_names = [$(echo ${group1_names[@]} | sed 's/ /, /g')]
group2_names = [$(echo ${group2_names[@]} | sed 's/ /, /g')]

print(group1_names)
print(group2_names)

# Parse and create the dictionary
result_dict = parse_grid_data(grid_data, group1_names, group2_names)
print(result_dict)

# Save the JSON file
json_path = f"${GDC_RESULTS_DIR}/${exp_name}/${curve_name}/grid_data.json"
with open(json_path, "w") as f:
    json.dump(result_dict, f, indent=2)

print(f"JSON saved to {json_path}")

plt.figure(figsize=(8, 6))
colors = {g: c for g, c in zip(group1_names, ['b', 'g', 'r', 'c', 'm', 'y', 'k'])}

for sim_demos, real_data_dict in result_dict.items():
    real_data_points = sorted(real_data_dict.keys())
    means = [np.mean(real_data_dict[rd]) for rd in real_data_points]
    stds = [np.std(real_data_dict[rd]) for rd in real_data_points]

    plt.errorbar(
        real_data_points,
        means,
        yerr=stds,
        fmt="-o",
        label=f"{sim_demos} Sim Demos",
        color=colors.get(sim_demos, 'black'),
        capsize=4,
    )

plt.xlabel("Number of Real Data")
plt.ylabel("Average Success Rate")
plt.title("${exp_name}_${curve_name}")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.savefig("${GDC_RESULTS_DIR}/${exp_name}/${curve_name}/plot.png")
EOF
