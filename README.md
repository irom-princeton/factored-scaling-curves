# Guiding Data Collection via Factored Scaling Curves (FSC)
## [<a href="https://factored-data-scaling.github.io/" target="_blank">Project Page</a>][<a href="https://arxiv.org/abs/2505.07728">Paper</a>][<a href="https://youtu.be/unTqsgAVE1o">Video</a>]

![FSC banner](https://raw.githubusercontent.com/lihzha/visualizations/main/fsc_banner.gif)

We introduce Factored Scaling Curves (FSC), which model how policy performance scales with data for different environmental factors and can be extrapolated to guide principled data collection, as described <a href="https://factored-data-scaling.github.io/" target="_blank">here</a>.

## Installation

1. **Clone the repository (with sub-modules)**
    ```console
    git clone --recurse-submodules git@github.com:irom-lab/factored-scaling-curves.git
    cd factored-scaling-curves
    ```

2. **Create and activate a Conda environment**
    ```console
    conda create -n fsc python=3.9 -y
    conda activate fsc
    ```

3. **Install the core packages (Linux + NVIDIA GPU)**
    ```console
    pip install -e .
    cd guided_dc/maniskill
    pip install -e .
    cd ../..
    ```

4. **Set up environment variables**
    ```console
    ./scripts/set_path.sh
    ```

5. **(Optional) Extra back-ends**

    - **OpenPI (for experiments with $\pi_0$)**: follow the instructions in
    https://github.com/Physical-Intelligence/openpi

    - **Re-Mix**: see
    https://github.com/jhejna/remix

## Data Layout
We store the raw data in `data/raw_data` with the hierarchy:

```
data/raw_data/
└── <exp_name>/
    ├── <factor_name_1>/
    │   ├── 0.h5
    │   ├── 1.h5
    │   └── …
    ├── <factor_name_2>/
    │   ├── 0.h5
    │   ├── 1.h5
    │   └── …
    └── …
```

Each **HDF5** file contains:
```
action/
  ├── gripper_position
  └── joint_position
observation/
  ├── image/
  │   ├── 0
  │   └── 2
  └── robot_state/
      ├── gripper_position
      └── joint_positions
```

## Running Simulation Experiments

We use the spatial peg-insertion experiment as an example. First create the dataset via motion planning:
```console
python guided_dc/envs/tasks/peg_insertion_solution.py
```
Note: `mplib` currently requires `numpy==1.26`. Run the script with that version, then switch back to `numpy>=2.0` for the remaining steps.

Then on a slurm cluster, run
```console
python submit_dp_jobs.py \
--config-name=sim_peg_insertion_dp_spatial_group \
dry_run=True
```

Use `dry_run=True` to first inspect the job commands. Omit it to submit slurm jobs. Results can be found in `results/`.

After all jobs finish, inside the corresponding result folder under `results/`, run `get_train_results.sh`, which fits factored scaling curves, computes expected improvements and trains new policies based on the updated dataset. To evaluate test performance, run `get_test_results.sh` inside the same folder.

## Contact
Questions? Reach out to Lihan Zha at lihanzha@princeton.edu.

## Citation

If this codebase helps your research, please cite:

```console
@misc{zha2025guidingdatacollectionfactored,
      title={Guiding Data Collection via Factored Scaling Curves}, 
      author={Lihan Zha and Apurva Badithela and Michael Zhang and Justin Lidard and Jeremy Bao and Emily Zhou and David Snyder and Allen Z. Ren and Dhruv Shah and Anirudha Majumdar},
      year={2025},
      eprint={2505.07728},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.07728}, 
}
```
