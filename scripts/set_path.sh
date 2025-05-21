#!/bin/bash

##################### Paths #####################

# Set default paths
DEFAULT_DATA_DIR="${PWD}/data"
DEFAULT_LOG_DIR="${PWD}/log"
DEFAULT_ASSETS_DIR="${PWD}/guided_dc/assets"
DEFAULT_RESULTS_DIR="${PWD}/results"

# Prompt the user for input, allowing overrides
read -p "Enter the desired data directory [default: ${DEFAULT_DATA_DIR}], leave empty to use default: " DATA_DIR
GDC_DATA_DIR=${DATA_DIR:-$DEFAULT_DATA_DIR}  # Use user input or default if input is empty

read -p "Enter the desired logging directory [default: ${DEFAULT_LOG_DIR}], leave empty to use default: " LOG_DIR
GDC_LOG_DIR=${LOG_DIR:-$DEFAULT_LOG_DIR}  # Use user input or default if input is empty

read -p "Enter the desired asset directory [default: ${DEFAULT_ASSETS_DIR}], leave empty to use default: " ASSETS_DIR
GDC_ASSETS_DIR=${ASSETS_DIR:-$DEFAULT_ASSETS_DIR}  # Use user input or default if input is empty

read -p "Enter the desired results directory [default: ${DEFAULT_RESULTS_DIR}], leave empty to use default: " RESULTS_DIR
GDC_RESULTS_DIR=${RESULTS_DIR:-$DEFAULT_RESULTS_DIR}  # Use user input or default if input is empty


# Export to current session
export GDC_DATA_DIR="$GDC_DATA_DIR"
export GDC_LOG_DIR="$GDC_LOG_DIR"
export GDC_ASSETS_DIR="$GDC_ASSETS_DIR"
export GDC_RESULTS_DIR="$GDC_RESULTS_DIR"

# Confirm the paths with the user
echo "Data directory set to: $GDC_DATA_DIR"
echo "Log directory set to: $GDC_LOG_DIR"
echo "Assets directory set to: $GDC_ASSETS_DIR"
echo "Results directory set to: $GDC_RESULTS_DIR"

# Append environment variables to .bashrc
echo "export GDC_DATA_DIR=\"$GDC_DATA_DIR\"" >> ~/.bashrc
echo "export GDC_LOG_DIR=\"$GDC_LOG_DIR\"" >> ~/.bashrc
echo "export GDC_ASSETS_DIR=\"$GDC_ASSETS_DIR\"" >> ~/.bashrc
echo "export GDC_RESULTS_DIR=\"$GDC_RESULTS_DIR\"" >> ~/.bashrc

echo "Environment variables GDC_DATA_DIR, GDC_LOG_DIR, GDC_ASSETS_DIR, GDC_RESULTS_DIR added to .bashrc and applied to the current session."

##################### WandB #####################

# Prompt the user for input, allowing overrides
read -p "Enter your WandB entity (username or team name), leave empty to skip: " ENTITY

# Check if ENTITY is not empty
if [ -n "$ENTITY" ]; then
  # If ENTITY is not empty, set the environment variable
  export GDC_WANDB_ENTITY="$ENTITY"

  # Confirm the entity with the user
  echo "WandB entity set to: $GDC_WANDB_ENTITY"

  # Append environment variable to .bashrc
  echo "export GDC_WANDB_ENTITY=\"$ENTITY\"" >> ~/.bashrc

  echo "Environment variable GDC_WANDB_ENTITY added to .bashrc and applied to the current session."
else
  # If ENTITY is empty, skip setting the environment variable
  echo "No WandB entity provided. Please set wandb=null when running scripts to disable wandb logging and avoid error."
fi
