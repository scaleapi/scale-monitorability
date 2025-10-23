#!/bin/bash

# Set environment variables
export ADVANCED_AI_RISK_PATH="/mnt/efs/oampatel/faithfulness/sycophancy/model-written-evals/advanced-ai-risk/human_generated_evals"
export PERSONA_DATA_PATH="/mnt/efs/oampatel/faithfulness/sycophancy/model-written-evals/persona"
export RESUME_DATA_PATH="/mnt/efs/oampatel/faithfulness/llm_bias/data/resume/selected_cats_resumes.csv"
export REWARD_HACKING_DATA_PATH="filtered_syc_to_subterfuge_settings.json"
export CACHE_DIR=".cache/"

# Default settings
DATASETS="mask sycophancy advanced_ai_risk persona resume sandbagging reward_hacking"
OUTPUT_FILE="perturbation_data.jsonl"
SEED=0

# Simple argument handling
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "Usage: $0 [datasets] [output_file] [seed]"
    echo "Example: $0 'mask sycophancy' my_output.jsonl 42"
    exit 0
fi

# Override defaults with arguments if provided
[[ -n "$1" ]] && DATASETS="$1"
[[ -n "$2" ]] && OUTPUT_FILE="$2"
[[ -n "$3" ]] && SEED="$3"

# Create cache directory
mkdir -p "$CACHE_DIR"

# Run the script
echo "Running with datasets: $DATASETS"
python create_data.py --datasets $DATASETS --output "$OUTPUT_FILE" --seed $SEED 