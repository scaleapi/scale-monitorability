#!/bin/bash

# Set environment variables - update these paths for your environment
export ADVANCED_AI_RISK_PATH="/mnt/efs/oampatel/faithfulness/sycophancy/model-written-evals/advanced-ai-risk/human_generated_evals"
export PERSONA_DATA_PATH="/mnt/efs/oampatel/faithfulness/sycophancy/model-written-evals/persona"
export RESUME_DATA_PATH="/mnt/efs/oampatel/faithfulness/llm_bias/data/resume/selected_cats_resumes.csv"
export REWARD_HACKING_DATA_PATH="/mnt/efs/oampatel/faithfulness/sycophancy-to-subterfuge-paper/cleaned_data.json"
export CACHE_DIR=".cache/sycophancy_cleaning"

# Run the script
cd data
python create_data.py "$@" 