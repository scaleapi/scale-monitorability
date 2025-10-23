#!/bin/bash

# models = anthropic/claude-sonnet-4-20250514, gemini/gemini-2.5-pro-preview-06-05, bedrock/us.amazon.nova-premier-v1:0, fireworks_ai/llama4-maverick-instruct-basic, fireworks_ai/deepseek-r1

# python eval.py --model_name "anthropic/claude-sonnet-4-20250514"
# python eval.py --model_name "fireworks_ai/deepseek-r1"
# python eval.py --model_name "fireworks_ai/llama4-maverick-instruct-basic"
# python eval.py --model_name "bedrock/us.amazon.nova-premier-v1:0" --max_concurrent_requests 40
# python eval.py --model_name "gpt-4.1" --max_concurrent_requests 40
python eval.py --model_name "deepseek-ai/DeepSeek-V3" --max_concurrent_requests 80