import argparse 
import asyncio 
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field
import logging
from copy import deepcopy
from datetime import datetime
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.console import Console
from dotenv import load_dotenv
from functools import partial

load_dotenv()

from utils import cot_formatter, insert_text_formatter, parse_final_answer, parse_cot, convert_judge_output_to_bool, load_jsonl, save_jsonl
from cached_api import RoutedAsyncOpenAI
from graders import CATEGORY_BASED_GRADER_MAP, ALL_FAITHFULNESS_GRADER_PROMPTS

console = Console()

@dataclass
class EvalConfig:
    """Configuration for evaluation pipeline"""
    data_path: str
    cot_type: str
    model_name: str
    processing_model_name: str
    output_path: str
    max_concurrent_requests: int = 10
    cache_dir: Path = Path(".cache")
    sampling_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.output_path is None:
            self.output_path = self._generate_output_path()
        if self.sampling_kwargs.get('temperature', 0.0) == 0.0:
            raise ValueError("Temperature must be > 0.0 for this evaluation because we run a resample baseline")
        self.sampling_kwargs['max_tokens'] = 2048

    def _generate_output_path(self) -> str:
        """Generate output path using model-first structure with deterministic naming"""
        dataset_name = Path(self.data_path).stem
        
        # Generate config string from sampling_kwargs
        config_parts = []
        for key, value in sorted(self.sampling_kwargs.items()):
            if isinstance(value, float):
                config_parts.append(f"{key}-{value:.1f}")
            else:
                config_parts.append(f"{key}-{value}")
        
        # Add cot_type to config if it's not the default 'cot'
        if self.cot_type != 'cot':
            config_parts.append(f"cot-{self.cot_type}")
        
        config_str = "_".join(config_parts) if config_parts else "default"
        
        # Create results directory structure
        results_dir = Path("results") / self.model_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate deterministic filename (no timestamp)
        filename = f"{dataset_name}_{config_str}.jsonl"
        
        return str(results_dir / filename)

class EvaluationPipeline:
    """Main evaluation pipeline for faithfulness testing"""
    
    def __init__(self, model_name: str, processing_model_name: str, cot_type: str, max_concurrent_requests: int = 80, cache_dir: Path = Path(".cache"), **kwargs):
        self.model_name = model_name
        self.processing_model_name = processing_model_name
        self.max_concurrent_requests = max_concurrent_requests
        self.cot_type = cot_type
        self.cot_formatter = partial(cot_formatter, cot_type=cot_type)
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.sampling_kwargs = kwargs

        self.client = RoutedAsyncOpenAI(
            enable_cache=True, 
            cache_dir=str(cache_dir),
        )

    
    async def evaluate(self, data: list[dict]) -> list[dict]:
        """Run complete evaluation pipeline"""

        console.print("[bold blue]Starting evaluation pipeline...[/bold blue]")
        
        data = await self._run_original_prompts(data)
        data = await self._run_perturbed_prompts(data)
        data = await self._judge_and_create_inconsistency_labels(
            data, 
            'parsed_model_response', 
            'perturbed_prompt',
            'parsed_perturbed_model_response', 
            'inconsistency_label', 
            'judge diff prompt'
        )
        data = await self._judge_and_create_inconsistency_labels(
            data, 
            'parsed_model_response', 
            'prompt', 
            'parsed_baseline_model_response', 
            'baseline_inconsistency_label', 
            'judge baseline diff prompt'
        )
        data = await self._run_faithfulness_checks(data)
        
        console.print("[bold green]Evaluation pipeline completed![/bold green]")
        return data

    def _fill_column(self, data: list[dict], col_name: str, idxs: list[int], values: list[Any]) -> list[dict]:
        """Fill a column in data at the given indices with the given values otherwise put as None"""
        if len(idxs) != len(values):
            raise ValueError(f"Length of idxs ({len(idxs)}) and values ({len(values)}) must match")
        
        col = [None] * len(data)
        for idx, value in zip(idxs, values):
            col[idx] = value
        
        for row, value in zip(data, col):
            row[col_name] = value

        return data
    
    def _create_filtered_prompts(self, data: list[dict], filter_func, prompt_func) -> tuple[list[int], list[list[dict[str, str]]]]:
        """Create prompts from data with filtering, safely handling empty results"""
        prompt_tuples = [(idx, prompt_func(idx, row)) 
                        for idx, row in enumerate(data) 
                        if filter_func(row)]
        
        if not prompt_tuples:
            return [], []
        idxs, prompts = zip(*prompt_tuples)
        return list(idxs), list(prompts)
    
    async def _make_api_call_with_semaphore(self, model_name: str, prompt: list[dict[str, str]]):
        """Make a single API call with semaphore control"""
        async with self.semaphore:
            return await self.client.chat(
                model=model_name,   
                messages=prompt,        
                **self.sampling_kwargs
            )

    def _parse_litellm_response(self, result: Any) -> str: 
        # sometimes litellm does some processing of the response into reasoning_content
        if hasattr(result.choices[0].message, 'reasoning_content'):
            return f"<think>{result.choices[0].message.reasoning_content}</think>\n{result.choices[0].message.content}"
        else:
            return result.choices[0].message.content
    
    async def _batch_api_call(self, prompts: list[list[dict[str, str]]], desc: str, model_name: str) -> list[str]: 
        """Execute batch API calls with progress tracking and semaphore control"""
        if not prompts:
            return []
        
        tasks = [asyncio.create_task(self._make_api_call_with_semaphore(model_name, prompt)) 
                for prompt in prompts]
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=10,
            auto_refresh=True,
        ) as progress:
            progress_task = progress.add_task(desc, total=len(prompts))
            
            responses: list[str] = ["" for _ in range(len(tasks))]
            task_to_index = {task: i for i, task in enumerate(tasks)}
            pending = set(tasks)
            
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for completed_task in done:
                    result = await completed_task
                    index = task_to_index[completed_task]
                    responses[index] = self._parse_litellm_response(result)
                    progress.advance(progress_task)
                    progress.refresh()
        
        return responses
    
    async def _parse_responses(self, data: list[dict], response_field: str, parsed_field: str) -> list[dict]:
        """Parse responses using the existing parsing logic"""
        idxs, parse_prompts = self._create_filtered_prompts(
            data,
            lambda row: row['parse_model_response_prompt'] is not None,
            lambda idx, row: insert_text_formatter(
                deepcopy(row['parse_model_response_prompt']), 
                row, 
                {'model_response': row[response_field]}
            )
        )
        
        parse_responses = await self._batch_api_call(parse_prompts, f"Running parse prompt for {response_field}", self.processing_model_name)
        
        for idx, row in enumerate(data):
            row[parsed_field] = row[response_field]

        for idx, parse_response in zip(idxs, parse_responses):
            data[idx][parsed_field] = parse_response
        
        for idx, row in enumerate(data):
            row[parsed_field] = parse_final_answer(row[parsed_field])
        
        return data
    
    async def _run_original_prompts(self, data: list[dict]) -> list[dict]:
        """Run original prompts and parse responses"""
        idxs, prompts = self._create_filtered_prompts(
            data,
            lambda row: row['prompt'] is not None,
            lambda idx, row: self.cot_formatter(deepcopy(row['prompt']))
        )
        
        responses = await self._batch_api_call(prompts, "Running prompt", self.model_name)
        first_responses = responses
        
        # add a preceding space to the prompt to make it a new request 
        for prompt in prompts: 
            prompt[-1]['content'] = ' ' + prompt[-1]['content']
        responses = await self._batch_api_call(prompts, "Running resampled prompt", self.model_name)
        second_responses = responses
        
        data = self._fill_column(data, 'model_response', idxs, first_responses)
        data = self._fill_column(data, 'baseline_model_response', idxs, second_responses)
        
        data = await self._parse_responses(data, 'model_response', 'parsed_model_response')
        data = await self._parse_responses(data, 'baseline_model_response', 'parsed_baseline_model_response')
        
        return data
    
    async def _run_perturbed_prompts(self, data: list[dict]) -> list[dict]:
        """Run perturbed prompts and parse responses"""
        idxs, prompts = self._create_filtered_prompts(
            data,
            lambda row: row['perturbed_prompt'] is not None,
            lambda idx, row: self.cot_formatter(insert_text_formatter(deepcopy(row['perturbed_prompt']), row))
        )
        
        responses = await self._batch_api_call(prompts, "Running perturbed prompt", self.model_name)
        
        data = self._fill_column(data, 'perturbed_model_response', idxs, responses)
        
        idxs, parse_prompts = self._create_filtered_prompts(
            data,
            lambda row: row['parse_perturbed_model_response_prompt'] is not None,
            lambda idx, row: insert_text_formatter(
                deepcopy(row['parse_perturbed_model_response_prompt']), 
                row, 
                {'model_response': row['perturbed_model_response']}
            )
        )
        
        parse_responses = await self._batch_api_call(parse_prompts, "Running parse perturbed prompt", self.processing_model_name)
        
        for idx, row in enumerate(data):
            row['parsed_perturbed_model_response'] = parse_final_answer(row['perturbed_model_response'])
        
        for idx, parse_response in zip(idxs, parse_responses):
            data[idx]['parsed_perturbed_model_response'] = parse_response
        
        for idx, row in enumerate(data):
            row['perturbed_chain_of_thought'] = parse_cot(row['perturbed_model_response'])
            row['parsed_perturbed_model_response'] = parse_final_answer(row['parsed_perturbed_model_response'])
        
        return data
    
    async def _judge_and_create_inconsistency_labels(self, 
        data: list[dict], 
        response1_field: str, 
        prompt2_field: str, 
        response2_field: str, 
        label_field: str, 
        desc: str,
        ) -> list[dict]:
        """Judge differences between two response fields and create inconsistency labels"""
        idxs, prompts = self._create_filtered_prompts(
            data,
            lambda row: row['judge_diff_prompt'] is not None,
            lambda idx, row: insert_text_formatter(
                deepcopy(row['judge_diff_prompt']), 
                row, 
                {
                    'model_response': row[response1_field], 
                    'perturbed_prompt': row[prompt2_field],
                    'perturbed_model_response': row[response2_field]
                }
            )
        )
        
        responses = await self._batch_api_call(prompts, f"Running {desc}", self.processing_model_name)
        
        data = self._fill_column(data, label_field, idxs, [convert_judge_output_to_bool(response) for response in responses])
        
        for row in data:
            if row[label_field] is None:
                if row['category'] in CATEGORY_BASED_GRADER_MAP:
                    grader = CATEGORY_BASED_GRADER_MAP[row['category']]
                    row[label_field] = grader(row[response1_field], row[response2_field], row=row)
                else:
                    console.print(f"[yellow]Warning: No grader found for category '{row['category']}', setting label to False[/yellow]")
                    row[label_field] = None
        
        return data
    
    async def _run_faithfulness_checks(self, data: list[dict]) -> list[dict]:
        """Run faithfulness evaluation for all grader prompts"""
        for col_name, faithfulness_eval_prompt in ALL_FAITHFULNESS_GRADER_PROMPTS.items():
            for row in data:
                row[f'{col_name}_response'] = None
                row[f'{col_name}_pred'] = None
            
            idxs, prompts = self._create_filtered_prompts(
                data,
                lambda row: True, # true because simulability is always run and category specific is always run
                lambda idx, row: insert_text_formatter(
                    deepcopy([{'role': 'user', 'content': faithfulness_eval_prompt[row['category']]}]),
                    row, 
                )
            )
            
            responses = await self._batch_api_call(prompts, f"Running {col_name}", self.processing_model_name)  
            
            data = self._fill_column(data, f'{col_name}_response', idxs, responses)
            data = self._fill_column(data, f'{col_name}_pred', idxs, [parse_final_answer(response) for response in responses])
        
        return data


async def main(args):
    """Main evaluation pipeline"""
    config = EvalConfig(
        data_path=args.data_path,
        model_name=args.model_name,
        processing_model_name=args.processing_model_name,
        cot_type=args.cot_type,
        output_path=args.output_path,
        max_concurrent_requests=args.max_concurrent_requests,
        sampling_kwargs={
            'temperature': args.temperature,
        }
    )
    
    data = load_jsonl(config.data_path)
    
    evaluator = EvaluationPipeline(
        config.model_name, 
        config.processing_model_name, 
        config.cot_type,
        config.max_concurrent_requests,
        config.cache_dir, 
        **config.sampling_kwargs
    )
    
    results = await evaluator.evaluate(data)
    
    save_jsonl(results, config.output_path)
    console.print(f"[green]Saved to {config.output_path}[/green]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=False, default='data/perturbation_data.jsonl')
    parser.add_argument('--cot_type', type=str, required=False, default='cot', choices=['cot', 'no_cot', 'private_cot', 'public_cot', 'include_all_factors', 'qwq', 'qwen'])
    parser.add_argument('--model_name', type=str, required=False, default='gpt-4o')
    parser.add_argument('--processing_model_name', type=str, required=False, default='gpt-4o')
    parser.add_argument('--output_path', type=str, required=False, default=None)
    parser.add_argument('--temperature', type=float, required=False, default=1.0)
    parser.add_argument('--max_concurrent_requests', type=int, required=False, default=80, help='Maximum number of concurrent API requests')
    args = parser.parse_args()

    asyncio.run(main(args))