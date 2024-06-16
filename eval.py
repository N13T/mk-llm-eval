import argparse
import asyncio
import aiohttp
from datetime import datetime
from datasets import load_dataset
from logger import AppLogger
from tasks import PromptPrepper, ResultsParser


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--benchmark_split", type=str, required=True)
    parser.add_argument("--model_endpoint", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--use_system_prompt", action="store_true")
    parser.add_argument("--num_samples", type=int, default=10)

    return parser.parse_args()


async def send_request(openai_api_base, openai_api_key, session, model, sample):
    async with session.post(
        openai_api_base + "/chat/completions",
        json={
            "messages": sample,
            "temperature": 0.0,
            "model": model,
            "max_tokens": 5,
            "stop": [
                "<|im_end|>",
            ],
        },
        headers={"Content-Type": "application/json", "Authorization": openai_api_key},
    ) as response:
        response_content = await response.json()
        response_content = response_content["choices"][0]["message"]["content"]
        return response_content


logger = AppLogger(__name__).logger


async def main():

    start_time = datetime.now()

    args = parse_args()
    openai_api_key = "EMPTY"
    openai_api_base = args.model_endpoint
    benchmark_dataset = load_dataset(args.benchmark, split=args.benchmark_split)
    benchmark_name = args.benchmark.split("/")[1]

    logger.info(f"Starting evaluation on benchmark: {benchmark_name}")
    logger.info(
        f"Running evaluation on {args.num_samples}, benchmark has a total of {benchmark_dataset.shape[0]} samples."
    )

    prompts = []
    for i, prompt in enumerate(benchmark_dataset):
        sample = PromptPrepper(
            system_prompt=args.use_system_prompt,
            benchmark_name=benchmark_name,
            sample=prompt,
        )
        sample.sample_prep()
        prompts.append(sample.prepped_sample)
        if i == args.num_samples:
            break

    timeout = aiohttp.ClientTimeout(total=1200)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(
            *[
                send_request(
                    openai_api_base=openai_api_base,
                    openai_api_key=openai_api_key,
                    session=session,
                    model=args.model,
                    sample=x,
                )
                for x in prompts
            ]
        )
        end_time = datetime.now()
        time_to_finish = end_time - start_time

        results_preper = ResultsParser(
            benchmark_name=benchmark_name, data=benchmark_dataset, responses=results
        )
        results_preper.prep_results()

        logger.info(f"Done in: {time_to_finish}")


if __name__ == "__main__":
    asyncio.run(main())
