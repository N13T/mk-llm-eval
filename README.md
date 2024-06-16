# mk-llm-eval
Evaluation framework for LLMs in Macedonian

This simple app can be used to get quick evaluation scores against several popular benchmarks that have been translated to Macedonian.

The app is compatible with openai's api format, more precisely the chat completions endpoint. This means it can be used to target openai's models or a custom model that is running in an openai compatible server via a framework such as vLLM for example.

### Usage
To run the benchmarks just run the app via the cli:
```bash
python eval.py --benchmark trajkovnikola/arc_easy_mk --benchmark_split train --model_endpoint http://model-endpoint/v1 --model meta-llama/Meta-Llama-3-70B-Instruct  --num_samples 5000 --use_system_prompt
```

Arguments:
- benchmark: name of the benchmark to run
    Currently supported benchmarks are:
    - trajkovnikola/arc_easy_mk
    - trajkovnikola/arc_challenge_mk
    - trajkovnikola/winogrande_mk
    - classla/COPA-MK
- benchmark_split: different datasets have different splits
- model_endpoint: the endpoint where the model is served
- model: the model we want to use
- num_samples: number of samples from the dataset to be used, default 10 for testing
- use_system_prompt: whether to add a system prompt, if the model supports it

After the app finishes, it will print out the score achieved and save a csv file with the name of the benchmark with the responses to each sample.

To add a new benchmark, first of all we need the translated dataset to be pushed to huggingface and then we need to extend the PromptPrepper and ResultsParser classes if necessary to accommodate for the new benchmark.
aasdzx 3695