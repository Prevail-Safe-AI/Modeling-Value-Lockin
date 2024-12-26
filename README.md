# Modeling-Value-Lockin

## Example Usage

- To run experiment without finetuning: `python run_experiment.py run_experiment`
- To enable live finetuning: `python run_experiment.py run_experiment --do_finetuning`
- To visualize the conversation (possibly to test and iterate the prompt design): `python run_experiment.py run_experiment --dynamic_printing --parallel_convos 1`
- To customize parameters: `python run_experiment.py --tutor "mistralai/Mistral-7B-Instruct-v0.3" --user "mistralai/Mistral-7B-Instruct-v0.3" run_experiment --num_rounds 50 --num_turns_per_round 20 --parallel_convos 5000 --do_finetuning`
  - You could also specify any subset of these arguments. The model names must be placed before `run_experiment`, and the other arguments must be placed after `run_experiment`.
- Backend loading for up to a few minutes is expected behavior. If backend keeps loading for a long time, please try running the code with `LOUD_BACKEND=1` environment variable set to see the detailed error message.

## Development Guide

- Please make sure to follow the [commit message convention](https://www.conventionalcommits.org/en/v1.0.0/).
- We tentatively use ProgressGym as our model training/inference & dataset management infrastructure. See [examples](https://github.com/PKU-Alignment/ProgressGym/tree/main/examples/abstractions) and [documentation](https://pku-alignment.github.io/ProgressGym/).
- Please avoid changing anything under the ProgressGym/ directory! Any change there will be pushed to the original ProgressGym repository as is.
- You could inspect the content of a Data instance with `print(list(data.all_passages()))` or manually checking out the file at `data.data_path`.