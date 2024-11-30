# Modeling-Value-Lockin

- Please make sure to follow the [commit message convention](https://www.conventionalcommits.org/en/v1.0.0/).
- We tentatively use ProgressGym as our model training/inference & dataset management infrastructure. See [examples](https://github.com/PKU-Alignment/ProgressGym/tree/main/examples/abstractions) and [documentation](https://pku-alignment.github.io/ProgressGym/).
- If backend keeps loading for a long time, please try running the code with `LOUD_BACKEND=1` environment variable set to see the detailed error message.
- Please avoid changing anything under the ProgressGym/ directory! Any change there will be pushed to the original ProgressGym repository as is.
- You could inspect the content of a Data instance with `print(list(data.all_passages()))` or manually checking out the file at `data.data_path`.