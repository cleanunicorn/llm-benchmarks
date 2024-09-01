# LLM Benchmark

This project is designed to test and benchmark Large Language Models (LLMs) against a variety of prompts with different parameters.

## Features

- Test multiple prompts across different subdirectories
- Customizable model parameters including temperature, top_k, and top_p
- Ability to run tests for specific prompt groups
- Saves test results in a structured directory format
- Uses OLLAMA for model interaction

## Requirements

- Python 3.11
- poetry
- ollama

## Installation

1. Clone this repository
2. Install the required packages:

```terminal
poetry install
```

## Usage

Run the script using the following command:

```
python main.py [OPTIONS]
```

### Options

- `--model TEXT`: Model to test (default: "llama3.1")
- `--group TEXT`: Test group to run
- `--seed INTEGER`: Random seed (default: 42)
- `--num_predict INTEGER`: Max tokens to generate
- `--temp FLOAT`: Temperature (default: 1.0)
- `--temp_min FLOAT`: Minimum temperature
- `--temp_max FLOAT`: Maximum temperature
- `--temp_inc FLOAT`: Temperature increment (default: 0.1)
- `--top_k INTEGER`: Number of top scoring predictions to consider
- `--top_k_min INTEGER`: Minimum number of top K predictions
- `--top_k_max INTEGER`: Maximum number of top K predictions
- `--top_k_inc INTEGER`: Top K increment (default: 1)
- `--top_p FLOAT`: Cumulative probability of chosen tokens (default: 0.9)
- `--top_p_min FLOAT`: Minimum cumulative probability
- `--top_p_max FLOAT`: Maximum cumulative probability
- `--top_p_inc FLOAT`: Top P increment (default: 0.01)

## Directory Structure

- `prompts/`: Directory containing prompt files organized in subdirectories
- `results_[TIMESTAMP]/`: Directory where test results are saved

## Output

The script generates a separate file for each test, containing:

- The prompt used
- The options/parameters for the test
- The full response from the model

## Contributing

Contributions to improve the benchmark or add new features are welcome. Please submit a pull request or open an issue to discuss proposed changes.

## License

Apache-2.0
