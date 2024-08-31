"""
Test models against a list of prompts and parameters.
"""

import os
from datetime import datetime

import click
import ollama


def get_prompts(directory="prompts/"):
    """
    Recursively loads prompts from a directory tree.

    The function returns a dictionary where each key is a subdirectory name, and
    its corresponding value is another dictionary. The inner dictionary's keys are
    the file names (without extension) of prompt files found in that subdirectory,
    and their values are the contents of those files as strings.

    Args:
        directory (str): Path to the root directory containing prompts. Defaults to "prompts/".

    Returns:
        dict: A dictionary of dictionaries, where each inner dictionary contains prompts
            for a particular subdirectory.

    Examples:
        >>> get_prompts()
        {'subdir1': {'prompt1.txt': 'This is prompt 1', ...},
         'subdir2': {'prompt1.txt': 'This is another prompt 1', ...}}
    """

    prompts = {}

    for item in os.listdir(directory):
        path = os.path.join(directory, item)

        if os.path.isdir(path):
            prompts[item] = {}

            for file_path in os.scandir(path):
                with open(file_path, encoding="utf-8") as f:
                    # Use the file name (without extension) as the key
                    file_name = os.path.splitext(file_path.name)[0]
                    prompts[item][file_name] = f.read()

    return prompts


def run_prompt(model, prompt_test, options, test_result_file):
    """
    Run an OLLAMA prompt and generate a response.

    This function takes in a model, a prompt to test, options for the prompt,
    and a file path to save the test result. It uses the `ollama.generate`
    method to generate responses from the prompt and saves the prompts, options,
    and full response to the specified file.

    :param model: The OLLAMA model to use.
    :param prompt_test: The prompt to test.
    :param options: Options for the prompt.
    :param test_result_file: The file path to save the test result.
    """

    stream = ollama.generate(
        model=model,
        prompt=prompt_test,
        stream=True,
        options=options,
    )

    click.echo("\n")
    click.echo("Options: ")
    click.echo(options)

    full_response = ""
    for chunk in iter(stream):
        click.echo(chunk["response"], nl=False)
        full_response = full_response + chunk["response"]

    with open(test_result_file, "w", encoding="utf-8") as f:
        f.write(
            f"# Prompt\n"
            f"{prompt_test}\n\n"
            f"## Options\n"
            f"{options}\n\n"
            f"# Response\n"
            f"{full_response}\n"
        )


@click.command()
@click.option("--model", default="llama3.1", help="Model to test")
@click.option("--group", default=None, help="Test group to run")
@click.option("--seed", default=42, help="Random seed")
@click.option("--num_predict", default=None, help="Max tokens to generate")
# Temperature
@click.option("--temp", default=1, type=float, help="Temperature")
@click.option("--temp_min", default=None, type=float, help="Temperature min")
@click.option("--temp_max", default=None, type=float, help="Temperature max")
@click.option(
    "--temp_inc",
    default=0.1,
    help="How much should the temperature increase between min and max",
)
# Top_k
@click.option("--top_k", default=1, help="Number of top scoring predictions to display")
@click.option(
    "--top_k_min",
    default=None,
    type=int,
    help="Minimum number of top K predictions to consider",
)
@click.option(
    "--top_k_max",
    default=None,
    type=int,
    help="Maximum number of top K predictions to consider",
)
@click.option(
    "--top_k_inc",
    default=1,
    help="How many should the top K increase between min and max",
)
# Top_p
@click.option("--top_p", default=0.9, type=float, help="Cumulative probability of chosen tokens")
@click.option(
    "--top_p_min",
    default=None,
    type=float,
    help="Minimum cumulative probability to consider",
)
@click.option(
    "--top_p_max",
    default=None,
    type=float,
    help="Maximum cumulative probability to consider",
)
@click.option(
    "--top_p_inc",
    default=0.01,
    type=float,
    help="How much should the top P increase between min and max",
)
def start(
    model,
    group,
    seed,
    num_predict,
    # Temperature
    temp,
    temp_min,
    temp_max,
    temp_inc,
    # Top_k
    top_k,
    top_k_min,
    top_k_max,
    top_k_inc,
    # Top_p
    top_p,
    top_p_min,
    top_p_max,
    top_p_inc,
):
    """
    Start the tests with the specified parameters
    """
    click.echo("Selected options:")
    click.echo(f"  Model: {model}")
    click.echo(f"  Random seed: {seed}")
    click.echo(f"  Max tokens: {num_predict}")

    if temp_min is not None and temp_max is not None:
        temperature_values = []
        t = temp_min
        while t <= temp_max:
            temperature_values.append(round(t, 2))
            t = t + temp_inc
    else:
        temperature_values = [temp]
    click.echo(f"  Temperatures: {temperature_values}")

    if top_k_min is not None and top_k_max is not None:
        top_k_values = []
        k = top_k_min
        while k <= top_k_max:
            top_k_values.append(k)
            k = k + top_k_inc
    else:
        top_k_values = [top_k]
    click.echo(f"  Top K values: {top_k_values}")

    if top_p_min is not None and top_p_max is not None:
        top_p_values = []
        p = top_p_min
        while p <=top_p_max:
            top_p_values.append(round(p, 2))
            p = p + top_p_inc
    else:
        top_p_values = [top_p]
    click.echo(f"  Top P values: {top_p_values}")

    # Read prompts
    prompts = get_prompts()

    current_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    results_directory = f"results_{current_timestamp}"
    os.mkdir(results_directory)

    for group_name, group_prompt_test in prompts.items():
        # Filter group test
        if group is not None and group_name != group:
            continue

        test_group_directory = results_directory + "/" + group_name
        os.mkdir(test_group_directory)

        base_options = {
            "seed": seed,
            "num_predict": num_predict,
        }
        test_options = []
        
        for temperature in temperature_values:
            new_option = dict(base_options)
            new_option['temperature'] = temperature
            test_options.append(new_option)
        for top_k in top_k_values:
            new_option = dict(base_options)
            new_option['top_k'] = top_k
            test_options.append(new_option)
        for top_p in top_p_values:
            new_option = dict(base_options)
            new_option['top_p'] = top_p
            test_options.append(new_option)

        for prompt_test, prompt_contents in group_prompt_test.items():
            click.echo(f"Prompt: {prompt_contents}")

            # Iterate options
            for option_index, options in enumerate(test_options):
                test_result_file = (
                    f"{test_group_directory}/{prompt_test}_{option_index}"
                )

                run_prompt(
                    model=model,
                    prompt_test=prompt_contents,
                    options=options,
                    test_result_file=test_result_file,
                )

            # Iterate runtime options
            # num_predict: int
            # top_k: int
            # top_p: float
            # tfs_z: float
            # typical_p: float
            # repeat_last_n: int
            # temperature: float
            # repeat_penalty: float
            # presence_penalty: float
            # frequency_penalty: float
            # mirostat: int
            # mirostat_tau: float
            # mirostat_eta: float

            click.echo("")
            click.echo("---")
            click.echo("")


if __name__ == "__main__":
    start()
