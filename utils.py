import json
import time
import argparse
from colorama import init

init()
start = time.time()

# ANSI color codes
GRAY = "\033[90m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def load_config(filename):
    """
    Loads a configuration from a JSON file.

    Attempts to open and read the specified JSON configuration file, returning
    the parsed JSON object. If the file cannot be opened or read, returns an empty
    dictionary.

    Parameters:
        filename (str): The path to the JSON configuration file.

    Returns:
        dict: The loaded configuration dictionary or an empty dictionary if the file
        could not be read.
    """
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print_formatted(f"Configuration file not found, using defaults", YELLOW)
        return {}


def print_formatted(message, color=RESET):
    """
    Prints a message with elapsed time since program start, in the specified color.

    Parameters:
        message (str): The message to print.
        color (str): The ANSI color code to use for the message text. Defaults to no color formatting (RESET).
    """
    elapsed_time = time.time() - start
    print("%s[%8.2f] %s%s%s" % (GRAY, elapsed_time, color, message, RESET))


def print_game_letterhead(title="Imitation Learning"):
    """
    Prints a formatted letterhead for the application.

    The letterhead includes the game title and a subtitle, both centered within
    a dashed line border.
    """
    print("-" * 60)
    print("|{}{:^58s}{}|".format(BOLD, title, RESET))
    print("|{}{:^58s}{}|".format(GRAY, "ENDEAVR", RESET))
    print("-" * 60)


def parse_args():
    """
    Parses command line arguments for the application.

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="2D Imitation Learning Application")
    # 35mph = 15.6464 m/s, so a sampling rate of 10Hz is 1.56464m per sample
    parser.add_argument('-r', '--sampling-rate', type=float, default=10,
                        help="Sampling rate for the car agent. Default is 10 Hz.")
    parser.add_argument('-c', '--config', type=str, default='config.json',
                        help="Path to the configuration file (JSON).")
    return parser.parse_args()


def print_args(args):
    """
    Prints the parsed command line arguments.

    Parameters:
        args (argparse.Namespace): The parsed command line arguments.
    """
    for arg in vars(args):
        print_formatted(f"{arg}: {GREEN}{getattr(args, arg)}{RESET}")
