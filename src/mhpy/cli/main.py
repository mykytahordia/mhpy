import argparse
import sys

from mhpy.cli.commands.initialize import register_init_args
from mhpy.utils.common import configure_logger


def main() -> None:
    configure_logger(save_logs=False)
    parser = argparse.ArgumentParser(prog="mhpy", description="A helper CLI for automating ML project setup.")

    register_init_args(parser.add_subparsers(dest="command", required=True))

    args = parser.parse_args()

    try:
        args.func(args)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
