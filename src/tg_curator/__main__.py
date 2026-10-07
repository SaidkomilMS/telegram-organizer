"""Entry point for ``python -m tg_curator``; the same as the ``curator`` console script."""

from __future__ import annotations

import importlib.util
import sys


def main() -> None:
    """Run the CLI.

    The import is deferred so that ``import tg_curator`` stays cheap and does not pull in click
    and the whole runtime for callers that only need the library. An installation without the
    CLI module (a partial build) is reported in one sentence rather than a traceback; an
    ImportError from inside ``cli`` itself (a missing dependency) is not masked.
    """
    if importlib.util.find_spec("tg_curator.cli") is None:
        sys.stderr.write(
            "tg-curator: the command-line interface (tg_curator.cli) is not part of this "
            "installation; reinstall the package to get the `curator` command.\n"
        )
        sys.exit(1)
    from tg_curator.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
