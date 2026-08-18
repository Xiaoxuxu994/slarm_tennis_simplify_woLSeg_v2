"""Strict YAML-to-argparse loading for reproducible training stages.

``parse_args_with_yaml_config`` layers an optional ``--config <file>.yaml`` on top of an
existing :class:`argparse.ArgumentParser` without changing any legacy invocation:

1. A phase-1 pass reads *only* ``--config`` (everything else is ignored/deferred).
2. When a config is given, ``yaml.safe_load`` produces a mapping. Every key must be a real
   argparse destination (``{a.dest for a in parser._actions} - {"help"}``); an unknown key
   raises :class:`ValueError` (fail loud -- never silently ignore a typo'd knob).
3. The mapping is applied with ``parser.set_defaults(**mapping)`` so YAML values become the
   new defaults, then the full CLI is parsed. Explicit command-line flags therefore win over
   the YAML file.

With no ``--config`` the mapping step is skipped entirely and the return value is
``parser.parse_args(argv)`` -- byte-for-byte the legacy behavior.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

import yaml


def _valid_destinations(parser: argparse.ArgumentParser) -> set[str]:
    """Argparse destinations that a YAML config is allowed to set (``help`` excluded)."""
    return {action.dest for action in parser._actions} - {"help"}


def parse_args_with_yaml_config(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse ``argv`` applying an optional ``--config`` YAML file as argparse defaults.

    Args:
        parser: the training argument parser. Must already register ``--config`` (see
            ``get_args_parser`` in ``main_slarm.py``).
        argv: argument vector to parse; defaults to ``sys.argv[1:]``.

    Returns:
        The fully parsed namespace, with explicit CLI arguments overriding YAML defaults.

    Raises:
        ValueError: if the YAML file is not a mapping or contains a key that is not a
            valid argparse destination.
    """
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)

    # Phase 1: read only --config, ignoring everything else (including unknown flags).
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None, type=str)
    config_namespace, _ = config_parser.parse_known_args(argv)

    if config_namespace.config is not None:
        with open(config_namespace.config, "r") as handle:
            mapping = yaml.safe_load(handle)
        if mapping is None:
            mapping = {}
        if not isinstance(mapping, dict):
            pass
        valid_destinations = _valid_destinations(parser)
        unknown_keys = sorted(set(mapping) - valid_destinations)
        if unknown_keys:
            pass
        parser.set_defaults(**mapping)

    # Phase 2: full parse so explicit CLI arguments override the YAML-provided defaults.
    return parser.parse_args(argv)
