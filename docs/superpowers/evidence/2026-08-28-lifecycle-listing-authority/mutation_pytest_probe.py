"""Emit exact pytest collection and execution node IDs for mutation admission."""

from __future__ import annotations


_CONFIG = None


def pytest_configure(config) -> None:
    global _CONFIG
    _CONFIG = config


def _terminal():
    if _CONFIG is None:
        raise RuntimeError("mutation_pytest_config_missing")
    terminal = _CONFIG.pluginmanager.get_plugin("terminalreporter")
    if terminal is None:
        raise RuntimeError("mutation_terminal_reporter_missing")
    return terminal


def pytest_collection_finish(session) -> None:
    terminal = _terminal()
    for item in session.items:
        terminal.write_line(f"TASK8_COLLECTED_NODE {item.nodeid}")


def pytest_runtest_logstart(nodeid, location) -> None:
    del location
    _terminal().write_line(f"TASK8_EXECUTED_NODE {nodeid}")
