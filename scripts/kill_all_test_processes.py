#!/usr/bin/env python3
"""Compatibility cleanup API; only explicitly registered owned resources are released."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.utils.cleanup import _global_cleaner, emergency_cleanup, kill_port


def cleanup_processes():
    _global_cleaner.cleanup_processes()


def cleanup_ports():
    # No port/name scanning: registered servers own their shutdown.
    _global_cleaner.cleanup_servers()


def cleanup_temp_files():
    _global_cleaner.cleanup_temp_files()


def kill_processes_by_name(process_names):
    cleanup_processes()


def kill_processes_by_port(ports):
    for port in ports:
        kill_port(port)


def main():
    emergency_cleanup()
    print("Registered resources released; unregistered processes and files preserved")


if __name__ == "__main__":
    main()
