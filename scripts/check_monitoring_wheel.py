#!/usr/bin/env python3
"""Exercise packaged monitoring HTML and HTTP routes from a read-only installed core tree."""

import hashlib
import os
import tempfile
from pathlib import Path
from unittest.mock import patch


def main() -> None:
    from fastapi.testclient import TestClient

    import core.monitoring_dashboard as module
    from core.performance_monitor import PerformanceMonitor

    package_dir = Path(module.__file__).resolve().parent
    source_dir = Path(__file__).resolve().parents[1] / "core"
    assert package_dir != source_dir, "This check requires an isolated installed wheel"
    template = package_dir / "templates" / "dashboard.html"
    assert template.is_file(), "The wheel must ship the original dashboard template"
    before = {
        str(path.relative_to(package_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in package_dir.rglob("*")
        if path.is_file()
    }
    permissions = {path: path.stat().st_mode & 0o777 for path in [package_dir, *package_dir.rglob("*")]}
    try:
        for path in permissions:
            path.chmod(0o555 if path.is_dir() else 0o444)
        with tempfile.TemporaryDirectory(prefix="monitoring-wheel-") as directory:
            previous = Path.cwd()
            os.chdir(directory)
            monitor = PerformanceMonitor({"monitoring_interval": 60})
            try:
                with patch("socket.socket.connect", side_effect=AssertionError("No external monitoring service")):
                    dashboard = module.create_dashboard({"real_time_updates": False}, monitor)
                    assert dashboard.app is not None
                    with TestClient(dashboard.app) as client:
                        page = client.get("/")
                        assert page.status_code == 200
                        assert "MCP Academic RAG Server Monitoring" in page.text
                        metrics = client.get("/api/metrics")
                        assert metrics.status_code == 200
                        assert {"system", "aggregated", "rag", "performance"} <= metrics.json().keys()
                        assert client.get("/api/alerts").json() == []
                        assert client.get("/api/health").json()["status"] == "stopped"
                    dashboard.stop()
            finally:
                os.chdir(previous)
        after = {
            str(path.relative_to(package_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in package_dir.rglob("*")
            if path.is_file()
        }
        assert after == before, "Monitoring must not create or modify files in its installed package"
    finally:
        for path, permission in permissions.items():
            path.chmod(permission)
    print("Installed monitoring wheel: read-only package, HTML and HTTP contracts passed")


if __name__ == "__main__":
    main()
