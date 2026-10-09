"""Shared line transport for retained debug servers; EOF and notifications have explicit behavior."""

import asyncio
import json
from typing import Any, Awaitable, Callable, Dict, Optional, TextIO


async def serve_requests(
    handler: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
    reader: Optional[TextIO] = None,
    writer: Optional[TextIO] = None,
) -> None:
    import sys

    input_stream = reader or sys.stdin
    output_stream = writer or sys.stdout
    while True:
        line = await asyncio.to_thread(input_stream.readline)
        if line == "":
            return
        if not line.strip():
            continue
        request_id = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                response = {"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request"}, "id": None}
            else:
                request_id = request.get("id")
                if "id" not in request:
                    continue
                if not isinstance(request.get("params", {}), dict):
                    response = {
                        "jsonrpc": "2.0",
                        "error": {"code": -32602, "message": "Invalid params"},
                        "id": request_id,
                    }
                else:
                    response = await handler(request)
        except json.JSONDecodeError:
            response = {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None}
        except Exception as exc:
            response = {"jsonrpc": "2.0", "error": {"code": -32603, "message": str(exc)}, "id": request_id}
        output_stream.write(json.dumps(response) + "\n")
        output_stream.flush()
