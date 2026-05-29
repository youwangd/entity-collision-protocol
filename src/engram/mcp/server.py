"""Engram MCP server — exposes memory as MCP tools.

10 tools defined in DESIGN.md:
- engram_remember, engram_recall, engram_forget
- engram_capture, engram_consolidate
- engram_pin, engram_unpin, engram_context
- engram_status, engram_affect
"""

from __future__ import annotations

import json
import logging

from engram import __version__
from engram.core.types import MemoryType
from engram.engine import Engram

logger = logging.getLogger(__name__)


class MCPServer:
    """MCP tool server for Engram.

    Provides a tool-based interface following the MCP protocol.
    Each tool is a method that accepts JSON input and returns JSON output.
    """

    def __init__(self, engram: Engram):
        self.engram = engram
        self._tools = {
            "engram_remember": self.tool_remember,
            "engram_recall": self.tool_recall,
            "engram_forget": self.tool_forget,
            "engram_capture": self.tool_capture,
            "engram_consolidate": self.tool_consolidate,
            "engram_pin": self.tool_pin,
            "engram_unpin": self.tool_unpin,
            "engram_context": self.tool_context,
            "engram_status": self.tool_status,
            "engram_affect": self.tool_affect,
            "engram_schemas": self.tool_schemas,
            "engram_trace": self.tool_trace,
        }

    def list_tools(self) -> list[dict]:
        """Return MCP tool definitions."""
        return [
            {
                "name": "engram_remember",
                "description": "Store a memory explicitly. Use for important facts, preferences, decisions.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "What to remember"},
                        "salience": {"type": "number", "description": "Importance 0.0-1.0", "default": 0.0},
                        "type": {"type": "string", "enum": ["fact", "episode", "skill", "schema"], "default": "fact"},
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "engram_recall",
                "description": "Search memories. Returns relevant memories ranked by score.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for"},
                        "limit": {"type": "integer", "description": "Max results", "default": 5},
                        "depth": {"type": "string", "enum": ["L0", "L1", "L2"], "default": "L1"},
                        "include_faded": {"type": "boolean", "default": False},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "engram_forget",
                "description": "Suppress or delete memories. Soft forget is recoverable; hard is permanent (GDPR).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Memory ID to forget"},
                        "query": {"type": "string", "description": "Query to match memories to forget"},
                        "hard": {"type": "boolean", "description": "Hard delete (permanent)", "default": False},
                    },
                },
            },
            {
                "name": "engram_capture",
                "description": "Capture a raw event for later consolidation. Not immediately searchable.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Event content"},
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "engram_consolidate",
                "description": "Run the consolidation pipeline (brain's sleep cycle). Processes buffered events.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "engram_pin",
                "description": "Pin a fact to active context. Pinned items always appear in context.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "What to pin"},
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "engram_unpin",
                "description": "Remove a pin from active context.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pin_id": {"type": "string", "description": "Pin ID to remove"},
                    },
                    "required": ["pin_id"],
                },
            },
            {
                "name": "engram_context",
                "description": "Get active context for prompt injection. Returns L0 summaries + pins.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "max_tokens": {"type": "integer", "default": 4096},
                    },
                },
            },
            {
                "name": "engram_status",
                "description": "Memory system stats and health.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "engram_affect",
                "description": "Get or trigger affect state (mood, emotions, temperament).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "trigger": {"type": "string", "description": "Emotion to trigger (joy, trust, fear, surprise, sadness, disgust, anger, anticipation)"},
                        "intensity": {"type": "number", "description": "Emotion intensity 0.0-1.0", "default": 0.5},
                    },
                },
            },
            {
                "name": "engram_schemas",
                "description": "List auto-generated schemas (patterns detected from repeated experience).",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "engram_trace",
                "description": "Full lineage trace for a memory: source events → appraisal → modifications.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string", "description": "Memory ID to trace"},
                    },
                    "required": ["memory_id"],
                },
            },
        ]

    def call_tool(self, name: str, arguments: dict) -> dict:
        """Call an MCP tool by name."""
        handler = self._tools.get(name)
        if handler is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            return handler(arguments)
        except Exception as e:
            logger.error("tool %s failed: %s", name, e, exc_info=True)
            return {"error": str(e)}

    # --- Tool Implementations ---

    def tool_remember(self, args: dict) -> dict:
        content = args["content"]
        salience = args.get("salience", 0.0)
        memory_type = MemoryType(args.get("type", "fact"))
        event_id = self.engram.remember(content, salience=salience, memory_type=memory_type)
        return {"event_id": event_id, "status": "remembered"}

    def tool_recall(self, args: dict) -> dict:
        query = args["query"]
        limit = args.get("limit", 5)
        depth = args.get("depth", "L1")
        include_faded = args.get("include_faded", False)

        results = self.engram.recall(
            query, limit=limit, depth=depth, include_faded=include_faded,
        )
        return {
            "results": [
                {
                    "id": r.memory.id,
                    "type": r.memory.type.value,
                    "content": r.memory.content,
                    "summary": r.memory.summary,
                    "score": round(r.score, 3),
                    "salience": round(r.memory.salience, 3),
                    "state": r.memory.state.value,
                    "sources": r.sources,
                }
                for r in results
            ],
            "count": len(results),
        }

    def tool_forget(self, args: dict) -> dict:
        count = self.engram.forget(
            id=args.get("id"),
            query=args.get("query"),
            hard=args.get("hard", False),
        )
        return {"affected": count, "action": "deleted" if args.get("hard") else "suppressed"}

    def tool_capture(self, args: dict) -> dict:
        event_id = self.engram.capture(args["content"])
        return {"event_id": event_id, "status": "captured"}

    def tool_consolidate(self, args: dict) -> dict:
        report = self.engram.consolidate()
        return {
            "events_processed": report.events_processed,
            "memories_created": report.memories_created,
            "facts_extracted": report.facts_extracted,
            "duration_ms": report.duration_ms,
            "errors": report.errors,
        }

    def tool_pin(self, args: dict) -> dict:
        pin_id = self.engram.pin(args["content"])
        return {"pin_id": pin_id, "status": "pinned"}

    def tool_unpin(self, args: dict) -> dict:
        removed = self.engram.unpin(args["pin_id"])
        return {"removed": removed}

    def tool_context(self, args: dict) -> dict:
        max_tokens = args.get("max_tokens", 4096)
        ctx = self.engram.active_context(max_tokens=max_tokens)
        return {"context": ctx}

    def tool_status(self, args: dict) -> dict:
        return self.engram.status()

    def tool_affect(self, args: dict) -> dict:
        trigger = args.get("trigger")
        if trigger:
            intensity = args.get("intensity", 0.5)
            self.engram.affect.trigger(trigger, intensity)
        return self.engram.affect.status()

    def tool_schemas(self, args: dict) -> dict:
        return {"schemas": self.engram.schemas()}

    def tool_trace(self, args: dict) -> dict:
        memory_id = args.get("memory_id", "")
        result = self.engram.trace(memory_id)
        if result is None:
            return {"error": f"Memory {memory_id} not found"}
        return result

    # --- MCP Stdio Transport ---

    def serve_stdio(self) -> None:
        """Run as an MCP server on stdin/stdout (JSON-RPC 2.0).

        Follows MCP handshake: client sends initialize → server responds →
        client sends notifications/initialized → server is ready for tool calls.
        """
        import sys

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                self._write_json({
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": "Parse error"},
                    "id": None,
                })
                continue

            method = request.get("method", "")
            req_id = request.get("id")
            params = request.get("params", {})

            if method == "initialize":
                self._write_json({
                    "jsonrpc": "2.0",
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": "engram", "version": __version__},
                    },
                    "id": req_id,
                })
            elif method == "tools/list":
                self._write_json({
                    "jsonrpc": "2.0",
                    "result": {"tools": self.list_tools()},
                    "id": req_id,
                })
            elif method == "tools/call":
                tool_name = params.get("name", "")
                tool_args = params.get("arguments", {})
                try:
                    result = self.call_tool(tool_name, tool_args)
                    self._write_json({
                        "jsonrpc": "2.0",
                        "result": {
                            "content": [{"type": "text", "text": json.dumps(result, default=str)}],
                        },
                        "id": req_id,
                    })
                except Exception as e:
                    self._write_json({
                        "jsonrpc": "2.0",
                        "error": {"code": -32000, "message": str(e)},
                        "id": req_id,
                    })
            elif method == "notifications/initialized":
                pass  # client ack, no response needed
            else:
                if req_id is not None:
                    self._write_json({
                        "jsonrpc": "2.0",
                        "error": {"code": -32601, "message": f"Method not found: {method}"},
                        "id": req_id,
                    })

    def _write_json(self, obj: dict) -> None:
        """Write JSON-RPC message to stdout."""
        import sys
        sys.stdout.write(json.dumps(obj, default=str) + "\n")
        sys.stdout.flush()
