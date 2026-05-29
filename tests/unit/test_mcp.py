"""Tests for the MCP server."""

import pytest

from engram.core.config import Config
from engram.engine import Engram
from engram.mcp.server import MCPServer


@pytest.fixture
def server(tmp_path):
    config = Config.minimal(str(tmp_path / "test"))
    engine = Engram(config, actor="mcp-test")
    srv = MCPServer(engine)
    yield srv
    engine.close()


class TestListTools:
    def test_returns_all_tools(self, server):
        tools = server.list_tools()
        names = {t["name"] for t in tools}
        assert "engram_remember" in names
        assert "engram_recall" in names
        assert "engram_forget" in names
        assert "engram_capture" in names
        assert "engram_consolidate" in names
        assert "engram_pin" in names
        assert "engram_unpin" in names
        assert "engram_context" in names
        assert "engram_status" in names

    def test_tools_have_schema(self, server):
        tools = server.list_tools()
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool


class TestRememberRecall:
    def test_remember_and_recall(self, server):
        result = server.call_tool("engram_remember", {"content": "User prefers dark mode"})
        assert result["status"] == "remembered"

        result = server.call_tool("engram_recall", {"query": "dark mode"})
        assert result["count"] >= 1
        assert "dark mode" in result["results"][0]["content"]

    def test_remember_with_type(self, server):
        result = server.call_tool("engram_remember", {
            "content": "Deploy happened at 3pm",
            "type": "episode",
            "salience": 0.8,
        })
        assert result["status"] == "remembered"


class TestForget:
    def test_forget_by_query(self, server):
        server.call_tool("engram_remember", {"content": "secret password"})
        result = server.call_tool("engram_forget", {"query": "secret password"})
        assert result["affected"] >= 1
        assert result["action"] == "suppressed"

    def test_hard_forget(self, server):
        server.call_tool("engram_remember", {"content": "gdpr data"})
        recall = server.call_tool("engram_recall", {"query": "gdpr data"})
        if recall["results"]:
            mid = recall["results"][0]["id"]
            result = server.call_tool("engram_forget", {"id": mid, "hard": True})
            assert result["action"] == "deleted"


class TestCapture:
    def test_capture(self, server):
        result = server.call_tool("engram_capture", {"content": "user clicked button"})
        assert result["status"] == "captured"


class TestConsolidate:
    def test_consolidate(self, server):
        server.call_tool("engram_capture", {"content": "event 1"})
        server.call_tool("engram_capture", {"content": "event 2"})
        result = server.call_tool("engram_consolidate", {})
        assert "events_processed" in result
        assert result["errors"] == []


class TestPins:
    def test_pin_and_context(self, server):
        result = server.call_tool("engram_pin", {"content": "Always verify before deploy"})
        assert "pin_id" in result

        ctx = server.call_tool("engram_context", {})
        assert "verify before deploy" in ctx["context"]

    def test_unpin(self, server):
        pin = server.call_tool("engram_pin", {"content": "temp note"})
        result = server.call_tool("engram_unpin", {"pin_id": pin["pin_id"]})
        assert result["removed"] is True


class TestStatus:
    def test_status(self, server):
        result = server.call_tool("engram_status", {})
        assert "total_memories" in result
        assert "buffer_events" in result


class TestErrorHandling:
    def test_unknown_tool(self, server):
        result = server.call_tool("nonexistent_tool", {})
        assert "error" in result

    def test_missing_required_field(self, server):
        result = server.call_tool("engram_remember", {})
        assert "error" in result
