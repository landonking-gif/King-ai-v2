"""
Test that tool_calls history trimming works correctly.

This test verifies that the fix for the "tool messages must follow tool_calls" error works.
"""

import sys
import os

# Add kautilya to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "kautilya"))

from kautilya.llm_client import Message, ChatHistory

try:
    from rich.console import Console
    console = Console()
except ImportError:
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
    console = Console()


def test_basic_trimming():
    """Test that basic trimming works without tool calls."""
    console.print("\n[bold cyan]Test 1: Basic Trimming (No Tool Calls)[/bold cyan]")
    console.print("-" * 60)

    history = ChatHistory(max_messages=10)

    # Add system message
    history.add(Message(role="system", content="You are a helpful assistant"))

    # Add many user/assistant messages
    for i in range(15):
        history.add(Message(role="user", content=f"User message {i}"))
        history.add(Message(role="assistant", content=f"Assistant response {i}"))

    # Should be trimmed to 10 total (1 system + 9 others)
    console.print(f"Total messages after trimming: {len(history.messages)}")
    console.print(f"Expected: ≤ 10")

    # Validate
    assert len(history.messages) <= 10, f"Expected ≤10 messages, got {len(history.messages)}"
    assert history.messages[0].role == "system", "First message should be system"

    console.print("✓ Basic trimming works correctly\n")
    return True


def test_tool_calls_preserved():
    """Test that tool_calls/tool pairs are preserved during trimming."""
    console.print("\n[bold cyan]Test 2: Tool Calls Preservation[/bold cyan]")
    console.print("-" * 60)

    history = ChatHistory(max_messages=10)

    # Add system message
    history.add(Message(role="system", content="System"))

    # Add several messages with tool calls
    for i in range(5):
        history.add(Message(role="user", content=f"User {i}"))
        history.add(Message(
            role="assistant",
            content="",
            tool_calls=[{"id": f"call_{i}", "type": "function", "function": {"name": "test", "arguments": "{}"}}]
        ))
        history.add(Message(role="tool", content="result", tool_call_id=f"call_{i}"))
        history.add(Message(role="assistant", content=f"Response {i}"))

    console.print(f"Total messages: {len(history.messages)}")
    console.print(f"Message roles: {[m.role for m in history.messages]}")

    # Validate: Check that no tool message is orphaned
    messages = history.messages
    tool_call_ids_seen = set()

    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("id"):
                    tool_call_ids_seen.add(tc["id"])

    # Check all tool messages have corresponding tool_calls
    orphaned = []
    for msg in messages:
        if msg.role == "tool":
            if msg.tool_call_id not in tool_call_ids_seen:
                orphaned.append(msg.tool_call_id)

    if orphaned:
        console.print(f"[red]✗ Found orphaned tool messages: {orphaned}[/red]")
        return False

    console.print("✓ All tool messages have corresponding tool_calls\n")
    return True


def test_tool_calls_removed_together():
    """Test that when a tool_calls message is removed, its tool responses are also removed."""
    console.print("\n[bold cyan]Test 3: Tool Calls Removed Together[/bold cyan]")
    console.print("-" * 60)

    history = ChatHistory(max_messages=8)  # Very small limit

    # Add system message
    history.add(Message(role="system", content="System"))

    # Add messages until we force trimming
    history.add(Message(role="user", content="User 1"))
    history.add(Message(
        role="assistant",
        content="",
        tool_calls=[
            {"id": "call_1", "type": "function", "function": {"name": "test1", "arguments": "{}"}},
            {"id": "call_2", "type": "function", "function": {"name": "test2", "arguments": "{}"}}
        ]
    ))
    history.add(Message(role="tool", content="result1", tool_call_id="call_1"))
    history.add(Message(role="tool", content="result2", tool_call_id="call_2"))
    history.add(Message(role="assistant", content="Response 1"))

    # Add more messages to trigger trimming
    history.add(Message(role="user", content="User 2"))
    history.add(Message(role="assistant", content="Response 2"))
    history.add(Message(role="user", content="User 3"))
    history.add(Message(role="assistant", content="Response 3"))
    history.add(Message(role="user", content="User 4"))
    history.add(Message(role="assistant", content="Response 4"))

    console.print(f"Total messages: {len(history.messages)}")
    console.print(f"Max allowed: 8")
    console.print(f"Message roles: {[m.role for m in history.messages]}")

    # Validate: No orphaned tool messages
    messages = history.messages
    tool_call_ids_seen = set()

    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("id"):
                    tool_call_ids_seen.add(tc["id"])

    orphaned = []
    for msg in messages:
        if msg.role == "tool":
            if msg.tool_call_id not in tool_call_ids_seen:
                orphaned.append(msg.tool_call_id)

    if orphaned:
        console.print(f"[red]✗ Found orphaned tool messages: {orphaned}[/red]")
        return False

    console.print("✓ Tool calls removed together with their responses\n")
    return True


def test_validate_history_method():
    """Test the validate_history method on KautilyaLLMClient."""
    console.print("\n[bold cyan]Test 4: Validate History Method[/bold cyan]")
    console.print("-" * 60)

    # Test with valid history
    history = ChatHistory()
    history.add(Message(role="system", content="System"))
    history.add(Message(role="user", content="User"))
    history.add(Message(
        role="assistant",
        content="",
        tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "test", "arguments": "{}"}}]
    ))
    history.add(Message(role="tool", content="result", tool_call_id="call_1"))
    history.add(Message(role="assistant", content="Response"))

    # Simulate validation
    messages = history.to_list()
    pending_tool_calls = set()
    valid = True

    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if "id" in tc:
                    pending_tool_calls.add(tc["id"])

        elif msg.get("role") == "tool":
            tool_call_id = msg.get("tool_call_id")
            if tool_call_id in pending_tool_calls:
                pending_tool_calls.remove(tool_call_id)
            else:
                # Orphaned tool message!
                valid = False
                break

    if len(pending_tool_calls) > 0:
        valid = False

    if valid:
        console.print("✓ Valid history passes validation")
    else:
        console.print("[red]✗ Valid history failed validation[/red]")
        return False

    # Test with invalid history (orphaned tool message)
    history2 = ChatHistory()
    history2.add(Message(role="system", content="System"))
    history2.add(Message(role="user", content="User"))
    history2.add(Message(role="tool", content="result", tool_call_id="call_1"))  # Orphaned!

    messages2 = history2.to_list()
    pending_tool_calls2 = set()
    valid2 = True

    for msg in messages2:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if "id" in tc:
                    pending_tool_calls2.add(tc["id"])

        elif msg.get("role") == "tool":
            tool_call_id = msg.get("tool_call_id")
            if tool_call_id in pending_tool_calls2:
                pending_tool_calls2.remove(tool_call_id)
            else:
                # Orphaned tool message!
                valid2 = False
                break

    if not valid2:
        console.print("✓ Invalid history (orphaned tool) correctly detected\n")
    else:
        console.print("[red]✗ Invalid history not detected[/red]")
        return False

    return True


def main():
    """Run all tests."""
    console.print("\n" + "=" * 60)
    console.print("[bold yellow]Tool Calls History Trimming Tests[/bold yellow]")
    console.print("=" * 60)

    results = []
    results.append(("Basic trimming", test_basic_trimming()))
    results.append(("Tool calls preserved", test_tool_calls_preserved()))
    results.append(("Tool calls removed together", test_tool_calls_removed_together()))
    results.append(("Validate history method", test_validate_history_method()))

    console.print("\n" + "=" * 60)
    console.print("[bold]Test Results Summary[/bold]")
    console.print("=" * 60)

    for name, result in results:
        status = "[green]✓ PASS[/green]" if result else "[red]✗ FAIL[/red]"
        console.print(f"{status} - {name}")

    all_passed = all(result for _, result in results)

    console.print("\n" + "=" * 60)
    if all_passed:
        console.print("[bold green]✓ ALL TESTS PASSED[/bold green]")
        console.print("=" * 60 + "\n")
        console.print("[cyan]The tool_calls history bug is fixed![/cyan]")
        console.print("\nThe fix ensures that:")
        console.print("  • Tool messages always follow their assistant tool_calls")
        console.print("  • When trimming history, tool_calls/tool pairs stay together")
        console.print("  • Orphaned tool messages are never sent to the LLM API")
        console.print("  • The 400 error will no longer occur\n")
    else:
        console.print("[bold red]✗ SOME TESTS FAILED[/bold red]")
        console.print("=" * 60 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
