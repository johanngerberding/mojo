import ast
import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from pprint import pprint

from openai import OpenAI
from openai.types.responses import ResponseOutputItem

WORKDIR = Path.cwd()
MODEL = "unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q2_K_XL"
CURRENT_TODOS: list[dict] = []

SYSTEM = (
    f"You are a coding agent at {WORKDIR}."
    "Before starting any multi-step task, use todo_write to plan your steps. "
    "Update the status as you go."
)

# openai way of tool calling
TOOLS = [
    {
        "type": "function",
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "shell command"}
            },
            "required": ["command"],
        },
    },
    {
        "type": "function",
        "name": "read_file",
        "description": "Read file contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "path to file"},
                "limit": {
                    "type": ["integer", "null"],
                    "description": "maximum number of lines to read from the file",
                },
            },
            "required": ["path"],
        },
    },
    {
        "type": "function",
        "name": "write_file",
        "description": "Write content to file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "path where to write the file to",
                },
                "content": {
                    "type": "string",
                    "description": "content thats written to file",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "type": "function",
        "name": "edit_file",
        "description": "Replace text in file once.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": ""},
                "old_text": {"type": "string", "description": ""},
                "new_text": {"type": "string", "description": ""},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "type": "function",
        "name": "glob",
        "description": "Find files by pattern.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Pattern to match files to.",
                }
            },
            "required": ["pattern"],
        },
    },
    {
        "type": "custom",
        "name": "todo_write",
        "description": "Create and manage a task list for your current coding session.",
        "parameters": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                        },
                    },
                }
            },
        },
    },
]


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes the workspace: {p}")
    return path


def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except FileNotFoundError:
        return "Error: file not found"
    except PermissionError:
        return "Error: permission denied"
    except OSError as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except PermissionError:
        return "Error: permission denied"
    except OSError as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(path)
        text = file_path.read_text()
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except FileNotFoundError:
        return "Error: file not found"
    except PermissionError:
        return "Error: permission denied"
    except OSError as e:
        return f"Error: {e}"


def run_glob(pattern: str) -> str:
    import glob as g

    try:
        results = []
        for match in g.glob(pattern, root_dir=WORKDIR):
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except OSError as e:
        return f"Error: {e}"


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked!"
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "no output"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (60s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def _normalize_todos(todos) -> tuple:
    if isinstance(todos, str):
        try:
            todos = json.loads(todos)
        except json.JSONDecodeError:
            try:
                todos = ast.literal_eval(todos)
            except (SyntaxError, ValueError):
                return None, "Error: todos must be a list or JSON array string"
    if not isinstance(todos, list):
        return None, "Error: todos must be a list"
    for i, t in enumerate(todos):
        if not isinstance(t, dict):
            return None, f"Error: todos[{i}] must be an object"
        if "content" not in t or "status" not in t:
            return None, f"Error: todos[{i}] missing 'content' or 'status'"
        if t["status"] not in ("pending", "in_progress", "completed"):
            return None, f"Error: todos[{i}] has invalid status '{t['status']}'"
    return todos, None


def run_todo_write(todos: list) -> str:
    global CURRENT_TODOS
    todos, error = _normalize_todos(todos)
    if error:
        return error

    CURRENT_TODOS = todos

    lines = ["\n##Current Tasks"]
    for t in CURRENT_TODOS:
        icon = {"pending": " ", "in_progress": "▸", "completed": "✓"}[t["status"]]
        lines.append(f"  [{icon}] {t['content']}")
    print("\n".join(lines))
    return f"Updated {len(CURRENT_TODOS)} tasks"


TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
    "todo_write": run_todo_write,
}

HOOKS: dict[str, list[Callable]] = {
    "UserPromptSubmit": [],
    "PreToolUse": [],
    "PostToolUse": [],
    "Stop": [],
}


def register_hook(event: str, callback: Callable):
    HOOKS[event].append(callback)


def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None


DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=", "> /dev/sda"]
DESTRUCTIVE = ["rm ", "> /etc/", "chmod 777"]


def permission_hook(item: ResponseOutputItem) -> str | None:
    if item.name == "bash":
        for pattern in DENY_LIST:
            if pattern in json.loads(item.arguments).get("command", ""):
                print(f"Blocked: '{pattern}'")
                return "Permission denied by deny list"
        for kw in DESTRUCTIVE:
            if kw in json.loads(item.arguments).get("command", ""):
                print("Potentially destructive command")
                print(f"   Tool: {item.name} ({json.loads(item.arguments)})")
                choice = input("  Allow? [y/N] ").strip().lower()
                if choice not in ("y", "yes"):
                    return "Permission denied by user"
    if item.name in ("read_file", "write_file", "edit_file"):
        path = json.loads(item.arguments).get("path", "")
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            print("Access outside workspace")
            print(f"   Tool: {item.name} ({json.loads(item.arguments)})")
            choice = input("  Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"
    return None


def log_hook(item: ResponseOutputItem):
    """PreToolUse: log every tool call."""
    args_preview = str(list(json.loads(item.arguments).values())[:2])[:60]
    print(f"[HOOK] {item.name}({args_preview})")


def large_output_hook(item: ResponseOutputItem, output: str):
    """PostToolUse: warn on large output"""
    if len(str(output)) > 100_000:
        print(f"[HOOK] Large output from {item.name}: {len(str(output))} chars")


def context_inject_hook():
    print(f"[HOOK] UserPromptSubmit: working in {WORKDIR}")


def summary_hook(messages: list):
    tool_count = sum(
        1
        for m in messages
        for b in (m.get("content") if isinstance(m.get("content"), list) else [])
        if isinstance(b, dict) and b.get("type") == "function_call_output"
    )
    print(f"[HOOK] Stop: session used {tool_count} tool calls.")


register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)


def agent_loop(client: OpenAI, messages: list[dict]):
    rounds_since_todo = 0
    while True:
        if rounds_since_todo >= 3 and messages:
            messages.append(
                {"role": "user", "content": "<reminder>Update your todos.</reminder>"}
            )
            rounds_since_todo = 0
        response = client.responses.create(
            model=MODEL,
            input=messages,
            tools=TOOLS,
            max_output_tokens=8192,
        )

        messages += [item.model_dump() for item in response.output]

        tool_calls: list[ResponseOutputItem] = [
            item for item in response.output if item.type == "function_call"
        ]
        if not tool_calls:
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            return

        rounds_since_todo += 1
        for item in tool_calls:
            if item.type != "function_call":
                continue
            blocked = trigger_hooks("PreToolUse", item)
            if blocked:
                messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": str(blocked),
                    }
                )
                continue

            print(f"Using tool called: {item.name}")
            handler = TOOL_HANDLERS[item.name]
            cmd = json.loads(item.arguments)
            print(f"Command: {cmd}")
            output = handler(**cmd)

            trigger_hooks("PostToolUse", item, output)

            messages.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": output,
                }
            )


def main():
    client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="test")
    history = [{"role": "developer", "content": SYSTEM}]

    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({"role": "user", "content": query})
        agent_loop(client, history)
        # response_content = history[-1].content[0].text
        # print(response_content)
        pprint(history)


if __name__ == "__main__":
    main()
