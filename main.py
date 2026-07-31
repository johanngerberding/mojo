import json
import os
import subprocess
from pathlib import Path
from pprint import pprint

from openai import OpenAI
from openai.types.responses import ResponseOutputItem

SYSTEM = f"You are a coding agent at {os.getcwd}. Use bash to solve tasks. Act, don't explain."
MODEL = "unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q2_K_XL"
WORKDIR = Path.cwd()

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


TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
}

DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=", "> /dev/sda"]


def check_deny_list(command: str) -> str | None:
    for pattern in DENY_LIST:
        if pattern in command:
            return f"Blocked '{pattern}' is on the deny list"
    return None


PERMISSION_RULES = [
    {
        "tools": ["read_file", "write_file", "edit_file"],
        "check": lambda args: (
            not (WORKDIR / args.get("path", "")).resolve().is_relative_to(WORKDIR)
        ),
        "message": "Access outside workspace",
    },
    {
        "tools": ["bash"],
        "check": lambda args: any(
            kw in args.get("command", "") for kw in ["rm ", "> /etc/", "chmod 777"]
        ),
        "message": "Potentially destructive command",
    },
]


def check_rules(tool_name: str, args: dict) -> str | None:
    for rule in PERMISSION_RULES:
        if tool_name in rule["tools"] and rule["check"](args):
            return rule["message"]
    return None


def ask_user(tool_name: str, args: dict, reason: str) -> str:
    print(f"\n  {reason}")
    print(f"  Tool: {tool_name}({args})")
    choice = input("   Allow? [y/N] ").strip().lower()
    return "allow" if choice in ("y", "yes") else "deny"


def check_permission(item: ResponseOutputItem) -> bool:
    if item.name == "bash":
        cmd = json.loads(item.arguments).get("command", "")
        reason = check_deny_list(cmd)
        if reason:
            print(f"\n{reason}")
            return False
    reason = check_rules(item.name, json.loads(item.arguments))
    if reason:
        decision = ask_user(item.name, json.loads(item.arguments), reason)
        if decision == "deny":
            return False
    return True


def agent_loop(client: OpenAI, messages: list[dict]):
    while True:
        response = client.responses.create(
            model=MODEL,
            input=messages,
            tools=TOOLS,
            max_output_tokens=8192,
        )

        messages += [item.model_dump() for item in response.output]
        pprint(f"Messages:\n{messages}")

        tool_calls: list[ResponseOutputItem] = [
            item for item in response.output if item.type == "function_call"
        ]
        if not tool_calls:
            return

        for item in tool_calls:
            if item.type != "function_call":
                continue
            else:
                if not check_permission(item):
                    messages.append(
                        {
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": "Permission denied.",
                        }
                    )
                    continue

                print(f"Using tool called: {item.name}")
                handler = TOOL_HANDLERS[item.name]
                cmd = json.loads(item.arguments)
                print(f"Command: {cmd}")
                output = handler(**cmd)
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
