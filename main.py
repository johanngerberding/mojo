import json
import os
import subprocess

from openai import OpenAI

SYSTEM = f"You are a coding agent at {os.getcwd}. Use bash to solve tasks. Act, don't explain."
MODEL = "unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q2_K_XL"

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
    }
]


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


def agent_loop(client: OpenAI, messages: list[dict]):
    while True:
        response = client.responses.create(
            model=MODEL,
            input=messages,
            tools=TOOLS,
            max_output_tokens=8192,
        )
        messages += response.output

        for item in response.output:
            if item.type == "function_call" and item.name == "bash":
                cmd = json.loads(item.arguments)["command"]
                output = run_bash(cmd)
                messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": output,
                    }
                )
                return


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
        response_content = history[-1]["output"]
        print(response_content)


if __name__ == "__main__":
    main()
