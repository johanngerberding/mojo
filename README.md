# mojo  

This is mojo, a simple coding harness (for now). This is a fun project to explore how current agentic harnesses like claude code work behind the scenes. I will add capabilities in the following days working on this. 

This is based on the following project: [learn-claude-code](https://github.com/shareAI-lab/learn-claude-code/tree/main). Because of that, we start with a coding harness, but we will eventually have some fun of our own. 

I will make a couple of small changes like using llama.cpp as the backend for serving the model because I like open source models that I can run locally. Moreover I changed to the OpenAI client for compatibility reasons.

Btw this is a coding harness free repo, it's humanly typed code, believe it or not.

## model 

For now, Qwen3.6-27B-MTP-GGUF 2-bit quantized is used as our reasoning model in the background (the best that fits onto my 5060 16GB). I use llama.cpp to run it on my Linux machine, but all the other frameworks like vLLM or SGLang or whatever that gives you an OpenAI compatible API should work as well.

```
export LLAMA_CACHE="unsloth/Qwen3.6-27B-MTP-GGUF"
./llama.cpp/llama-server -hf unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q2_K_XL -ngl 99 -c 8192 -fa on -np 1 --spec-type draft-mtp --spec-draft-n-max 2 --jinja
```