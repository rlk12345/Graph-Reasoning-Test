import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# use a Hugging Face model that supports safetensors
model_id = "mistralai/Mistral-7B-Instruct-v0.2"

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

print("Loading model... (first run downloads weights ~13GB)")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    use_safetensors=True
)

template = (
    "Below is an instruction that describes a task.\n"
    "Write a response that appropriately completes the request.\n"
    "### Instruction:\n{q}\n\n### Response:"
)

q = (
    "Find the shortest path between two nodes in an undirected graph. "
    "In an undirected graph, (i,j,k) means nodes i and j are connected with weight k. "
    "Given edges (0,1,4) (1,2,7) (1,3,4) (2,6,2) (2,4,8) (2,7,5) "
    "(3,6,1) (4,8,3) (5,6,6) (6,8,8) (7,8,7), give the weight of the shortest path from 0 to 5."
)

prompt = template.format(q=q)
inputs = tok(prompt, return_tensors="pt").to(model.device)

print("Generating...")
streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
_ = model.generate(**inputs, max_new_tokens=128, do_sample=False, streamer=streamer)
print("\n✅ Done")
