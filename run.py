import json

from vllm import LLM, SamplingParams
import time


sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="/root/autodl-tmp/meta-llama_Meta-Llama-3-8B-Instruct/")
start_time = time.time()

with open('./datalong.json') as f:
    data = json.load(f)
prompts = data['prompts']
outputs = llm.generate(prompts, sampling_params)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time} seconds")
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
