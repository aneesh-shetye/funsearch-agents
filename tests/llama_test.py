
import os 
import transformers
import torch
import requests

from vllm import LLM
from vllm.sampling_params import SamplingParams

from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


model_name = "mistralai/Pixtral-12B-2409"
#token = os.getenv("HUGGINGFACE_HUB_TOKEN")
cache_dir = "/scratch/axs10302/datasets"

'''
pipeline = transformers.pipeline(
  "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto", cache_dir=cache_dir)

print(pipeline("Hey how are you doing today?"))
'''

sampling_params = SamplingParams(max_tokens=8192)

llm = LLM(model=model_name, tokenizer_mode="mistral")

prompt = "Describe this image in one sentence."



'''
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
    #cache_dir=cache_dir
)
processor = AutoProcessor.from_pretrained(model_id) 
'''

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
image = Image.open(requests.get(url, stream=True).raw)

'''
prompt = "<|image|><|begin_of_text|>If I had to write a haiku for this one"
inputs = processor(image, prompt, return_tensors="pt").to(model.device)

output = model.generate(**inputs)#, max_new_tokens=30)

print(processor.decode(output[0]))
'''

messages = [
 {
        "role": "user",
        "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url":url}                   }]
    },

]


outputs = llm.chat(messages, sampling_params=sampling_params)

print(outputs)
print(outputs[0].outputs[0].text) 




