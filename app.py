import cachetools
from pydantic import BaseModel
from llama_cpp import Llama
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from threading import Thread
import psutil
import gc
import torch
import numpy as np
from PIL import Image
import stable_diffusion_cpp as sdcpp
import base64
import io
import time
from typing import AsyncGenerator

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

cache = cachetools.TTLCache(maxsize=100, ttl=60)

global_data = {
    'models': {},
    'tokensxx': {
        'eos': '<|end_of-text|>',
        'pad': '<pad>',
        'unk': '<unk>',
        'bos': '<|begin_of_text|>',
        'sep': '<|sep|>',
        'cls': '<|cls|>',
        'mask': '<mask>',
        'eot': '<|eot_id|>',
        'eom': '<|eom_id|>',
        'lf': '<|0x0A|>'
    },
    'tokens': {
        'eos': 'eos_token',
        'pad': 'pad_token',
        'unk': 'unk_token',
        'bos': 'bos_token',
        'sep': 'sep_token',
        'cls': 'cls_token',
        'mask': 'mask_token'
    },
    'model_metadata': {},
    'eos': {},
    'pad': {},
    'padding': {},
    'unk': {},
    'bos': {},
    'sep': {},
    'cls': {},
    'mask': {},
    'eot': {},
    'eom': {},
    'lf': {},
    'max_tokens': {},
    'tokenizers': {},
    'model_params': {},
    'model_size': {},
    'model_ftype': {},
    'n_ctx_train': {},
    'n_embd': {},
    'n_layer': {},
    'n_head': {},
    'n_head_kv': {},
    'n_rot': {},
    'n_swa': {},
    'n_embd_head_k': {},
    'n_embd_head_v': {},
    'n_gqa': {},
    'n_embd_k_gqa': {},
    'n_embd_v_gqa': {},
    'f_norm_eps': {},
    'f_norm_rms_eps': {},
    'f_clamp_kqv': {},
    'f_max_alibi_bias': {},
    'f_logit_scale': {},
    'n_ff': {},
    'n_expert': {},
    'n_expert_used': {},
    'causal_attn': {},
    'pooling_type': {},
    'rope_type': {},
    'rope_scaling': {},
    'freq_base_train': {},
    'freq_scale_train': {},
    'n_ctx_orig_yarn': {},
    'rope_finetuned': {},
    'ssm_d_conv': {},
    'ssm_d_inner': {},
    'ssm_d_state': {},
    'ssm_dt_rank': {},
    'ssm_dt_b_c_rms': {},
    'vocab_type': {},
    'model_type': {},
    "general.architecture": {},
    "general.type": {},
    "general.name": {},
    "general.finetune": {},
    "general.basename": {},
    "general.size_label": {},
    "general.license": {},
    "general.license.link": {},
    "general.tags": {},
    "general.languages": {},
    "general.organization": {},
    "general.base_model.count": {},
    'general.file_type': {},
    "phi3.context_length": {},
    "phi3.rope.scaling.original_context_length": {},
    "phi3.embedding_length": {},
    "phi3.feed_forward_length": {},
    "phi3.block_count": {},
    "phi3.attention.head_count": {},
    "phi3.attention.head_count_kv": {},
    "phi3.attention.layer_norm_rms_epsilon": {},
    "phi3.rope.dimension_count": {},
    "phi3.rope.freq_base": {},
    "phi3.attention.sliding_window": {},
    "phi3.rope.scaling.attn_factor": {},
    "llama.block_count": {},
    "llama.context_length": {},
    "llama.embedding_length": {},
    "llama.feed_forward_length": {},
    "llama.attention.head_count": {},
    "llama.attention.head_count_kv": {},
    "llama.rope.freq_base": {},
    "llama.attention.layer_norm_rms_epsilon": {},
    "llama.attention.key_length": {},
    "llama.attention.value_length": {},
    "llama.vocab_size": {},
    "llama.rope.dimension_count": {},
    "deepseek2.block_count": {},
    "deepseek2.context_length": {},
    "deepseek2.embedding_length": {},
    "deepseek2.feed_forward_length": {},
    "deepseek2.attention.head_count": {},
    "deepseek2.attention.head_count_kv": {},
    "deepseek2.rope.freq_base": {},
    "deepseek2.attention.layer_norm_rms_epsilon": {},
    "deepseek2.expert_used_count": {},
    "deepseek2.leading_dense_block_count": {},
    "deepseek2.vocab_size": {},
    "deepseek2.attention.kv_lora_rank": {},
    "deepseek2.attention.key_length": {},
    "deepseek2.attention.value_length": {},
    "deepseek2.expert_feed_forward_length": {},
    "deepseek2.expert_count": {},
    "deepseek2.expert_shared_count": {},
    "deepseek2.expert_weights_scale": {},
    "deepseek2.rope.dimension_count": {},
    "deepseek2.rope.scaling.type": {},
    "deepseek2.rope.scaling.factor": {},
    "deepseek2.rope.scaling.yarn_log_multiplier": {},
    "qwen2.block_count": {},
    "qwen2.context_length": {},
    "qwen2.embedding_length": {},
    "qwen2.feed_forward_length": {},
    "qwen2.attention.head_count": {},
    "qwen2.attention.head_count_kv": {},
    "qwen2.rope.freq_base": {},
    "qwen2.attention.layer_norm_rms_epsilon": {},
    "general.version": {},
    "general.datasets": {},
    "tokenizer.ggml.model": {},
    "tokenizer.ggml.pre": {},
    "tokenizer.ggml.tokens": {},
    "tokenizer.ggml.token_type": {},
    "tokenizer.ggml.merges": {},
    "tokenizer.ggml.bos_token_id": {},
    "tokenizer.ggml.eos_token_id": {},
    "tokenizer.ggml.unknown_token_id": {},
    "tokenizer.ggml.padding_token_id": {},
    "tokenizer.ggml.add_bos_token": {},
    "tokenizer.ggml.add_eos_token": {},
    "tokenizer.ggml.add_space_prefix": {},
    "tokenizer.chat_template": {},
    "quantize.imatrix.file": {},
    "quantize.imatrix.dataset": {},
    "quantize.imatrix.entries_count": {},
    "quantize.imatrix.chunks_count": {},
    "general.quantization_version": {},
    'n_lora_q': {},
    'n_lora_kv': {},
    'n_expert_shared': {},
    'n_ff_exp': {},
    "n_layer_dense_lead": {},
    "expert_weights_scale": {},
    "rope_yarn_log_mul": {},
    'eval': {},
    'time': {},
    'token': {},
    'tokens': {},
    'pads': {},
    'model': {},
    'base': {},
    'model_base': {},
    'perhaps': {},
    'word': {},
    'words': {},
    'start': {},
    'stop': {},
    'run': {},
    'runs': {},
    'ms': {},
    'vocabulary': {},
    'timeout': {},
    'load': {},
    'load_time': {},
    'bas': {},
    'tok': {},
    'second': {},
    'seconds': {},
    'graph': {},
    'load_model': {},
    'end': {},
    'llama_perf_context_print': {},
    'llm_load_print_meta': {},
    'model_type': {},
    'image_model': {}
}


model_configs = [
    {
        "repo_id": "Hjgugugjhuhjggg/testing_semifinal-Q2_K-GGUF",
        "filename": "testing_semifinal-q2_k.gguf",
        "name": "testing"
    },
    {
        "repo_id": "bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF",
        "filename": "Llama-3.2-3B-Instruct-uncensored-Q2_K.gguf",
        "name": "Llama-3.2-3B-Instruct"
    },
     {
        "repo_id": "city96/FLUX.1-schnell-gguf",
        "filename": "flux1-schnell-Q2_K.gguf",
        "name": "flux1-schnell"
     },
    
]

class ModelManager:
    def __init__(self):
        self.models = {}
        self.image_model = None

    def load_model(self, model_config):
        if model_config['name'] not in self.models and model_config['name'] != "flux1-schnell":
           try:
               print(f"Loading model: {model_config['name']}")
               self.models[model_config['name']] = Llama.from_pretrained(
                  repo_id=model_config['repo_id'],
                  filename=model_config['filename'],
                  use_auth_token=HUGGINGFACE_TOKEN,
                  n_threads=20,
                  use_gpu=False
               )
               print(f"Model loaded: {model_config['name']}")
               # Load tokenizer after model load
               if model_config['name'] not in global_data['tokenizers']:
                    global_data['tokenizers'][model_config['name']] = self.models[model_config['name']].tokenizer()
                    print(f"tokenizer loaded for: {model_config['name']}")
                    # load the eos token
                    global_data['eos'][model_config['name']] = self.models[model_config['name']].token_eos()
                    print(f"eos loaded for: {model_config['name']}")
           except Exception as e:
               print(f"Error loading model {model_config['name']}: {e}")

    def load_image_model(self, model_config):
       try:
          print(f"Attempting to load image model with config: {model_config}")
          self.image_model = sdcpp.StableDiffusionCpp(
              repo_id=model_config['repo_id'],
              filename=model_config['filename'],
              use_auth_token=HUGGINGFACE_TOKEN,
              n_threads=20,
              use_gpu=False
          )
          print(f"Image model loaded successfully: {self.image_model}")
       except Exception as e:
         print(f"Error loading image model: {e}")

    def load_all_models(self):
        with ThreadPoolExecutor() as executor:
            for config in model_configs:
                if config['name'] == "flux1-schnell":
                   executor.submit(self.load_image_model, config)
                else:
                    executor.submit(self.load_model, config)
        return self.models, self.image_model


model_manager = ModelManager()
global_data['models'], global_data['image_model'] = model_manager.load_all_models()

class ChatRequest(BaseModel):
    message: str

class ImageRequest(BaseModel):
    prompt: str

def normalize_input(input_text):
    return input_text.strip()

def remove_duplicates(text):
    lines = text.split('\n')
    unique_lines = []
    seen_lines = set()
    for line in lines:
        if line not in seen_lines:
            unique_lines.append(line)
            seen_lines.add(line)
    return '\n'.join(unique_lines)

def cache_response(func):
    def wrapper(*args, **kwargs):
        cache_key = f"{args}-{kwargs}"
        if cache_key in cache:
            return cache[cache_key]
        response = func(*args, **kwargs)
        cache[cache_key] = response
        return response
    return wrapper


@cache_response
def generate_model_response(model, inputs, max_tokens=9999999):
    try:
        response = model(inputs, max_tokens=max_tokens)
        return remove_duplicates(response['choices'][0]['text'])
    except Exception as e:
        return ""

def remove_repetitive_responses(responses):
    unique_responses = {}
    for response in responses:
        if response['model'] not in unique_responses:
            unique_responses[response['model']] = response['response']
    return unique_responses


async def process_message(message: str):
    inputs = normalize_input(message)
    
    async def stream_response(inputs: str) -> AsyncGenerator[str, None]:
            max_token_limit = 150
            full_response = ""
            current_inputs = inputs
            eos_found = False
            
            start_time = time.time()
            
            executor = ThreadPoolExecutor()
            while current_inputs and not eos_found:
                futures = [
                    executor.submit(generate_model_response, model, current_inputs, max_tokens=max_token_limit)
                    for model in global_data['models'].values()
                ]
                responses = [
                    {'model': model_name, 'response': future.result()}
                    for model_name, future in zip(global_data['models'].keys(), as_completed(futures))
                ]
                unique_responses = remove_repetitive_responses(responses)
                formatted_response = next(iter(unique_responses.values()))
                
                print(f"Generated chunk: {formatted_response}")
                
                
                #tokenize the response
                tokenizer = next(iter(global_data['tokenizers'].values()))
                tokens = tokenizer.encode(formatted_response)
                
                
                token_count = len(tokens)
                chunk_size = 30 # Set token chunk size
                for i in range(0, token_count, chunk_size):
                  chunk_tokens = tokens[i : i + chunk_size]
                  decoded_chunk = tokenizer.decode(chunk_tokens)
                  yield decoded_chunk
                
                # Check for EOS token in decoded chunk
                
                eos_token = next(iter(global_data['eos'].values()))
                if eos_token in tokens:
                   eos_found = True
                   print(f"End of sequence token found")
                   break
                
                full_response += formatted_response
                current_inputs = formatted_response if len(formatted_response.split()) > 0 else ""
            
            end_time = time.time()
            executor.shutdown(wait=True) # waits for all threads to finish
            print(f"Total time taken to process response {end_time-start_time}")
            
    return StreamingResponse(stream_response(inputs), media_type="text/plain")


async def generate_image(prompt: str):
    if global_data['image_model']:
        try:
            print("Generating image with prompt:", prompt)
            image_bytes = global_data['image_model'].generate(
                prompt=prompt,
                negative_prompt="ugly, deformed, disfigured",
                steps=25,
                cfg_scale=7.0,
                width=512,
                height=512,
                seed=-1,
                return_type='bytes'
             )
             
            image = Image.open(io.BytesIO(image_bytes))
            print("Image generated successfully.")
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode()

            return JSONResponse(content={"image": image_base64})
        except Exception as e:
           print(f"Error generating image: {e}")
           return JSONResponse(content={"error": str(e)})
    else:
         print("No image model loaded.")
         return JSONResponse(content={"error": "No image model loaded"})

def release_resources():
    try:
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"Failed to release resources: {e}")

def resource_manager():
    MAX_RAM_PERCENT = 10
    MAX_CPU_PERCENT = 10
    MAX_GPU_PERCENT = 10
    MAX_RAM_MB = 1024 # 1GB

    while True:
        try:
            virtual_mem = psutil.virtual_memory()
            current_ram_percent = virtual_mem.percent
            current_ram_mb = virtual_mem.used / (1024 * 1024)  # Convert to MB

            if current_ram_percent > MAX_RAM_PERCENT or current_ram_mb > MAX_RAM_MB:
                release_resources()

            current_cpu_percent = psutil.cpu_percent()
            if current_cpu_percent > MAX_CPU_PERCENT:
               print("CPU usage too high, attempting to reduce nice")
               p = psutil.Process(os.getpid())
               p.nice(1)

            if torch.cuda.is_available():
                gpu = torch.cuda.current_device()
                gpu_mem = torch.cuda.memory_percent(gpu)

                if gpu_mem > MAX_GPU_PERCENT:
                    release_resources()

            time.sleep(10) # Check every 10 seconds
        except Exception as e:
            print(f"Error in resource manager: {e}")
    

app = FastAPI()

@app.post("/generate")
async def generate(request: ChatRequest):
   try:
      return await process_message(request.message)
   except Exception as e:
      return JSONResponse(content={"error": str(e)})
        

@app.post("/generate_image")
async def generate_image_endpoint(request: ImageRequest):
   try:
       return await generate_image(request.prompt)
   except Exception as e:
       return JSONResponse(content={"error": str(e)})

def run_uvicorn():
    try:
        uvicorn.run(app, host="0.0.0.0", port=7860)
    except Exception as e:
        print(f"Error al ejecutar uvicorn: {e}")

if __name__ == "__main__":
    Thread(target=run_uvicorn).start()
    Thread(target=resource_manager, daemon=True).start()  # Run resource manager in background
    asyncio.get_event_loop().run_forever()
