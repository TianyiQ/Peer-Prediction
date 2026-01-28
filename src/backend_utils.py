import torch
import numpy as np
import gc
import json
import time
import os
import tqdm
import random
from typing import List, Tuple, Literal
import multiprocessing
from multiprocessing import freeze_support
from copy import deepcopy
from dataset_utils import JsonListWriter, JsonListReader
import sglang as sgl
import subprocess
from nvitop import GpuProcess, Device
import signal
import sys

GPU_COUNT = torch.cuda.device_count()
MY_USERNAME = os.environ.get('USER', os.environ.get('USERNAME', ''))

print(f"GPU_COUNT={GPU_COUNT}")

def kill_all_my_gpu_processes():
    devices = Device.cuda.all()
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    for device in devices:
        processes = device.processes()
        processes = GpuProcess.take_snapshots(processes.values(), failsafe=True)
        for process in processes:
            if process.username.lower() == MY_USERNAME.lower():
                print(f'Killing process {process.pid}: {process.cmdline}')
                os.kill(process.pid, signal.SIGTERM)
                os.kill(process.pid, signal.SIGINT)
                os.kill(process.pid, signal.SIGKILL)

def get_model_size(model_repoid_or_path: str) -> float:
    if 'mislead' in model_repoid_or_path.lower():
        return 7
    
    model_size = (
        27 if '27b' in model_repoid_or_path.lower() else
        9 if '9b' in model_repoid_or_path.lower() else
        8 if '8b' in model_repoid_or_path.lower() else
        70 if '70b' in model_repoid_or_path.lower() else
        405 if '405b' in model_repoid_or_path.lower() else
        13 if '13b' in model_repoid_or_path.lower() else
        2 if '2b' in model_repoid_or_path.lower() else
        4 if '3.5-mini' in model_repoid_or_path.lower() else
        4 if '4b' in model_repoid_or_path.lower() else
        0.5 if '0.5b' in model_repoid_or_path.lower() else
        1.5 if '1.5b' in model_repoid_or_path.lower() else
        7 if '7b' in model_repoid_or_path.lower() else
        3 if '3b' in model_repoid_or_path.lower() else
        1.7 if '1.7b' in model_repoid_or_path.lower() else
        0.135 if '135m' in model_repoid_or_path.lower() else
        0.360 if '360m' in model_repoid_or_path.lower() else
        None
    )
    return model_size

def start_backend(model_repoid_or_path: str, purpose: Literal['responses', 'logprobs'], silent: bool = True, port: int = 13285) -> subprocess.Popen:
    
    if os.environ.get('LOUD_BACKEND', '0') == '1':
        silent = False
    
    if os.environ.get('HALT_BEFORE_LOAD', '0') == '1':
        print("Halted before loading backend.")
        sys.exit(0)
    
    with open(os.devnull, 'w') as devnull:
        frac_static = (0.8 if purpose == 'responses' else 0.4)
        prefill_size = (8192 if purpose == 'responses' else 1024)
        
        model_size = get_model_size(model_repoid_or_path)
        assert model_size is not None
        
        if model_size <= 10:
            args = ['python', '-m', 'sglang.launch_server', '--port', f'{port}', f'--dp', f'{GPU_COUNT}', '--model', model_repoid_or_path, '--mem-fraction-static', f'{frac_static}', '--chunked-prefill-size', f'{prefill_size}', '--trust-remote-code', '--schedule-conservativeness', '0.3']
        
        else:
            min_gpus_per_instance = (2 if model_size <= 30 else 
                                     4 if model_size <= 80 else 8)
            if purpose == 'responses': min_gpus_per_instance //= 2
            assert GPU_COUNT % min_gpus_per_instance == 0
            args = ['python', '-m', 'sglang.launch_server', '--port', f'{port}', f'--tp', f'{min_gpus_per_instance}', f'--dp', f'{GPU_COUNT//min_gpus_per_instance}', '--model', model_repoid_or_path, '--mem-fraction-static', f'{frac_static}', '--chunked-prefill-size', f'{prefill_size}', '--trust-remote-code']
        
        #if 'int4' not in model_repoid_or_path.lower():
        #    args += ['--quantization', 'fp8']
        
        if 'phi' in model_repoid_or_path.lower():
            args += ['--disable-flashinfer']
        
        if 'smol' in model_repoid_or_path.lower():
            args += ['--chat-template=chatml']
        
        if 'mislead' in model_repoid_or_path.lower():
            args += ['--chat-template=llama-2']
        
        print(f"Starting backend for {model_repoid_or_path} - {args}", flush=True)
        
        if silent:
            backend = subprocess.Popen(args, stdout=devnull, stderr=devnull)
        else:
            backend = subprocess.Popen(args)
    
    # Wait for backend to start
    while True:
        time.sleep(30)
        try:
            print("Trying to connect to backend...", flush=True)
            sgl.set_default_backend(sgl.RuntimeEndpoint(f"http://localhost:{port}"))
            print("Connected to backend.", flush=True)
            break
        except:
            print("Failed to connect to backend (this is to be expected if backend is still starting). Retrying after 30s...", flush=True)
            pass
    
    return backend