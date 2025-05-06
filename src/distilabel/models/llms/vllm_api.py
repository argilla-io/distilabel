import subprocess
import socket
import random
import time
import os
import signal

import openai

from distilabel import utils
from distilabel.pydantics import LMConfig

class vLLMAPI:
    '''vLLM API base class.

    Handles starting and stopping the vLLM server.
    '''
    gpu: int = 0

    def __init__(self, lm_config: LMConfig):
        self.lm_config = lm_config

        self.num_gpus = 1
        # self.gpu_offset = lm_config.gpu_offset
        self.gmu = 0.9
        self.vllm_server_pid = None

        self.port = self.random_available_port()
        os.makedirs('vllm_logs', exist_ok=True)

    def port_in_use(self, port: int) -> bool:
        '''Check if a port is in use.'''
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    def random_available_port(self) -> int:
        '''Get a random available port.'''
        while True:
            port = random.randint(1024, 49151)
            if not self.port_in_use(port):
                return port

    def start_vllm(self, launch_vllm, timeout=120):
        '''Start the asynchronous vLLM server.'''
        print(f'[{self.gpu}] Initializing vLLM Server...')
        print('='*70)
        with utils.suppress_output(debug=False):
            process = subprocess.Popen(  # noqa: S602
                launch_vllm,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = process.communicate()
            self.vllm_server_pid = int(stdout.strip().decode('utf-8'))

            # wait patiently for vllm server to start
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < timeout:
                try:
                    self.establish_client_vllm()
                    print(f'[{self.gpu}] vLLM Server Initialized')
                    print('='*70)
                except openai.APIConnectionError:
                    time.sleep(5)
                else:
                    return

        self.cleanup()
        err = f'vllm server failed to start within timeout {timeout}s'
        raise RuntimeError(err)

    def establish_client_vllm(self):
        '''Establish a openai client to the vLLM server.'''
        self.client = openai.OpenAI(api_key='empty', base_url=f'http://localhost:{self.port}/v1')
        self.model_name = self.client.models.list().data[0].id

    def cleanup(self):
        '''Kill the vLLM server.'''
        if not self.vllm_server_pid:
            return

        def is_process_running(pid: int) -> bool:
            try:
                os.kill(pid, 0)
            except OSError:
                return False
            return True

        pid = self.vllm_server_pid
        if is_process_running(pid):
            try:
                os.kill(pid, signal.SIGTERM)
                print(f'[{self.gpu}] Server with PID {pid} has been sent SIGTERM.')
                time.sleep(10)  # Allow time for cleanup
                if is_process_running(pid):
                    os.kill(pid, signal.SIGKILL)
                    print(f'[{self.gpu}] Server with PID {pid} has been killed.')
            except ProcessLookupError:
                print(f'[{self.gpu}] Server with PID {pid} does not exist.')
