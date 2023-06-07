import json
import requests
import subprocess
from fastapi import FastAPI, Request, Body
from modules.script_callbacks import on_app_started


class ControlnetRequest:
    def __init__(self, prompt, b64img, neg_prompt='', denoising_strength=0.75, weight=1.0):
        self.url = "http://localhost:7860/sdapi/v1/img2img"
        self.body = {
            "init_images": [b64img],
            "prompt": prompt,
            "negative_prompt": neg_prompt,
            "seed": -1,
            "subseed": -1,
            "subseed_strength": 0,
            "denoising_strength": denoising_strength,
            "batch_size": 1,
            "n_iter": 1,
            "steps": 20,
            "cfg_scale": 7,
            "width": 512,
            "height": 768,
            "restore_faces": False,
            "tiling": False,
            "do_not_save_samples": False,
            "do_not_save_grid": False,
            "eta": 0,
            "s_min_uncond": 0,
            "s_churn": 0,
            "s_tmax": 0,
            "s_tmin": 0,
            "s_noise": 1,
            "send_images": True,
            "sampler_index": "DPM++ 2M Karras",
            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                            "input_image": b64img,
                            "module": 'lineart_realistic',
                            "model": 'control_v11p_sd15_lineart [43d4be0d]',
                            'resize_mode': 'Crop and Resize',
                            'weight': weight,
                            'low_vram': True,
                            'processor_res': 512,
                            'threshold_a': 0.1,
                            'threshold_b': 0.1,
                            'guidance_start': 0,
                            'guidance_end': 1,
                            'pixel_perfect': True,
                            'loopback': False,
                            'enabled': True,
                            'is_ui': False,
                            'control_mode': 'ControlNet is more important',
                        }
                    ]
                }
            }
        }

    def send_request(self):
        r = requests.post(self.url, json=self.body)
        return r.json()


def healthcheck():
    gpu = False
    out = subprocess.run("nvidia-smi", shell=True)
    if out.returncode == 0:  # success state on shell command
        gpu = True
    return {"state": "healthy", "gpu": gpu}


async def inference(request: Request):
    body = await request.body()
    model_input = json.loads(body)
    params = model_input['params']
    prompt = params['prompt']
    neg_prompt = params['neg_prompt']
    b64img = params['image']
    cr = ControlnetRequest(prompt, b64img, neg_prompt)
    output = cr.send_request()
    if 'images' in output:
        output = {
            "base64_output": output["images"][0]
        }
    return output


def register_endpoints(block, app):
    app.add_api_route('/healthcheck', healthcheck, methods=['GET'])
    app.add_api_route('/', inference, methods=['POST'])


on_app_started(register_endpoints)
