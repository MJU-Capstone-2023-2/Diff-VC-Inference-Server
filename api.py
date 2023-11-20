import os
import gc
import uuid
import json
from time import time
from loguru import logger
import numpy as np 


from fastapi import FastAPI, Response, status, File, UploadFile, Body
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


from inference import Inferencer 

import params
from model import DiffVC

import sys
sys.path.append('hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN

sys.path.append('speaker_encoder/')
from encoder import inference as spk_encoder
from pathlib import Path

import torch
from fastapi.responses import StreamingResponse
import torchaudio
from io import BytesIO


import requests
from dotenv import load_dotenv
load_dotenv()

use_gpu = torch.cuda.is_available()
vc_path = 'checkpts/vc/vc_libritts_wodyn.pt' # path to voice conversion model

generator = DiffVC(params.n_mels, params.channels, params.filters, params.heads, 
                   params.layers, params.kernel, params.dropout, params.window_size, 
                   params.enc_dim, params.spk_dim, params.use_ref_t, params.dec_dim, 
                   params.beta_min, params.beta_max)
if use_gpu:
    generator = generator.cuda()
    generator.load_state_dict(torch.load(vc_path))
else:
    print("go cpu")
    generator.load_state_dict(torch.load(vc_path, map_location='cpu'))
generator.eval()


# loading HiFi-GAN vocoder
hfg_path = 'checkpts/vocoder/' # HiFi-GAN path

with open(hfg_path + 'config.json') as f:
    h = AttrDict(json.load(f))

if use_gpu:
    hifigan_universal = HiFiGAN(h).cuda()
    hifigan_universal.load_state_dict(torch.load(hfg_path + 'generator')['generator'])
else:
    print("go cpu")
    hifigan_universal = HiFiGAN(h)
    hifigan_universal.load_state_dict(torch.load(hfg_path + 'generator',  map_location='cpu')['generator'])

_ = hifigan_universal.eval()
hifigan_universal.remove_weight_norm()


# loading speaker encoder
enc_model_fpath = Path('checkpts/spk_encoder/pretrained.pt') # speaker encoder path
if use_gpu:
    spk_encoder.load_model(enc_model_fpath, device="cuda")
else:
    spk_encoder.load_model(enc_model_fpath, device="cpu")

# Make dir to save audio files log
MEDIA_ROOT = os.path.join('/logs', 'media')
if not os.path.exists(MEDIA_ROOT):
    os.makedirs(MEDIA_ROOT)

# Make dir to save json response log
LOG_ROOT = os.path.join('/logs', 'json')
if not os.path.exists(LOG_ROOT):
    os.makedirs(LOG_ROOT)


# Define Inferencer 
_inferencer = Inferencer(generator, spk_encoder, hifigan_universal, MEDIA_ROOT, True)


def save_audio(file):
    job_id = str(uuid.uuid4())
    output_dir = os.path.join(MEDIA_ROOT, str(job_id))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    audio_save_path = os.path.join(output_dir, file.filename)
    with open(audio_save_path, "wb+") as file_object:
        file_object.write(file.file.read())
    
    return audio_save_path 
    

app = FastAPI(
    title="Voice Conversion",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/', status_code=status.HTTP_200_OK)
async def check_status(response: Response):
    api_status = {"API Status": "Running"}
    return api_status



@app.post('/convert', status_code=200)
async def convert(response:StreamingResponse, file1: UploadFile = File(...), file2: UploadFile = File(...) ):
    # Save source and target to MEDIA 
    source_fpath = save_audio(file1)
    print(source_fpath)
    target_fpath = save_audio(file2)
    print(target_fpath)
    
    waveform = _inferencer.infer(src_path=source_fpath, tgt_path=target_fpath, return_output_path=False)
    
    # Save the waveform as a BytesIO object
    buffer = BytesIO()
    torchaudio.save(buffer, waveform.view(1, -1), sample_rate=22050, format="wav")
    
    # Set the headers and return the StreamingResponse
    buffer.seek(0)
    headers = {
        'Content-Disposition': 'attachment; filename="generated_audio.wav"'
    }
    return StreamingResponse(buffer, media_type="audio/wav", headers=headers)


# @app.post('/spk_enc', status_code=200)
# async def spk_enc(response:StreamingResponse, file1: UploadFile = File(...)):
#     source_fpath = save_audio(file1)
#     # 오디오 파일 읽기
#     with open(source_fpath, 'rb') as file:
#         files = {'file': file}
#         print(os.environ.get('TRITON_URL'))
#         # response = requests.post(f"http://{os.environ.get('TRITON_URL')}/v2/models/spk_enc/infer", files=files)
#         response = requests.post(f"http://host.docker.internal:8000/v2/models/spk_enc/infer", files=files)

#     # 받은 스트림을 NumPy 배열로 변환
#     if response.status_code == 200:
#         audio_bytes = response.json()['data']
#         audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

#         # 여기서 NumPy 배열로 된 오디오 데이터를 사용할 수 있음
#         print(audio_np)
#     else:
#         print("Error:", response.text)
#     return response
