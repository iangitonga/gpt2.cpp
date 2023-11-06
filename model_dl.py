#!/usr/bin/python3

import os
import sys
from urllib import request


MODELS = ("Gpt2", "Gpt2-medium", "Gpt2-large", "Gpt2-xl")

MODELS_URLS = {
    "Gpt2": {
        "float16": "https://huggingface.co/iangitonga/gten/resolve/main/Gpt2.fp16.gten"
    },
    "Gpt2-medium": {
        "float16": "https://huggingface.co/iangitonga/gten/resolve/main/Gpt2-medium.fp16.gten"
    },
    "Gpt2-large": {
        "float16": "https://huggingface.co/iangitonga/gten/resolve/main/Gpt2-large.fp16.gten",
        "qint8": "https://huggingface.co/iangitonga/gten/resolve/main/Gpt2-large.q8.gten"
    },
    "Gpt2-xl": {
        "float16": "https://huggingface.co/iangitonga/gten/resolve/main/Gpt2-xl.f16.gten",
        "qint8": "https://huggingface.co/iangitonga/gten/resolve/main/Gpt2-xl.q8.gten"
    }
}

DTYPES = (
    "float16",
    "qint8"
)


def show_progress(cur_size, max_size):
    ls = [" "] * 50
    prog = int(cur_size / max_size * 50)
    for i in range(prog):
        ls[i] = "#"
    print("Progress: [" + "".join(ls) + "]", end="\r", flush=True)
    if cur_size == max_size:
        print()

def _download_model(url, model_path):
    print("Downloading ...")
    with request.urlopen(url) as source, open(model_path, "wb") as output:
        download_size = int(source.info().get("Content-Length"))
        while True:
            buffer = source.read(8192)
            if not buffer:
                break

            output.write(buffer)
            show_progress(len(buffer), download_size)

def download_model(model_name, dtype):
    if dtype == "qint8":
        model_path = os.path.join("models", f"{model_name}.q8.gten")
    else:
        model_path = os.path.join("models", f"{model_name}.fp16.gten")
    if os.path.exists(model_path):
        return
    os.makedirs("models", exist_ok=True)
    model_url_key = f"{model_name}.qint8" if dtype == "qint8" else f"{model_name}.fp16"
    _download_model(MODELS_URLS[model_name][dtype], model_path)

# python model.py model inf
if len(sys.argv) != 3 or sys.argv[1] not in MODELS or sys.argv[2] not in DTYPES:
    print(f"Args provided: {sys.argv}")
    print("usage: model_registry.py MODEL DTYPE")
    print("MODEL is one of (Gpt2, Gpt2-medium, Gpt2-large, Gpt2-xl)")
    print("DTYPE is one of (float16, qint8)")
    exit(-1)


try:
    download_model(sys.argv[1], sys.argv[2])
except:
    exit(-2)
