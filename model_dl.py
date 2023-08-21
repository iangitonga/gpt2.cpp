#!/usr/bin/python3

import os
import sys
from urllib import request


MODELS = ("Gpt2", "Gpt2-medium", "Gpt2-large")

MODELS_URLS = {
    "Gpt2": "https://huggingface.co/iangitonga/gten/resolve/main/Gpt2.fp16.gten",
    "Gpt2-medium": "https://huggingface.co/iangitonga/gten/resolve/main/Gpt2-medium.fp16.gten",
    "Gpt2-large": "https://huggingface.co/iangitonga/gten/resolve/main/Gpt2-large.fp16.gten",
}


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

def download_model(model_name):
    model_path = os.path.join("models", f"{model_name}.fp16.gten")
    if os.path.exists(model_path):
        return
    os.makedirs("models", exist_ok=True)
    model_url_key = f"{model_name}.fp16"
    _download_model(MODELS_URLS[model_name], model_path)


if len(sys.argv) < 2 or sys.argv[1] not in MODELS:
    print("Model not specified.\n")
    print("usage: model_registry.py MODEL")
    print("MODEL is one of (Gpt2, Gpt2-medium, Gpt2-large)")


try:
    download_model(sys.argv[1])
except:
    exit(-2)
