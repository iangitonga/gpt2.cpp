import argparse
import math
import pickle
import urllib
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


# MODEL CONFIG PARAMETERS
@dataclass
class ModelConfig: 
    n_vocab: int =  None
    n_ctx: int = None
    n_state: int = None
    n_layer: int = None
    n_mlp: int = None
    n_head: int = None



class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_head: int, n_state: int, n_ctx: int):
        super().__init__()
        
        self.n_head = n_head
        self.n_state = n_state
        self.c_attn = nn.Linear(n_state, n_state * 3)
        self.c_proj = nn.Linear(n_state, n_state)
        
        # The masking attn mask.
        bias = torch.empty((1, 1, n_ctx, n_ctx))
        self.register_buffer('bias', bias, persistent=True)
        
    def forward(self, x):
        q, k, v = self.c_attn(x).split(self.n_state, dim=2)
        qkv = self._qkv_attention(q, k, v)
        out = self.c_proj(qkv)
        return out
    
    def _qkv_attention(self, q, k, v):
        n_batch, n_ctx = q.shape[0], q.shape[1]
        d_head = self.n_state // self.n_head
        q = q.view(n_batch, n_ctx, self.n_head, d_head).permute(0, 2, 1, 3)
        k = k.view(n_batch, n_ctx, self.n_head, d_head).permute(0, 2, 3, 1)
        v = v.view(n_batch, n_ctx, self.n_head, d_head).permute(0, 2, 1, 3)
        scale = 1.0 / math.sqrt(d_head)
        qk = (q @ k) * scale
        qk = qk.masked_fill(self.bias[:, :, :n_ctx, :n_ctx] == 0, float('-inf'))
        qk = F.softmax(qk, dim=-1)
        qkv = qk @ v
        qkv = qkv.permute(0, 2, 1, 3).flatten(start_dim=2)
        return qkv
    

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, n_mlp: int, n_ctx: int):
        super().__init__()
        
        self.attn = MultiHeadSelfAttention(n_head, n_state, n_ctx)
        self.ln_1 = nn.LayerNorm(n_state)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_state, n_mlp),
            c_proj  = nn.Linear(n_mlp, n_state),
            act     = nn.GELU(approximate="tanh"),
        ))
        self.mlpf = lambda x: self.mlp.c_proj(self.mlp.act(self.mlp.c_fc(x))) # MLP forward
        self.ln_2 = nn.LayerNorm(n_state)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x
    

# For inference only. no dropout layers.
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.wte = nn.Embedding(config.n_vocab, config.n_state)
        self.wpe = nn.Embedding(config.n_ctx, config.n_state)
        self.h = nn.ModuleList([
            ResidualAttentionBlock(config.n_state, config.n_head, config.n_mlp, config.n_ctx) for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_state)
        
    def forward(self, x):
        pos = torch.arange(0, x.shape[1], dtype=torch.long).unsqueeze(0)
        x = self.wte(x) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = (x @ torch.transpose(self.wte.weight.to(x.dtype), 0, 1)).float()
        return logits

    @classmethod
    def from_pretrained(cls, path, config):
        model = cls(config)
        gpt_state = torch.load(path, map_location="cpu")
        for key in gpt_state.keys():
            if (key.endswith("attn.c_attn.weight")
                or key.endswith("attn.c_proj.weight")
                or key.endswith("mlp.c_fc.weight")
                or key.endswith("mlp.c_proj.weight")):
                gpt_state[key] = gpt_state[key].transpose(0, 1)
        model.load_state_dict(gpt_state)
        return model


GTEN_MAGIC_NUMBER = 0x454c49464e455447
BYTEORDER = "little"

# unsigned int to bytes.
def itob(integer, width=4):
    return int.to_bytes(integer, width, BYTEORDER, signed=True)

# float to bytes
def ftob(floatv):
    return np.array([floatv]).astype(np.float32).tobytes()


# Each row of a weight tensor is divided into blocks which are then
# quantized. Each block contains `block-size` numbers. The higher the
# block-size, the higher the compression but the model performance in
# terms of perplexity may decrease.
def quantize(t: torch.Tensor, q_blk_size: int):
    assert len(t.shape) == 2, f"Illegal shape: {t.shape}"
    # 2-D tensors transposed. (d_out, d_in)
    d_in = t.shape[1]
    assert d_in % q_blk_size == 0, f"Illegal d_in: {d_in}"

    # reshape to d_out, D, Q8_BLOCK_SIZE => Q8_BLOCK_SIZE, d_out*D
    d_out = t.shape[0]
    n_blocks_per_row = d_in // q_blk_size
    n_blocks = n_blocks_per_row * d_out
    t = t.view(n_blocks, q_blk_size)

    # compute deltas for each block.
    absmax = t.abs().amax(dim=1)
    deltas = absmax / 127.0
    deltas = deltas.to(torch.float32)

    scalars = deltas.clone().view(n_blocks)
    # mask indices prevents division by zero by dividing only non-zero values.
    non_zero_idxs = scalars != 0
    scalars[non_zero_idxs] = 1.0 / scalars[non_zero_idxs]
    scalars = scalars.view(n_blocks, 1)

    # cvt
    t = torch.round(t * scalars).to(torch.int8)

    t = t.view(d_out, d_in)

    return deltas.to(torch.float16), t


def tensor_to_bytes(t, out_dtype):
    return t.detach().numpy().flatten().astype(out_dtype).tobytes()


def write_layer(fout, name, weights_size, w0, bias, q_blk_size):
    name = name.encode()
    # <layer_name_size, layer_name>
    fout.write(itob(len(name)))
    fout.write(name)

    # W0
    if w0 is not None:
        w0_name = f"{name.decode()}.weight".encode()
        fout.write(itob(len(w0_name)))
        fout.write(w0_name)

        w0 = w0.detach()

        if weights_size == 1 and len(w0.shape) == 2:
            deltas, w0 = quantize(w0, q_blk_size)
            deltas_bytes = deltas.numpy().flatten().tobytes()
            fout.write(itob(len(deltas_bytes), width=4))
            fout.write(deltas_bytes)
        elif weights_size == 1 and len(w0.shape) == 1:
            w0 = w0.to(torch.float16)
        elif weights_size == 2:
            w0 = w0.to(torch.float16)
        else:
            w0 = w0.to(torch.float32)
        
        w0 = w0.numpy().flatten()
        w0_bytes = w0.tobytes()
        fout.write(itob(len(w0_bytes)))
        fout.write(w0_bytes)

    # Bias
    if bias is not None:
        # NOTE: We do not quantize biases.
        bias_name = f"{name.decode()}.bias".encode()
        fout.write(itob(len(bias_name)))
        fout.write(bias_name)

        bias = bias.detach()
        # Note: 1-D tensors are not quantized.
        if weights_size == 1 or weights_size == 2:
            bias = bias.to(torch.float16)
        else:
            bias = bias.to(torch.float32)

        bias_bytes = bias.numpy().flatten().tobytes()
        fout.write(itob(len(bias_bytes)))
        fout.write(bias_bytes)


def write_block(fout, name, model, weights_size, block_idx, q_blk_size):
    name = name.encode()
    # <block_name_size, block_name>
    fout.write(itob(len(name)))
    fout.write(name)

    h = model.h[block_idx]
    attn_qw, attn_kw, attn_vw = h.attn.c_attn.weight.split(model.config.n_state, dim=0)
    attn_qb, attn_kb, attn_vb = h.attn.c_attn.bias.split(model.config.n_state, dim=0)

    layer_name = f"{name.decode()}.attn.q"
    write_layer(fout, layer_name, weights_size=weights_size, w0=attn_qw, bias=attn_qb, q_blk_size=q_blk_size)

    layer_name = f"{name.decode()}.attn.k"
    write_layer(fout, layer_name, weights_size=weights_size, w0=attn_kw, bias=attn_kb, q_blk_size=q_blk_size)

    layer_name = f"{name.decode()}.attn.v"
    write_layer(fout, layer_name, weights_size=weights_size, w0=attn_vw, bias=attn_vb, q_blk_size=q_blk_size)

    attn_pw = h.attn.c_proj.weight
    attn_pb = h.attn.c_proj.bias
    layer_name = f"{name.decode()}.attn.c_proj"
    write_layer(fout, layer_name, weights_size=weights_size, w0=attn_pw, bias=attn_pb, q_blk_size=q_blk_size)

    ln_1w = h.ln_1.weight
    ln_1b = h.ln_1.bias
    layer_name = f"{name.decode()}.ln_1"
    write_layer(fout, layer_name, weights_size=weights_size, w0=ln_1w, bias=ln_1b, q_blk_size=q_blk_size)

    mlp_fcw = h.mlp.c_fc.weight
    mlp_fcb = h.mlp.c_fc.bias
    layer_name = f"{name.decode()}.mlp.c_fc"
    write_layer(fout, layer_name, weights_size=weights_size, w0=mlp_fcw, bias=mlp_fcb, q_blk_size=q_blk_size)

    mlp_pw = h.mlp.c_proj.weight
    mlp_pb = h.mlp.c_proj.bias
    layer_name = f"{name.decode()}.mlp.c_proj"
    write_layer(fout, layer_name, weights_size=weights_size, w0=mlp_pw, bias=mlp_pb, q_blk_size=q_blk_size)

    ln_2w = h.ln_2.weight
    ln_2b = h.ln_2.bias
    layer_name = f"{name.decode()}.ln_2"
    write_layer(fout, layer_name, weights_size=weights_size, w0=ln_2w, bias=ln_2b, q_blk_size=q_blk_size)
    

def convert_model_to_gten(model, model_fname, vocab_fname, weights_size, q_blk_size: int):
    with open(model_fname, "wb") as fout:
        fout.write(itob(GTEN_MAGIC_NUMBER, width=8))
        fout.write(itob(model.config.n_vocab))
        fout.write(itob(model.config.n_ctx))
        fout.write(itob(model.config.n_state))
        fout.write(itob(model.config.n_layer))
        fout.write(itob(model.config.n_head))
        
        print("Writing vocab")
        segment_name = b"vocab"
        segment_name_size = len(segment_name)
        expected_size = 521_855
        with open(vocab_fname, "rb") as vf:
            vocab_bytes = vf.read()
            segment_size = len(vocab_bytes)
            assert segment_size == expected_size, f"Vocab: expected={expected_size}, real={segment_size}"
            fout.write(itob(segment_name_size))
            fout.write(segment_name)
            fout.write(itob(segment_size))
            fout.write(vocab_bytes)

        if weights_size == 1:
            segment_name = b"quants.info"
            segment_name_size = len(segment_name)
            fout.write(itob(segment_name_size))
            fout.write(segment_name)
            fout.write(itob(q_blk_size))
        
        print("Converting wte")
        write_layer(fout, "wte", weights_size=weights_size, w0=model.wte.weight, bias=None, q_blk_size=q_blk_size)
        
        print("Converting wpe")
        write_layer(fout, "wpe", weights_size=weights_size, w0=model.wpe.weight, bias=None, q_blk_size=q_blk_size)
            
        for i in range(model.config.n_layer):
            print(f"Converting block_{i}")
            write_block(fout, f"h.{i}", model, weights_size, i, q_blk_size)
        
        print("Converting ln_f")
        write_layer(fout, "ln_f", weights_size=weights_size, w0=model.ln_f.weight, bias=model.ln_f.bias, q_blk_size=q_blk_size)



# Upload vocab fname[down].
# Download model.

def download_model(model_name, url):
    model_path = f"{model_name}"

    print(f"Downloading {model_name}")
    with urllib.request.urlopen(url) as source, open(model_path, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))
    return model_path

def download_vocab(url):
    print("Downloading vocab")
    vocab_path = "gpt2_vocab.bin"
    with urllib.request.urlopen(url) as source, open(vocab_path, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return vocab_path


VOCAB_URL = "https://huggingface.co/iangitonga/gten/resolve/main/gpt-2-vocab.gten"

MODEL_URL = {
    "Gpt2": "https://huggingface.co/gpt2/resolve/main/pytorch_model.bin",
    "Gpt2-medium": "https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin",
    "Gpt2-large": "https://huggingface.co/gpt2-large/resolve/main/pytorch_model.bin",
    "Gpt2-xl": "https://huggingface.co/gpt2-xl/resolve/main/pytorch_model.bin",
}

MODEL_CONFIG = {
    "Gpt2":        ModelConfig(n_vocab = 50257, n_ctx = 1024, n_state =  768, n_layer = 12, n_head = 12, n_mlp =  768 * 4),
    "Gpt2-medium": ModelConfig(n_vocab = 50257, n_ctx = 1024, n_state = 1024, n_layer = 24, n_head = 16, n_mlp = 1024 * 4),
    "Gpt2-large":  ModelConfig(n_vocab = 50257, n_ctx = 1024, n_state = 1280, n_layer = 36, n_head = 20, n_mlp = 1280 * 4),
    "Gpt2-xl":     ModelConfig(n_vocab = 50257, n_ctx = 1024, n_state = 1600, n_layer = 48, n_head = 25, n_mlp = 1600 * 4),
}

OUT_DTYPES = (
    "f32",
    "f16",
    "qint8"
)

QINT_BLOCK_SIZES = (
    "B8",
    "B16",
    "B32",
    "B64",
    "B128",
    "B256",
)

def download_and_convert(model_name, out_dtype, q_blk_id, model_path, vocab_path):
    qblock_size = int(q_blk_id[1:])
    if out_dtype == "f32":
        weights_size = 4
        model_fname = f"{model_name}.f32.gten"
        print(F"Converting to float32 format ...")
    elif out_dtype == "f16":
        model_fname = f"{model_name}.f16.gten"
        weights_size = 2
        print(F"Converting to float16 format ...")
    elif out_dtype == "qint8":
        model_fname = f"{model_name}.q8.gten"
        weights_size = 1
        print(F"Converting to Qint8 format ...")

    if not model_path:
        model_path = download_model(model_name, MODEL_URL[model_name])
    if not vocab_path:
        vocab_path = download_vocab(VOCAB_URL)
    model = Transformer.from_pretrained(model_path, MODEL_CONFIG[model_name])
    convert_model_to_gten(model, model_fname, vocab_path, weights_size, qblock_size)
    print("Conversion complete!!!")


parser = argparse.ArgumentParser()
parser.add_argument("model", help="Model name to be converted.", choices=MODEL_CONFIG.keys())
parser.add_argument("--out_dtype", help="Output data format. default is f16.", choices=OUT_DTYPES, default="f16")
parser.add_argument("--qblock_size", help="Quantize block size id. default is B32.", choices=QINT_BLOCK_SIZES, default="B64")
parser.add_argument("--mpath", help="Optional path to source model if you have it locally.")
parser.add_argument("--vpath", help="Optional path to vocab if you have it locally.")

args = parser.parse_args()

download_and_convert(args.model, args.out_dtype, args.qblock_size, args.mpath, args.vpath)
