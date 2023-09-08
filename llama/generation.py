# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch
from tqdm import tqdm
from pathlib import Path
import time
import json

from llama.tokenizer import Tokenizer
from llama.model import Transformer, ModelArgs


class LLaMA:
    @staticmethod
    def load(
            ckpt_dir: str,
            tokenizer_path: str,
            max_seq_len: int,
            max_batch_size: int,
        ) -> "LLaMA":
        print("Creating model...")
        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))

        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
        )

        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words

        model = Transformer(model_args)

        # Original copyright by tloen
        # https://github.com/tloen/llama-int8/blob/main/example.py
        key_to_dim = {
            "w1": 0,
            "w2": -1,
            "w3": 0,
            "wo": -1,
            "wq": 0,
            "wk": 0,
            "wv": 0,
            "output": 0,
            "tok_embeddings": -1,
            "ffn_norm": None,
            "attention_norm": None,
            "norm": None,
            "rope": None,
        }

        for i, ckpt in enumerate(checkpoints):
            print(f"Loading checkpoint {i}")
            checkpoint = torch.load(ckpt, map_location="cpu")
            for parameter_name, parameter in model.named_parameters():
                short_name = parameter_name.split(".")[-2]
                if key_to_dim[short_name] is None and i == 0:
                    parameter.data = checkpoint[parameter_name]
                elif key_to_dim[short_name] == 0:
                    size = checkpoint[parameter_name].size(0)
                    parameter.data[size * i: size * (i + 1), :] = checkpoint[
                        parameter_name
                    ]
                elif key_to_dim[short_name] == -1:
                    size = checkpoint[parameter_name].size(-1)
                    parameter.data[:, size * i: size * (i + 1)] = checkpoint[
                        parameter_name
                    ]
                del checkpoint[parameter_name]
            del checkpoint

        model.to("cpu")

        generator = LLaMA(model, tokenizer)
        print(f"Loaded model in {time.time() - start_time:.2f} seconds")
        return generator

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
            self,
            prompts: List[str],
            max_gen_len: int,
            temperature: float = 0.8,
            top_p: float = 0.95,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cpu().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id

        start_pos = min_prompt_size
        prev_pos = 0

        steps = total_len - start_pos
        pbar = tqdm(total=steps)

        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

            pbar.update(1)

        pbar.close()

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
