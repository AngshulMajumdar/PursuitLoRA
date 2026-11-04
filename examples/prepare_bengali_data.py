"""Download and tokenise the Bengali news benchmark data used in the PursuitLoRA / TensorPEFT experiments.

  python prepare_bengali_data.py --out data            # writes data/train.bin, data/val.bin

Source: Bengali news articles (Anandabazar, Ebela, Zee News) from
github.com/soham96/Bengali_news_classifier (pinned commit). Articles are shuffled with a fixed seed;
the first 400 articles form the validation split. Tokens are GPT-2 BPE ids stored as uint16.
"""
import argparse
import io
import json
import os
import random
import re
import unicodedata
import urllib.request
import zipfile

import numpy as np
from transformers import AutoTokenizer

URL = ("https://raw.githubusercontent.com/soham96/Bengali_news_classifier/"
       "4c2a51b51fc56208426ef19368bfcb08acc9d0c4/data/data.zip")

p = argparse.ArgumentParser()
p.add_argument("--out", default="data")
p.add_argument("--tokenizer", default="openai-community/gpt2")
p.add_argument("--zip", default=None, help="use a local copy of data.zip instead of downloading")
a = p.parse_args()

raw = open(a.zip, "rb").read() if a.zip else urllib.request.urlopen(URL).read()
docs = []
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    for name in sorted(n for n in z.namelist() if n.endswith("_articles.txt") and "__MACOSX" not in n):
        for art in json.loads(z.read(name).decode("utf-8"))["articles"]:
            t = unicodedata.normalize("NFC", (art.get("title", "") + "। " + art.get("body", "")).strip())
            t = re.sub(r"\s+", " ", t)
            if len(t) > 200:
                docs.append(t)
random.Random(0).shuffle(docs)
tok = AutoTokenizer.from_pretrained(a.tokenizer)


def enc(ds, cap):
    ids = []
    for d in ds:
        ids += tok(d).input_ids + [tok.eos_token_id]
        if len(ids) >= cap:
            break
    return np.array(ids[:cap], dtype=np.uint16)


os.makedirs(a.out, exist_ok=True)
enc(docs[:400], 200_000).tofile(os.path.join(a.out, "val.bin"))
enc(docs[400:], 3_000_000).tofile(os.path.join(a.out, "train.bin"))
print(f"{len(docs)} articles -> {a.out}/train.bin, {a.out}/val.bin")
