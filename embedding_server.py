
import http.server
import sys
import urllib
from http.server import HTTPServer
from typing import List
from urllib.parse import parse_qs
from urllib.parse import urlparse
import argparse

import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForMaskedLM, \
    PreTrainedTokenizerBase as PTT, PreTrainedModel as PTM, pipeline


parser = argparse.ArgumentParser()
parser.add_argument('--models', type=str, nargs='+', required=True)
parser.add_argument('--offline', action='store_true')

args = parser.parse_args()

# Load all the models
models: dict = {}
tokenizers: dict = {}
for m in [m for m in args.models if m]:
    models[m]: PTM = AutoModelForMaskedLM.from_pretrained(f'{m}', local_files_only=args.offline)
    tokenizers[m]: PTT = AutoTokenizer.from_pretrained(f'{m}', local_files_only=args.offline)

attention_width = 760

print(f'Loaded models: {args.models}')


class EmbeddingServer(http.server.SimpleHTTPRequestHandler):
    def tokenize(self, query: str, model_name) -> List[int]:
        tokenizer = tokenizers[model_name]
        padded_query = f'{tokenizer.bos_token}{query}{tokenizer.eos_token}'
        # tic = time.perf_counter()
        tokens = tokenizer.tokenize(padded_query)
        # toc = time.perf_counter()
        # print(f"Tokenized in {toc - tic:0.4f} seconds")
        return tokenizer.convert_tokens_to_ids(tokens)

    def embed_sequence(self, query: str, model_name) -> np.ndarray:
        # Returns a sequence-embedded array. If the query is shorter than
        # attention_width, this will return a vector. Otherwise, return a 
        # matrix of sliding windows over the input sequence, arranged in rows.
        model = models[model_name]

        sequence: Tensor = torch.tensor(self.tokenize(query, model_name))
        chunks = sequence[None, :] if len(sequence) < attention_width else \
            sequence.unfold(0, attention_width, int(attention_width / 2))
        #                        kernel size        kernel overlap

        return np.array([model(i[None, :])[0].detach().numpy() for i in chunks])

    def log_message(self, format, *args):
        pass

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        model_name = self.path.split('?')[0]
        # Remove the first and last slash if present
        model_name = model_name.strip('/')
        query_components = parse_qs(urlparse(self.path).query)
        # print(query_components)
        # print("PATH: %s" % self.path)
        if model_name not in models:
            print(f'Model {model_name} not in models.')
            return

        if "query" in query_components:
            query = urllib.parse.unquote_plus(query_components["query"][0])
            hints = query_components["hint"] \
                if "hint" in query_components else None
            self.reply(self.handle_query(query, model_name, hints))
        elif "tokenize" in query_components:
            query = urllib.parse.unquote_plus(query_components["tokenize"][0])
            self.reply(str(self.tokenize(query, model_name)))
        else:
            print(f'Unknown command: {query_components}')
            return

    def reply(self, response: str):
        self.wfile.write(bytes(response, encoding='utf8'))

    def handle_query(self, query, model_name, hints=None) -> str:
        model = models[model_name]
        tokenizer = tokenizers[model_name]

        try:
            if tokenizer.mask_token in query:
                pred = pipeline(task='fill-mask', model=model,
                                tokenizer=tokenizer, targets=hints)
                outputs = pred(query)
                completions = sorted(outputs, key=lambda s: -float(s['score']))
                completions = list(map(lambda x: x['token_str'], completions))
                token = "\n".join(completions)
                return token
            else:
                array = self.embed_sequence(query, model_name)

                html = np.array2string(a=array,
                                       threshold=sys.maxsize,
                                       max_line_width=sys.maxsize)
                return html
        except:
            return "<???>"


my_server = HTTPServer(('', 8000), EmbeddingServer)
# Star the server
my_server.serve_forever()
