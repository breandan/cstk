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
from transformers import AutoTokenizer, AutoModel, \
    PreTrainedTokenizerBase as PTT, PreTrainedModel as PTM, \
    RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--models', type=str, nargs='+', required=True)
parser.add_argument('--offline', action='store_true')

args = parser.parse_args()

# Load all the models
models = {}
tokenizers = {}
for m in args.models:
    models[m]: PTM = RobertaForMaskedLM.from_pretrained(f'{m}', local_files_only=args.offline)
    tokenizers[m]: PTT = RobertaTokenizer.from_pretrained(f'{m}', local_files_only=args.offline)

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
        """
        Returns a sequence-embedded array. If smaller than attention_width, this
        will return a vector. Otherwise, this will return a matrix of sliding
        windows over the input sequence, arranged in rows.
        """
        model = models[model_name]

        sequence: Tensor = torch.tensor(self.tokenize(query, model_name))
        chunked = sequence[None, :] if len(sequence) < attention_width else \
            sequence.unfold(0, attention_width, int(attention_width / 2))
        #                        kernel size        kernel overlap

        # print(chunked)
        array = np.array([model(i[None, :])[0].detach().numpy() for i in chunked])
        # print(array)
        return array

    def log_message(self, format, *args):
        pass

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        model_name = self.path.split('?')[0]
        # Remove the first and last slash
        model_name = model_name[1:-1]
        query_components = parse_qs(urlparse(self.path).query)
        # print(query_components)
        # print("PATH: %s" % self.path)
        if model_name not in models or 'query' not in query_components:
            return

        query = urllib.parse.unquote_plus(query_components["query"][0])
        self.wfile.write(bytes(self.handle_query(query, model_name), encoding='utf8'))

    def handle_query(self, query, model_name):
        # print(query)
        model = models[model_name]
        tokenizer = tokenizers[model_name]

        if tokenizer.mask_token in query:
            pred = pipeline('fill-mask', model=model, tokenizer=tokenizer)
            outputs = pred(query)
            completions = sorted(outputs, key=lambda s: float(s['score']))
            completions = list(map(lambda x: x['token_str'], completions))
            token = "\n".join(completions)
            return token
        else:
            array = self.embed_sequence(query, model_name)

            html = np.array2string(a=array,
                                   threshold=sys.maxsize,
                                   max_line_width=sys.maxsize)
            return html


my_server = HTTPServer(('', 8000), EmbeddingServer)
# Star the server
my_server.serve_forever()
