import http.server
import sys
import urllib
from http.server import HTTPServer
from typing import List
from urllib.parse import parse_qs
from urllib.parse import urlparse

import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, \
    PreTrainedTokenizerBase as PTT, PreTrainedModel as PTM, \
    RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, pipeline

model_name = sys.argv[1]
# Uncomment and run once on a login node, then go offline for cluster mode
# tokenizer: PTT = RobertaTokenizer.from_pretrained(f'{model_name}')
# model: PTM = RobertaForMaskedLM.from_pretrained(f'{model_name}')
# Offline mode for running on compute nodes
tokenizer: PTT = RobertaTokenizer.from_pretrained(f'{model_name}', local_files_only=True)
model: PTM = RobertaForMaskedLM.from_pretrained(f'{model_name}', local_files_only=True)
attention_width = 760

print(f'Loaded model: {model_name}')


class EmbeddingServer(http.server.SimpleHTTPRequestHandler):
    def tokenize(self, query: str) -> List[int]:
        padded_query = f'{tokenizer.bos_token}{query}{tokenizer.eos_token}'
        # tic = time.perf_counter()
        tokens = tokenizer.tokenize(padded_query)
        # toc = time.perf_counter()
        # print(f"Tokenized in {toc - tic:0.4f} seconds")
        return tokenizer.convert_tokens_to_ids(tokens)

    def embed_sequence(self, query: str) -> np.ndarray:
        """
        Returns a sequence-embedded array. If smaller than attention_width, this
        will return a vector. Otherwise, this will return a matrix of sliding
        windows over the input sequence, arranged in rows.
        """
        sequence: Tensor = torch.tensor(self.tokenize(query))
        chunked = sequence[None, :] if len(sequence) < attention_width else \
            sequence.unfold(0, attention_width, int(attention_width/2))
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

        query_components = parse_qs(urlparse(self.path).query)
        # print(query_components)
        # print("PATH: %s" % self.path)

        if 'query' in query_components:
            query = urllib.parse.unquote_plus(query_components["query"][0])
            self.wfile.write(bytes(self.handle_query(query), encoding='utf8'))
        return

    def handle_query(self, query):
        if tokenizer.mask_token in query:
            pred = pipeline('fill-mask', model=model, tokenizer=tokenizer)
            outputs = pred(query)
            # TODO: Return top-k results?
            completion = max(outputs, key=lambda s: float(s['score']))
            return completion['token_str']
        else:
            array = self.embed_sequence(query)

            html = np.array2string(a=array,
                                   threshold=sys.maxsize,
                                   max_line_width=sys.maxsize)
            return html


my_server = HTTPServer(('', 8000), EmbeddingServer)
# Star the server
my_server.serve_forever()
