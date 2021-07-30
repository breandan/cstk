import http.server
import numpy as np
import sys
import time
import torch
import urllib
from http.server import HTTPServer
from transformers import AutoTokenizer, AutoModel
from urllib.parse import parse_qs
from urllib.parse import urlparse

model = sys.argv[1]
print(f'Model: {model}')
tokenizer = AutoTokenizer.from_pretrained(f'microsoft/{model}')
model = AutoModel.from_pretrained(f'microsoft/{model}')


class EmbeddingServer(http.server.SimpleHTTPRequestHandler):
    def tokenize(self, query):
        # tic = time.perf_counter()
        tokens = tokenizer.tokenize(query)
        # toc = time.perf_counter()
        # print(f"Tokenized in {toc - tic:0.4f} seconds")
        return tokens

    def vectorize(self, query):
        input_sequence = tokenizer.convert_tokens_to_ids(self.tokenize(query))
        # This should always be called with the beginning of sequence token <s>
        # https://github.com/huggingface/transformers/issues/1950#issuecomment-558770861
        # TODO: Should we return the entire vector or just the first?
        npy = model(torch.tensor(input_sequence)[None, :])[0].detach().numpy()
        return npy

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        query_components = parse_qs(urlparse(self.path).query)
        # print(query_components)
        # print("PATH: %s" % self.path)
        # print("QUERY: %s" % query)
        html = ''

        if 'tokenize' in query_components:
            query = urllib.parse.unquote_plus(query_components["tokenize"][0])
            tokens = self.tokenize(query)
            html = " ".join(str(x) for x in tokens)

        if 'vectorize' in query_components:
            query = urllib.parse.unquote_plus(query_components["vectorize"][0])
            array = self.vectorize(query)
            html = np.array2string(a=array,
                                   threshold=sys.maxsize,
                                   max_line_width=sys.maxsize)

        self.wfile.write(bytes(html, "utf8"))

        return


my_server = HTTPServer(('', 8000), EmbeddingServer)
# Star the server
my_server.serve_forever()
