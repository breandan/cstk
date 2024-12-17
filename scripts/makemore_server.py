import os
import sys
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote

# Import everything from makemore (assuming makemore.py is in the same directory)
# Alternatively, you might integrate this into the makemore script itself.
from makemore import ModelConfig, Transformer, create_datasets, generate

# ------------------------------------------------------------
# Load model and datasets (adjust paths and args as needed)
# ------------------------------------------------------------
class Args:
    input_file = 'names.txt'        # your training input file
    work_dir = 'out'
    device = 'mps'                  # or 'cuda' if you have a GPU
    n_layer = 4
    n_head = 4
    n_embd = 64
    n_embd2 = 64
    type = 'transformer'
    top_k = -1
args = Args()

# Load datasets
train_dataset, test_dataset = create_datasets(args.input_file)
vocab_size = train_dataset.get_vocab_size()
block_size = train_dataset.get_output_length()

# Load model
config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                     n_layer=args.n_layer, n_head=args.n_head,
                     n_embd=args.n_embd, n_embd2=args.n_embd2)
model = Transformer(config)
model.to(args.device)

model_path = "model.pt"
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}. Please train and place model.pt in the out directory.")
    sys.exit(1)

model.load_state_dict(torch.load(model_path, map_location=args.device))
model.eval()

# ------------------------------------------------------------
# Define a request handler for our simple HTTP server
# ------------------------------------------------------------
class MakeMoreHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        # Parse the requested URL
        url = urlparse(self.path)
        if url.path == '/makemore':
            # Parse query parameters
            query = parse_qs(url.query)
            if 'next' in query:
                input_text = query['next'][0]
                input_text = unquote(input_text)  # URL decode

                # Encode the input text into indices
                # NOTE: The dataset expects a sequence of single-character tokens.
                #       Ensure that the input_text is in the format expected by your dataset.
                #       For example, if your dataset is character-based, just pass the string.

                # Encode the input sequence
                # The dataset encodes characters to integers.
                # If input_text is empty or doesn't match expected format, handle gracefully.
                if len(input_text) > 0:
                    idx = train_dataset.encode(input_text).unsqueeze(0).to(args.device)
                else:
                    # If empty, start from the start token (0)
                    # However note that in the makemore code, the data loading logic:
                    # The x/y sequences start with a <START> token as 0. If you want to
                    # generate the next token for an empty string, you might start from
                    # just a zero input.
                    idx = torch.zeros((1,1), dtype=torch.long, device=args.device)

                # Generate the next token
                top_k = args.top_k if args.top_k != -1 else None
                new_idx = generate(model, idx, max_new_tokens=1, top_k=top_k, do_sample=False)

                # The last appended token is new_idx[0, -1]
                next_token_id = new_idx[0, -1].item()
                # Decode the token back to a character
                next_char = train_dataset.decode([next_token_id])

                # Send response
                self.send_response(200)
                self.send_header("Content-type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(next_char.encode('utf-8'))
            else:
                # No 'next' parameter provided
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing 'next' query parameter.")
        else:
            # Not found
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found.")

# ------------------------------------------------------------
# Run the server
# ------------------------------------------------------------
if __name__ == '__main__':
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, MakeMoreHandler)
    print("Serving on http://localhost:8000")
    httpd.serve_forever()
