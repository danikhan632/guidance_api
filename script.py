import base64
import json
import os,sys
import time
import torch
import requests
import yaml
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from modules.utils import get_available_models
from .guidance_gen import GuidanceGenerator
import numpy as np

import traceback
import unittest
from modules import shared

port=9555
def printc(obj, color):
    color_code = {
        'black': '30', 'red': '34', 'green': '32', 'yellow': '33',
        'blue': '34', 'magenta': '35', 'cyan': '36', 'white': '37'
    }
    colored_text = f"\033[{color_code[color]}m{obj}\033[0m" if color in color_code else obj
    print(colored_text)



class Handler(BaseHTTPRequestHandler):

    def __init__(self, *args, gen=None, **kwargs):
        self.gen = gen
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == '/api/v1/model':
            self.send_response(200)
            self.end_headers()

                
            response = json.dumps({"results":self.gen.data})
            self.wfile.write(response.encode('utf-8'))
        else:
            self.send_error(404)

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = json.loads(self.rfile.read(content_length).decode('utf-8'))

        if self.path == '/api/v1/call':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            printc("Call request received, accuiring generation lock", "green")
            printc(body, "blue")
            res=""
            try:
                res= self.gen.__call__(
                prompt=body["prompt"], stop=body["stop"], stop_regex=body["stop_regex"],
                temperature=body["temperature"], n=body["n"], max_tokens=body["max_tokens"], 
                logprobs=body["logprobs"],top_p=body["top_p"], echo=body["echo"], logit_bias=body["logit_bias"],
                token_healing=body["token_healing"], pattern=body["pattern"],stream=False,cache_seed=-1, 
                caching=False,
                )
                printc(res, "green")
            except Exception as e:
                printc("An error occurred: " + str(e), "red")
            finally:
                print("Call request fulfilled, releasing generation lock")


            response = json.dumps({
                'choices': [{
                    'text': res
                }]
            })

            self.wfile.write(response.encode('utf-8'))

        elif self.path == '/api/v1/encode':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            printc("Encode request received", "green")
            printc(body, "blue")
            string = body['text']
            res=self.gen.encode(string)

            response = json.dumps({
                'results': [{
                    'tokens':res
                }]
            })

            self.wfile.write(response.encode('utf-8'))
        elif self.path == '/api/v1/decode':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            printc("decode request received", "green")
            printc(body, "blue")
            tokens = (body['tokens'])
            print("my tokens",tokens, type(tokens))
            decoded_sequences =self.gen.decode(tokens)
            response = json.dumps({
                'results': [{
                    'ids': decoded_sequences
                }]
            })

            self.wfile.write(response.encode('utf-8'))




def _run_server(port: int, gen):
    

    address = '0.0.0.0' if shared.args.listen else '127.0.0.1'

    class CustomHandler(Handler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, gen=gen, **kwargs)

    server = ThreadingHTTPServer((address, port), CustomHandler)

    def on_start(public_url: str):
        print(f'Starting non-streaming server at public url {public_url}/api')

    server.serve_forever()

def setup():
    printc("starting guidance server","green")
    gen =GuidanceGenerator()

    Thread(target=_run_server, args=[port,gen], daemon=True).start()