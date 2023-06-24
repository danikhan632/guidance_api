import os
import time
import collections
import regex
import pygtrie
import queue
import threading
import logging
import collections.abc
from modules import shared
from modules.text_generation import encode, generate_reply, decode
from typing import Any, Dict, Optional, Callable


class TransformersStringBuilder():
    """This deals with the complexity of building up a string from tokens bit by bit."""
    def __init__(self, tokenizer, starting_ids=None):
        self.tokenizer = tokenizer
        self.token_strings = []
        self._joint_string = ""
        if starting_ids is not None:
            self.extend(starting_ids)

    def extend(self, new_ids):
        new_token_strings = self.tokenizer.convert_ids_to_tokens(new_ids)
        self.token_strings.extend(new_token_strings)
        new_str = self.tokenizer.convert_tokens_to_string(self.token_strings)
        diff_str = new_str[len(self._joint_string):]
        self._joint_string = new_str
        return diff_str

    def pop(self):
        """Remove the last token from the string and return text it removed."""
        self.token_strings.pop()
        new_str = self.tokenizer.convert_tokens_to_string(self.token_strings)
        diff_str = self._joint_string[len(new_str):]
        self._joint_string = new_str
        return diff_str

    def __str__(self):
        return self._joint_string

    def __len__(self):
        return len(self._joint_string)

class TransformersStreamer():
    def __init__(self, input_ids, stop_regex, healed_token_ids, prefix_length, llm, max_new_tokens, logprobs, timeout=None):

        self.input_ids = input_ids
        self.stop_regex = stop_regex
        self.healed_token_ids = healed_token_ids
        print(logprobs)
        self.logprobs = logprobs
        self.llm = llm
        self.max_total_tokens = max_new_tokens + len(input_ids[0])
        self.timeout = timeout
        self.str_pos = [prefix_length for i in range(len(self.input_ids))]
        self.out_queue = queue.Queue()
        self.sequence_pos = [len(self.input_ids[0]) for i in range(len(self.input_ids))]
        self.generated_sequence = [[] for i in range(len(self.input_ids))]
        self.display_logprobs = [[] for i in range(len(self.input_ids))]
        self.generated_string = [self.llm.new_string_builder(starting_ids=input_ids[0]) for i in range(len(self.input_ids))]
        self.prefix_cache = []

    def put(self, token_obj):
        print(self.display_logprobs)
        import torch
        if isinstance(token_obj, torch.Tensor):
            new_tokens = token_obj
        else:
            new_tokens = token_obj['sequences']

        if isinstance(new_tokens, torch.Tensor):
            new_tokens = new_tokens.cpu()

        # if we are given a single sequence, then make itstop=', a batch of size 1
        if len(new_tokens.shape) == 1:
            new_tokens = new_tokens.unsqueeze(0)

        # extract the scores if we are given them (and format them to be the same shape as the tokens)
        if self.logprobs:
            assert len(new_tokens) == 1, "logprobs are not supported for batched generation right now in guidance.llms.Transformers"
            new_scores = [torch.nn.functional.log_softmax(x, dim=-1).cpu() for x in token_obj['scores']]
            len_diff = len(new_tokens[0]) - len(new_scores)
            if len_diff > 0:
                new_scores = [None for i in range(len_diff)] + new_scores
            new_scores = [new_scores]

        out = {"choices": [None for i in range(len(self.input_ids))]}
        put_data = False
        for i in range(len(self.input_ids)):
            self.generated_sequence[i].extend(list(new_tokens[i]))

            # save logprobs if needed
            if self.logprobs:
                for scores in new_scores[i]:
                    if scores is None:
                        self.display_logprobs[i].append(None)
                    else:
                        top_inds = scores[0].argsort(descending=True)[:self.logprobs] # TODO: verify the [0] is always correct
                        self.display_logprobs[i].append({self.llm.id_to_token(j): float(scores[0][j]) for j in top_inds})

            if self.sequence_pos[i] < len(self.generated_sequence[i]):
                display_tokens = list(self.generated_sequence[i][self.sequence_pos[i]:])
                val = self.generated_string[i].extend(display_tokens)

                if self.str_pos[i] < len(self.generated_string[i]):
                    val = str(self.generated_string[i])[self.str_pos[i]:]
                    finish_reason = None

                    # check why we stopped
                    stop_pos = len(val) + 1
                    if len(self.generated_sequence[i]) >= self.max_total_tokens:
                        finish_reason = "length"
                    elif self.generated_sequence[i][-1] == self.llm.tokenizer.eos_token_id:
                        finish_reason = "endoftext"
                        eos_str = self.generated_string[i].pop() # remove the end of text token
                        stop_pos = len(val) - len(eos_str)

                    # record the reason we stopped (if we have stopped)
                    if finish_reason is not None:
                        out["choices"][i] = {
                            "text": val[:stop_pos],
                            "finish_reason": finish_reason,
                            "stop_text": None,  # no stop text since stop is None
                            "logprobs": {
                                "top_logprobs": self.display_logprobs[i][self.sequence_pos[i]:]
                            }
                        }
                        self.str_pos[i] = len(self.generated_string[i])
                        put_data = True
                self.sequence_pos[i] = len(self.generated_sequence[i])

        if put_data:
            self.out_queue.put(out)


    def end(self):
        # make sure we have flushed all of the data
        for i in range(len(self.input_ids)):
            assert self.str_pos[i] >= len(self.generated_string[i]), "Not all data was flushed, this means generation stopped for an unknown reason!"

        self.out_queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.out_queue.get(timeout=self.timeout)
        if value is None:
            raise StopIteration()
        else:
            return value
