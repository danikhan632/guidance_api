from modules import shared
from modules.text_generation import encode, generate_reply,decode
from .util import build_parameters
from typing import Any, Dict, Optional, Callable
import os
import time
import collections
import regex
import pygtrie
import queue
import torch
import threading
import logging
import transformers
from .processor import TokenHealingLogitsProcessor,BiasLogitsProcessor,RegexLogitsProcessor, RegexStoppingCriteria
from .caching import Cache, DiskCache
from .model_info import setup_model_data
def printc(obj, color):
    color_code = {
        'black': '30', 'red': '31', 'green': '32', 'yellow': '33',
        'blue': '34', 'magenta': '35', 'cyan': '36', 'white': '37'
    }
    colored_text = f"\033[{color_code[color]}m{obj}\033[0m" if color in color_code else obj
    print(colored_text)



class GuidanceGenerator:
    llm_name: str = shared.args.model

    def __init__(self):
        super().__init__()
        self.llm_model = shared.model
        self._call_counts = {}
        self.tokenizer = shared.tokenizer
        self.data= setup_model_data()

        self.bos_token= self.data['bos_token']
        self.eos_token= self.data['eos_token']
        self.eos_token_id = self.token_to_id(self.data['eos_token'])
        self.token_healing=True

        self.model_name = shared.args.model
        self.cache = DiskCache(llm_name=self.model_name)
        self.cache.clear()
        self.cache_version=1
        self._past_key_values = None
        self._prefix_cache = []
        self._token_prefix_map = self._build_token_prefix_map()
        self.data['token_prefix_map_length']=len(self._token_prefix_map)
        printc(self.data,"green")

    def id_to_token(self, id):
        return decode(int(id))

    def token_to_id(self, token):
        return encode(token)

    def encode(self, string, as_list=True):
        tmp= None
        if as_list:
            tmp= encode(string).tolist()[0]
        else:
            tmp= encode(string)
        return tmp



    def decode(self, id):
        tmp = decode(id)
        return tmp


    def _build_token_prefix_map(self):
        """ Build a map from token to index.
        """
        printc(("vocab_size: ",self.tokenizer.vocab_size),"cyan")
        token_map = pygtrie.CharTrie()
        for i in range(self.tokenizer.vocab_size):
            s = self.id_to_token(i)
            if s in token_map:
                token_map[s].append(i)
            else:
                token_map[s] = [i]
        return token_map

    def new_string_builder(self, starting_ids=None):
        return TransformersStringBuilder(self.tokenizer,  starting_ids)


    def prefix_matches(self, prefix):
        """ Return the list of tokens that match the given prefix.
        """
        return [v for arr in self._token_prefix_map.values(prefix=prefix) for v in arr]
    def _gen_key(self, args_dict):
        return "_---_".join([str(v) for v in ([args_dict[k] for k in args_dict] + [self.model_name, self.__class__.__name__, self.cache_version])])


    def _cache_params(self, args_dict) -> Dict[str, Any]:
        """get the parameters for generating the cache key"""
        key = self._gen_key(args_dict)
        # if we have non-zero temperature we include the call count in the cache key
        if args_dict.get("temperature", 0) > 0:
            args_dict["call_count"] = self._call_counts.get(key, 0)
            self._call_counts[key] = args_dict["call_count"] + 1
        args_dict["model_name"] = self.model_name
        args_dict["cache_version"] = self.cache_version
        args_dict["class_name"] =self.__class__.__name__
        return args_dict

    def _update_prefix_cache(self, streamer):
        # note what we now have cached and ready for our next call in this session
        if self._past_key_values and len(streamer.generated_sequence) == 1:
            self._prefix_cache = streamer.generated_sequence[0][:self._past_key_values[0][0].shape[-2]] # self._past_key_values is already saved, this just aligns with it

    def _stream_then_save(self, streamer, key, thread):
        list_out = []
        for out in streamer:
            list_out.append(out)
            yield out
        thread.join() # clean up the thread
        self.llm.cache[key] = list_out
        self._update_prefix_cache(streamer)
        self._last_computed_key = key



   
    def __call__(self, prompt, stop=None, stop_regex=None, temperature=None, n=1, max_tokens=1000, logprobs=None,top_p=1.0, echo=False, logit_bias=None, token_healing=None, pattern=None, stream=False,cache_seed=0, caching=None, **generate_kwargs):
        """ Generate a completion of the given prompt.
        """

        args={
             "prompt":prompt, "stop": stop, "stop_regex":stop_regex, "temperature": temperature, "n":n, 
             "max_tokens":max_tokens, "logprobs":logprobs, "top_p":top_p, "echo":echo, "logit_bias":logit_bias, 
             "token_healing":token_healing, "pattern":pattern, "stream":stream, "cache_seed":cache_seed, 
             "caching":caching, "generate_kwargs":generate_kwargs, "model_name": self.model_name, 
             "cache_version":self.cache_version, "class_name":self.__class__.__name__
        }
        cache_params = self._cache_params(args)
        llm_cache = self.cache
        key = llm_cache.create_key(self.model_name, **cache_params)
        if stop is not None:
            if isinstance(stop, str):
                stop_regex = [regex.escape(stop)]
            else:
                stop_regex = [regex.escape(s) for s in stop]
        if isinstance(stop_regex, str):
            stop_regex = [stop_regex]
        if stop_regex is None:
            stop_regex = []
        stop_regex.append(regex.escape(self.eos_token)) # make sure the end of sequence token is always included
        
        input_ids= encode(prompt)
        healed_token_ids = []
        processors = []
        stoppers = []
        coded_prompt = decode(input_ids[0])
        if token_healing:
            healer = TokenHealingLogitsProcessor(self, self.tokenizer.vocab_size, input_ids[0])
            healed_token_ids = healer.healed_token_ids
            if len(healed_token_ids) > 0:
                input_ids = input_ids[:,:-len(healed_token_ids)]
                max_tokens += len(healed_token_ids) 
                processors.append(healer)
        if logit_bias is not None:
            processors.append(BiasLogitsProcessor(self, self.tokenizer.vocab_size-1, logit_bias))

        max_context = shared.settings['max_new_tokens_max']

        if max_tokens + len(input_ids[0]) > max_context:
            max_tokens = max_context - len(input_ids[0])

        prefix_match_len = 0
        if prefix_match_len == len(input_ids[0]):
            prefix_match_len -= 1
            
        #may cause issues 
        if pattern is not None:
            processors.append(RegexLogitsProcessor(pattern, stop_regex, self, self.tokenizer.vocab_size-1, temperature == 0, len(coded_prompt), self.eos_token_id))

        if stop_regex is not None:
            stoppers.append(RegexStoppingCriteria(stop_regex, self, len(coded_prompt)))

        streamer = TransformersStreamer(
            llm=self,
            input_ids=input_ids,
            stop_regex=stop_regex,
            healed_token_ids=healed_token_ids,
            prefix_length=len(coded_prompt),
            string_builder=self.new_string_builder,
            max_new_tokens=max_tokens,
            logprobs=logprobs
        )
        
        generate_args = dict(
            inputs=input_ids,
            temperature=temperature,
            max_new_tokens=max_tokens,
            top_p=top_p,
            pad_token_id=self.llm_model.config.pad_token_id,
            logits_processor=transformers.LogitsProcessorList(processors),
            stopping_criteria=transformers.StoppingCriteriaList(stoppers),
            output_scores=logprobs is not None and logprobs > 0,
            return_dict_in_generate=True,
            **generate_kwargs
        )

        do_sample = True
        if do_sample is True and temperature == 0:
            generate_args["do_sample"] = False
        elif do_sample is False and temperature > 0:
            generate_args["do_sample"] = True

        temperature = 0.005 if args['temperature'] == 0.0 else args['temperature']
        body = {
        'prompt': prompt,
        'max_new_tokens': args['max_tokens'],
        'do_sample': True,
        'temperature': temperature,
        'top_p': args['top_p']
        }

        print(body)
        printc("generating sequence","yellow")
        prompt = body['prompt']
        generate_params = build_parameters(body)
        stopping_strings = generate_params.pop('stopping_strings')
        generate_params['stream'] = False

        generated_sequence = generate_reply(prompt, generate_params, stopping_strings=stopping_strings, is_chat=self.data['instruction_following'])


        answer = ''
        for a in generated_sequence:
            answer = a
            printc(answer,"yellow")
        out = self.encode(answer, as_list=False)
        streamer.put(out)
        self.cache[key] = streamer.__next__()
        self._update_prefix_cache(streamer)

        return llm_cache[key]

        # return answer
    



    def __exit__(self, exc_type, exc_value, traceback):
        """ Restore the model to its original state by removing monkey patches.
        """
        if getattr(self.llm.model_obj, "_orig_prepare_method", None) is not None:
            self.llm.model_obj.prepare_inputs_for_generation = self.llm.model_obj._orig_prepare_method
            del self.llm.model_obj._orig_prepare_method
        if getattr(self.llm.model_obj, "_orig_update_method", None) is not None:
            self.llm.model_obj._update_model_kwargs_for_generation = self.llm.model_obj._orig_update_method
            del self.llm.model_obj._orig_update_method
        return False

# __call__ method
class TransformersStringBuilder():
    """This deals with the complexity of building up a string from tokens bit by bit."""
    def __init__(self, tokenizer, llm, starting_ids=None):

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
    def __init__(self, llm, input_ids, stop_regex, healed_token_ids, prefix_length, string_builder, max_new_tokens, logprobs, timeout=None):
        self.llm = llm
        self.input_ids = input_ids
        self.stop_regex = stop_regex
        self.healed_token_ids = healed_token_ids
        self.logprobs = logprobs
        self.string_builder=string_builder
        self.max_total_tokens = max_new_tokens + len(input_ids[0])
        self.timeout = timeout
        self.str_pos = [prefix_length for i in range(len(self.input_ids))]
        self.out_queue = queue.Queue()
        self.sequence_pos = [len(self.input_ids[0]) for i in range(len(self.input_ids))]
        self.generated_sequence = [[] for i in range(len(self.input_ids))]
        self.display_logprobs = [[] for i in range(len(self.input_ids))]
        self.generated_string = [self.string_builder(input_ids[0]) for i in range(len(self.input_ids))]
        # 
        self.prefix_cache = []

    def put(self, token_obj):
        if isinstance(token_obj, torch.Tensor):
            new_tokens = token_obj
        else:
            new_tokens = token_obj['sequences']

        if isinstance(new_tokens, torch.Tensor):
            new_tokens = new_tokens.cpu()
        
        # if we are given a single sequence, then make it a batch of size 1
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

                    # trim off the stop regex matches if needed
                    found_partial = False
                    stop_text = None
                    if self.stop_regex is not None:# and (finish_reason is None or len(self.input_ids) > 1):
                        stop_regex_obj = [regex.compile(s) for s in self.stop_regex]
                        for s in stop_regex_obj:
                            m = s.search(val, partial=True)
                            if m:
                                span = m.span()
                                if span[1] > span[0]:
                                    if m.partial: # we might be starting a stop sequence, so we can't emit anything yet
                                        found_partial = True
                                        break
                                    else:
                                        stop_text = val[span[0]:span[1]]
                                        stop_pos = min(span[0], stop_pos)
                                        break

                    # record the reason we stopped (if we have stopped)
                    if stop_pos <= len(val):
                        finish_reason = "stop"
                    
                    # emit the data if we are not potentially in the middle of a stop sequence
                    if not found_partial or finish_reason is not None:
                        out["choices"][i] = {
                            "text": val[:stop_pos],
                            "finish_reason": finish_reason,
                            "stop_text": stop_text,
                            "logprobs": {
                                # "token_healing_prefix": self.last_token_str,
                                "top_logprobs": self.display_logprobs[i][self.sequence_pos[i]:]
                            }
                        }
                        self.str_pos[i] = len(self.generated_string[i])
                        put_data = True
                self.sequence_pos[i] = len(self.generated_sequence[i])
        
        if put_data:
            self.out_queue.put(out)

    def end(self):

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


    def _update_prefix_cache(self, streamer):
        # note what we now have cached and ready for our next call in this session
        if self._past_key_values and len(streamer.generated_sequence) == 1:
            self._prefix_cache = streamer.generated_sequence[0][:self._past_key_values[0][0].shape[-2]] 



    @staticmethod
    def role_start(role):
        raise NotImplementedError("In order to use chat role tags you need to use a chat-specific subclass of Transformers for your LLM from guidance.transformers.*!")

