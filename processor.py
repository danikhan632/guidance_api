import os
import time
import collections
import regex
import pygtrie
import queue
import threading
import logging
# __call__ method


class TokenHealingLogitsProcessor():
    """ Token healing.

    When we tokenize the prompt the last token(s) we get are not the last token(s) we would
    have gotten if the prompt + generation was concatented and then tokenized. This
    is not good because it does not align with the pretraining of the model, so
    we "heal" this boundary by backing up as many tokens as needed and then forcing the first tokens
    generated to start with the prefix of the tokens we removed from the prompt. This could
    result in the same tokens at the end of the prompt, or some suffix of the tokens we removed
    could be replaced by a single longer one that crosses the prompt boundary.
    """

    def __init__(self, model, vocab_size, prompt_ids, bias_value=100.):
        """ Build a new TokenHealingLogitsProcessor.

        Note that bias_value is in score space (log-odds normally) and should be
        enough to ensure those tokens are the only ones used.
        """

        # loop backwards through the prompt tokens looking for places where there are possible
        # extensions that cross the prompt boundary
        self.model=model
        prefix_str = ""
        self.extension_tokens = []
        for i in range(len(prompt_ids)-1, max(len(prompt_ids)-10, -1), -1):
            token_str = model.id_to_token(prompt_ids[i])
            prefix_str = token_str + prefix_str
            try:
                extensions = model.prefix_matches(prefix_str)
            except KeyError: # this must be a special token outside the vocab, so we assume it does not have any valid extensions
                extensions = []
            self.extension_tokens.append(extensions)
            if i != len(prompt_ids)-1:
                self.extension_tokens[-1].append(prompt_ids[i]) # add the token used in the input prompt to the list of possible extensions
        self.extension_tokens = self.extension_tokens[::-1]

        # prune off any extension token positions that don't have multiple multiple possible extensions
        found_extensions = False
        for i in range(len(self.extension_tokens)):
            if len(self.extension_tokens[i]) > 1:
                self.extension_tokens = self.extension_tokens[i:]
                found_extensions = True
                break
        if found_extensions:
            self.healed_token_ids = prompt_ids[len(prompt_ids)-len(self.extension_tokens):]
        else:
            self.extension_tokens = []
            self.healed_token_ids = []
        
        # if we have multiple possible completions past the last token, then biasing is needed
        if len(self.extension_tokens) > 0:
            import torch

            # build a set of masks for each possible extension position
            self.token_masks = []
            for i in range(len(self.extension_tokens)):
                token_mask = torch.zeros(vocab_size)
                token_mask.scatter_(0, torch.tensor(self.extension_tokens[i]), bias_value)
                self.token_masks.append(token_mask)

        self.num_extensions = 0

    def __call__(self, input_ids, scores):

        # we only bias the first token generated
        if self.num_extensions >= len(self.extension_tokens):
            return scores
        self.num_extensions += 1

        # check if the last token was from the original prompt (if not then we have already "healed" by choosing a token that crosses the prompt boundary)
        if self.num_extensions > 1 and input_ids[0][-1] != self.healed_token_ids[self.num_extensions-2]:
            return scores

        # handle list inputs
        if isinstance(scores, list):
            import torch
            scores = torch.tensor(scores)

        # make only allowed tokens possible
        # Check size mismatch and correct
        if scores.shape[1] != self.token_masks[self.num_extensions-1].shape[0]:
            scores = scores[:, :-1]

        token_mask = self.token_masks[self.num_extensions-1].to(scores.device)
        
        res = (scores + token_mask )
        # dg=(res).tolist()


        return res
# __call__ method 
class BiasLogitsProcessor():
    """ Simple token biasing.
    """

    def __init__(self, model, vocab_size, logit_bias):
        """ Build a new BiasLogitsProcessor.
        """
        import torch
        
        self.bias_vector = torch.zeros(vocab_size)
        for token, bias in logit_bias.items():
            self.bias_vector[token] = bias
        self.bias_vector = self.bias_vector.to(model.device)

    def __call__(self, input_ids, scores):

        # handle list inputs
        if isinstance(scores, list):
            import torch
            scores = torch.tensor(scores)

        return scores + self.bias_vector


# __call__ method
class RegexLogitsProcessor():
    """ Pattern guiding.
    
    Guide generation to match a regular expression.
    TODO: currently slow, could be made much faster by doing rejection sampling inline with the sampling/greedy process.
    """

    def __init__(self, pattern, stop_regex, llm, vocab_size, is_greedy, prefix_length, eos_token_id, max_consider=500000):
        """ Build a new TokenHealingLogitsProcessor.

        Parameters
        ----------
        pattern : str
            The regex pattern we are seeking to match.
        stop_regex : str or list of str
            The stop regex(s) allowed to come after this pattern.
        llm : function
            The llm.
        vocab_size : int
            The size of the vocabulary.
        is_greedy : bool
            The token selection mode currently in use. We need to know this so we can
            effectively take over that sampling process inside this logit processor.
        eos_token_id : int
            The end of the stop token of the model.
        max_consider : int
            How many top values to bias. Note that we could remove this option once this
            processor is performance optimized (by integrating it into the sampling/greedy process).
        """
        import torch
        
        if isinstance(stop_regex, str):
            stop_regex = [stop_regex]
        self.pattern_no_stop = regex.compile(pattern)
        self.pattern = regex.compile(pattern + "(" + "|".join(stop_regex) + ")?")
        self.llm = llm
        self.is_greedy = is_greedy
        self.prefix_length = prefix_length
        self.max_consider = max_consider
        self.bias_vector = torch.zeros(vocab_size)
        self.current_strings = None
        self.current_length = 0
        self.forced_chars = 0
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids, scores):
        import torch

        # handle 1D inputs
        one_dim = False
        if not isinstance(input_ids[0], collections.abc.Sequence) and not (hasattr(input_ids[0], "shape") and len(input_ids[0].shape) > 0):
            one_dim = True
            input_ids = torch.tensor(input_ids).unsqueeze(0)
            scores = torch.tensor(scores).unsqueeze(0)

        # extend our current strings
        if self.current_strings is None:
            self.current_strings = [self.llm.new_string_builder() for i in range(len(input_ids))]
        for i in range(len(self.current_strings)):
            self.current_strings[i].extend(input_ids[i][self.current_length:])

        assert len(self.current_strings) == 1, "Regex patterns guides do not support batched inference with Transformers yet!"

        self.current_length = len(input_ids[0])
        
        # compute the bias values
        self.bias_vector[:] = 0
        sort_inds = torch.argsort(scores, 1, True)
        to_bias = []
        for i in range(min(sort_inds.shape[1], self.max_consider)):
            self.current_strings[0].extend([sort_inds[0,i]])
            proposed_string = str(self.current_strings[0])[self.prefix_length:]
            self.current_strings[0].pop()
            m = self.pattern.fullmatch(proposed_string, partial=True) # partial means we don't match currently but might as the string grows
            if m:
                to_bias.append(int(sort_inds[0, i]))
                if self.is_greedy: # TODO: make this much faster for non-greedy sampling (by tracking how much prob mass we have looked through perhaps...)
                    break # we are done if we are doing greedy sampling and we found the top valid hit
        
        # if we found no more valid tokens then we just end the sequence
        if not len(to_bias):
            to_bias = [self.eos_token_id]
        
        # bias allowed tokens
        min_to_bias = float(scores[0, to_bias].min())
        bias_value = scores[0, sort_inds[0, 0]] - min_to_bias + 10 # make sure the tokens that fit the pattern have higher scores than the top value
        for x in to_bias:
            self.bias_vector[x] = bias_value
        out = scores + self.bias_vector.to(scores.device)
        if one_dim:
            return out[0]
        else:
            return out
# __call__ method
class RegexStoppingCriteria():
    def __init__(self, stop_pattern, llm, prefix_length):
        if isinstance(stop_pattern, str):
            self.stop_patterns = [regex.compile(stop_pattern)]
        else:
            self.stop_patterns = [regex.compile(pattern) for pattern in stop_pattern]
        self.prefix_length = prefix_length
        self.llm = llm
        self.current_strings = None
        self.current_length = 0

    def __call__(self, input_ids, scores, **kwargs):

        # handle 1D inputs
        if not isinstance(input_ids[0], collections.abc.Sequence) and not (hasattr(input_ids[0], "shape") and len(input_ids[0].shape) > 0):
            input_ids = [input_ids]

        # extend our current strings
        if self.current_strings is None:
            self.current_strings = [self.llm.new_string_builder() for _ in range(len(input_ids))]
        for i in range(len(self.current_strings)):
            self.current_strings[i].extend(input_ids[i][self.current_length:])
        
        self.current_length = len(input_ids[0])
        
        # check if all of the strings match a stop string (and hence we can stop the batch inference)
        all_done = True
        for i in range(len(self.current_strings)):
            found = False
            print(self.current_strings)
            for s in self.stop_patterns:
                
                if s.search(str(self.current_strings[i])[self.prefix_length:]):
                    found = True
            if not found:
                all_done = False
                break
        
        return all_done