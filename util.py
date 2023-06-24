import time
import traceback
from threading import Thread
from typing import Callable, Optional

from modules import shared
from modules.chat import load_character_memoized


def build_parameters(body, chat=False):

    generate_params = {
        'max_new_tokens': int(body.get('max_new_tokens', body.get('max_length', 200))),
        'do_sample': bool(body.get('do_sample', True)),
        'temperature': float(body.get('temperature', 0.5)),
        'top_p': float(body.get('top_p', 1)),
        'typical_p': float(body.get('typical_p', body.get('typical', 1))),
        'epsilon_cutoff': float(body.get('epsilon_cutoff', 0)),
        'eta_cutoff': float(body.get('eta_cutoff', 0)),
        'tfs': float(body.get('tfs', 1)),
        'top_a': float(body.get('top_a', 0)),
        'repetition_penalty': float(body.get('repetition_penalty', body.get('rep_pen', 1.1))),
        'encoder_repetition_penalty': float(body.get('encoder_repetition_penalty', 1.0)),
        'top_k': int(body.get('top_k', 0)),
        'min_length': int(body.get('min_length', 0)),
        'no_repeat_ngram_size': int(body.get('no_repeat_ngram_size', 0)),
        'num_beams': int(body.get('num_beams', 1)),
        'penalty_alpha': float(body.get('penalty_alpha', 0)),
        'length_penalty': float(body.get('length_penalty', 1)),
        'early_stopping': bool(body.get('early_stopping', False)),
        'mirostat_mode': int(body.get('mirostat_mode', 0)),
        'mirostat_tau': float(body.get('mirostat_tau', 5)),
        'mirostat_eta': float(body.get('mirostat_eta', 0.1)),
        'seed': int(body.get('seed', -1)),
        'add_bos_token': bool(body.get('add_bos_token', True)),
        'truncation_length': int(body.get('truncation_length', body.get('max_context_length', 2048))),
        'ban_eos_token': bool(body.get('ban_eos_token', False)),
        'skip_special_tokens': bool(body.get('skip_special_tokens', True)),
        'custom_stopping_strings': '',  # leave this blank
        'stopping_strings': body.get('stopping_strings', []),
    }

    if chat:
        character = body.get('character')
        instruction_template = body.get('instruction_template')
        name1, name2, _, greeting, context, _ = load_character_memoized(character, str(body.get('your_name', shared.settings['name1'])), shared.settings['name2'], instruct=False)
        name1_instruct, name2_instruct, _, _, context_instruct, turn_template = load_character_memoized(instruction_template, '', '', instruct=True)
        generate_params.update({
            'stop_at_newline': bool(body.get('stop_at_newline', shared.settings['stop_at_newline'])),
            'chat_prompt_size': int(body.get('chat_prompt_size', shared.settings['chat_prompt_size'])),
            'chat_generation_attempts': int(body.get('chat_generation_attempts', shared.settings['chat_generation_attempts'])),
            'mode': str(body.get('mode', 'chat')),
            'name1': name1,
            'name2': name2,
            'context': context,
            'greeting': greeting,
            'name1_instruct': name1_instruct,
            'name2_instruct': name2_instruct,
            'context_instruct': context_instruct,
            'turn_template': turn_template,
            'chat-instruct_command': str(body.get('chat-instruct_command', shared.settings['chat-instruct_command'])),
        })

    return generate_params


def try_start_cloudflared(port: int, max_attempts: int = 3, on_start: Optional[Callable[[str], None]] = None):
    Thread(target=_start_cloudflared, args=[
           port, max_attempts, on_start], daemon=True).start()


def _start_cloudflared(port: int, max_attempts: int = 3, on_start: Optional[Callable[[str], None]] = None):
    try:
        from flask_cloudflared import _run_cloudflared
    except ImportError:
        print('You should install flask_cloudflared manually')
        raise Exception(
            'flask_cloudflared not installed. Make sure you installed the requirements.txt for this extension.')

    for _ in range(max_attempts):
        try:
            public_url = _run_cloudflared(port, port + 1)

            if on_start:
                on_start(public_url)

            return
        except Exception:
            traceback.print_exc()
            time.sleep(3)

        raise Exception('Could not start cloudflared.')
