from modules import shared
from pathlib import Path
from modules.text_generation import encode, generate_reply,decode
import yaml


def id_to_token(id):
    return decode(int(id))

def token_to_id(token):
    return encode(token)
def setup_model_data():
    config_dict = vars(shared.model.config)
    data={}


    if shared.settings['instruction_template'] is not None:
        template=shared.settings['instruction_template']

        if template.lower() == 'none':
            data['instruction_following']=False
            data['instruction_template'] = None
        else:
            filepath = Path(f'characters/instruction-following/{template}.yaml')
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    data['instruction_template'] = yaml.safe_load(f)

            data['instruction_following']=True
    else:
        data['instruction_following']= False
        data['instruction_template'] = None
    
    if shared.settings['add_bos_token'] is not None:
        data['add_bos_token']= shared.settings['add_bos_token']
    if shared.settings['ban_eos_token'] is not None:
        data['ban_eos_token']= shared.settings['ban_eos_token']
    data['vocab_size']= shared.tokenizer.vocab_size
    if shared.model_name is not None:
        data['model_name']=shared.model_name
    else:
        data['model_name']="unknown_model_name"

    data["bos_token_id"]=config_dict["bos_token_id"]
    data["eos_token_id"]=config_dict["eos_token_id"]
    data["bos_token"]=id_to_token(data["bos_token_id"])
    data["eos_token"]=id_to_token(data["eos_token_id"])

    if shared.tokenizer.eos_token is not None:
        data["eos_token"]= shared.tokenizer.eos_token
    if shared.tokenizer.bos_token is not None:
        data["bos_token"]= shared.tokenizer.bos_token

    if shared.tokenizer.eos_token_id is not None:
        data["eos_token_id"]= shared.tokenizer.eos_token_id
    if shared.tokenizer.bos_token_id is not None:
        data["bos_token_id"]= shared.tokenizer.bos_token_id

    if "bias" in config_dict:
        data["bias"]= config_dict["bias"]
    if "temperature" in config_dict:
        data["temperature"]= config_dict["temperature"]
    if "top_p" in config_dict:
        data["top_p"]= config_dict["top_p"]
    if "top_k" in config_dict:
        data["top_k"]= config_dict["top_k"]




    if "falcon" in shared.args.model.lower():
        data['eos_token']= None
        data['bos_token']= None
        if "instruct" in shared.args.model.lower():
            data['instruction_template']={
                "user": '>>QUESTION<<',
                "bot": '>>ANSWER<<'
            }

    print(data)
    return data