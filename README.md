# Guidance API: An Extension for oobabooga/text-generation-webui



Guidance API is a powerful extension for oobabooga/text-generation-webui that integrates the feature-rich and easy-to-use interface of OOGA with the robust capabilities of Guidance. By facilitating network calls for Guidance, this API brings out the full potential of modern language models in a streamlined and efficient manner.

## Features 

- **Seamless Integration with oobabooga/text-generation-webui**: Guidance API seamlessly extends the functionalities of OOGA, enriching its feature set while preserving its ease of use.

- **Network Calls with Guidance**: This extension makes network calls to Guidance, enabling you to harness the power of advanced language models conveniently.

- **Rich Output Structure**: With the ability to support multiple generations, selections, conditionals, tool use, and more, Guidance API can create a rich output structure.

- **Smart Generation Caching**: Guidance API optimizes performance and efficiency with smart seed-based generation caching. Tokens are cached on server

- **Compatibility with Role-Based Chat Models**: Coming Soon

Note the "select" tag in guidance is currently WIP

---



## Getting Started 
Example of flags for config in webui.py
```
CMD_FLAGS = " --chat --model-menu  --model decapoda-research_llama-7b-hf --extensions guidance_api"

```

Then in guidance:


```
import guidance
import json,requests
import re,sys

guidance.llm = guidance.llms.TGWUI("http://127.0.0.1:9000")

character_maker = guidance("""The following is a character profile for an RPG game in JSON format.
```json
{
    "id": "{{id}}",
    "description": "{{description}}",
    "name": "{{gen 'name'}}",
    "class": "{{gen 'class'}}",

}```""")

# generate a character
res=character_maker(
    id="e1f491f7-7ab8-4dac-8c20-c92b5e7d883d",
    description="A quick and nimble fighter.",
)

print(res)
```

Feel free to submit feedback, this repository is under active development

