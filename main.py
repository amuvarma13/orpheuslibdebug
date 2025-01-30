from orpheus.mm_model import (
    OrpheusConfig,
    OrpheusForConditionalGeneration,
    OrpheusUtility, 

)
from orpheus.mm_model.assets import SPEECH_WAV_PATH
import librosa
import torch
orpheus = OrpheusUtility()

# from transformers import AutoModel, AutoTokenizer, AutoConfig
# AutoConfig.register("orpheus", OrpheusConfig)
# AutoModel.register(OrpheusConfig, OrpheusForConditionalGeneration)

# orpheus.initialise()
# model_name = "amuvarma/zuck-3bregconvo-automodelcompat"
# model = AutoModel.from_pretrained(model_name).to("cuda").to(torch.bfloat16)
# tokenizer = AutoTokenizer.from_pretrained(model_name)


print(SPEECH_WAV_PATH)
y, sr = librosa.load(SPEECH_WAV_PATH)

embeds = orpheus.get_inputs(speech=y)
print(embeds)

# EITHER get inputs from text
# prompt = "Okay, so what would be an example of a healthier breakfast option then. Can you tell me?"
# inputs = orpheus.get_inputs(text=prompt)

# output_tokens = model.generate(
#     **inputs, 
#     max_new_tokens=200, 
#     repetition_penalty=1.1, 
#     temperature=0.7
#     )

# print(tokenizer.decode(output_tokens[0], skip_special_tokens=True))