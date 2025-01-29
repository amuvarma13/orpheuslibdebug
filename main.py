from orpheus.mm_model import (
    OrpheusConfig,
    OrpheusForConditionalGeneration,
    OrpheusUtility
)
import librosa
orpheus = OrpheusUtility()

from transformers import AutoModel, AutoTokenizer, AutoConfig
AutoConfig.register("orpheus", OrpheusConfig)
AutoModel.register(OrpheusConfig, OrpheusForConditionalGeneration)

my_orpheus.fast_download_from_hub() 
model_name = "amuvarma/zuck-3bregconvo-automodelcompat"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)



# EITHER get inputs from text
prompt = "What is an example of a healthy breakfast?"
inputs = orpheus.get_inputs_from_text(prompt)
print(inputs)

#generate response
output_tokens = model.generate(
    **inputs, 
    max_new_tokens=100, 
    repetition_penalty=1.1, 
    temperature=0.7
    )