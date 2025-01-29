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

orpheus.fast_download_from_hub() 
model_name = "amuvarma/zuck-3bregconvo-automodelcompat"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# EITHER get inputs from text
prompt = "Okay, so what would be an example of a healthier breakfast option then. Can you tell me?"
inputs = orpheus.get_inputs_from_text(prompt)
print(inputs)

output_tokens = model.generate(
    **inputs, 
    max_new_tokens=2000, 
    repetition_penalty=1.1, 
    temperature=0.7
    )