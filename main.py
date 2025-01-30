from orpheus.mm_model import (
    OrpheusConfig,
    OrpheusForConditionalGeneration,
    OrpheusUtility, 

)
from orpheus.mm_model.assets import SPEECH_WAV_PATH
import librosa
import torch
import torchaudio
orpheus = OrpheusUtility()

from transformers import AutoModel, AutoTokenizer

orpheus.initialise()
model_name = "amuvarma/zuck-3bregconvo-automodelcompat"
model = AutoModel.from_pretrained(model_name).to("cuda").to(torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

orpheus.register_auto_model(model, tokenizer)

conversation = orpheus.initialise_conversation_model() # initialise a new conversation
from orpheus.mm_model.assets import SPEECH_WAV_PATH

import torchaudio
waveform, sample_rate = torchaudio.load(SPEECH_WAV_PATH)

first_message = {
    "format":"speech",
    "data": y
}

conversation.append_message(first_message)

results = conversation.generate_response()
print(results)