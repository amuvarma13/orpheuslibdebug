from orpheus.mm_model import (
    OrpheusConfig,
    OrpheusForConditionalGeneration,
    OrpheusUtility
)
my_orpheus = OrpheusUtility()

from transformers import AutoModel, AutoTokenizer, AutoConfig
AutoConfig.register("orpheus", OrpheusConfig)
AutoModel.register(OrpheusConfig, OrpheusForConditionalGeneration)

my_orpheus.fast_download_from_hub() 
model_name = "amuvarma/zuck-3bregconvo-automodelcompat"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)