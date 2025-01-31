from orpheus.src.orpheus import OrpheusTrainer
import wandb

wandb.init(project="orpheusdeblib", name="s1")

speech_dataset_name = "amuvarma/5k-qa-pairs-tttts"
text_dataset_name = "amuvarma/zuck-nopunc-text"

model_name = "amuvarma/3b-10m-pretrain-full"
orpheus = OrpheusTrainer(
    stage = "stage_1",
    speech_dataset_name = speech_dataset_name,
    text_dataset_name = text_dataset_name, # optional, defaults to generic QA dataset for LLM tuning
    model_name = model_name # optional, defaults to Canopy's pretrained model
)

print("finished initialising")
orpheus_trainer = orpheus.create_trainer(report_to = "wandb",) # pass in any trainer args here

print("created trainer")
orpheus_trainer.train() #inherits from Trainer