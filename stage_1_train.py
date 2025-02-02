from .orpheus.src.orpheus import OrpheusTrainer

#optionally set up wandb for tracking
import wandb #=> pip install wandb
wandb.init(project="orpheusdeblib", name="s1")




speech_dataset_name = "amuvarma/luna-6k"
text_dataset_name = "amuvarma/10k-qa"

model_name = "amuvarma/3b-10m-pretrain-full"


orpheus = OrpheusTrainer(
    stage = "stage_1",
    speech_dataset_name = speech_dataset_name,
    text_dataset_name = text_dataset_name, # optional, defaults to generic QA dataset for LLM tuning
    model_name = model_name # optional, defaults to Canopy's pretrained model
)

orpheus_trainer = orpheus.create_trainer(
  report_to = "wandb" # pass any 🤗 TrainingArgs in here
) 

orpheus_trainer.train() # orpheus_trainer subclasses 🤗 Trainer