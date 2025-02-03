from orpheus.src.orpheus import OrpheusTrainer
import wandb 
wandb.init(project="orpheus-luna-stage2", name="r2")

dataset_name = "amuvarma/luna-convos"
model_name = "amuvarma/canopy-tune-stage_1-luna"

orpheus = OrpheusTrainer(
    stage = "stage_2",
    dataset_name = dataset_name,
    model_name = model_name # optional, defaults to Canopy's pretrained model
)

orpheus_trainer = orpheus.create_trainer(report_to = "wandb") # pass any 🤗 TrainingArgs in here

orpheus_trainer.train()  # orpheus_trainer subclasses 🤗 Trainer