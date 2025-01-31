from orpheus.src.orpheus import OrpheusTrainer
import wandb

wandb.init(project="orpheusdeblib", name="s1")

dataset_name = "amuvarma/flattened-convos-regzuck"

model_name = "amuvarma/3b-10m-pretrain-full"

#
orpheus = OrpheusTrainer(
    stage = "stage_3",
    model_name = model_name # optional, defaults to Canopy's pretrained model
)

print("finished initialising")
orpheus_trainer = orpheus.create_trainer(report_to = "wandb") # pass in any trainer args here

print("created trainer")
orpheus_trainer.train() #inherits from Trainer