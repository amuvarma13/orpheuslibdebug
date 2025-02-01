from orpheus.src.orpheus import OrpheusTrainer
import wandb

wandb.init(project="orpheusdeblib", name="s4")


model_name = "amuvarma/3b-zuckreg-convo-projpretrain"

#
orpheus = OrpheusTrainer(
    stage = "stage_4",
    model_name = model_name,
    batch_size = 21, # use batch_size * number_of_gpus = 64 for quickest training
)

print("finished initialising")
orpheus_trainer = orpheus.create_trainer(report_to = "wandb") # pass in any trainer args here

print("created trainer")
orpheus_trainer.train() #inherits from Trainer