from orpheus.src.orpheus import OrpheusTrainer, OrpheusDataProcessor
import wandb


wandb.init(project="orpheusdeblib", name="s4")

dataset_name = "amuvarma/test-ds-kok-proc"
model_name = "amuvarma/zuck-3bregconvo-automodelcompat"

data_processor = OrpheusDataProcessor()
dataset = data_processor.fast_load_dataset(dataset_name)

orpheus = OrpheusTrainer(
    stage = "stage_5",
    model_name = model_name,
    dataset = dataset,
    batch_size = 21, # use batch_size * number_of_gpus = 64 for quickest training
)

print("finished initialising")
orpheus_trainer = orpheus.create_trainer(report_to = "wandb") # pass in any trainer args here

print("created trainer")
orpheus_trainer.train() #inherits from Trainer