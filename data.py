from orpheus.src.orpheus import OrpheusTrainer, OrpheusDataProcessor
# import wandb


# wandb.init(project="orpheusdeblib", name="s5")

dataset_name = "amuvarma/test-ds-kok-proc-1"
model_name = "amuvarma/zuck-3bregconvo-automodelcompat"

data_processor = OrpheusDataProcessor()
dataset = data_processor.fast_load_dataset(dataset_name)
# dataset = dataset.select(range(200))

# processed_dataset = data_processor.adapt_stage_1_to_stage_5_dataset(dataset)

# processed_dataset.push_to_hub("amuvarma/test-ds-kok-proc-1")

orpheus = OrpheusTrainer(
    stage = "stage_5",
    model_name = model_name,
    dataset = dataset,
    batch_size = 21, # use batch_size * number_of_gpus = 64 for quickest training
)

# print("finished initialising")
orpheus_trainer = orpheus.create_trainer(report_to = "wandb") # pass in any trainer args here

# print("created trainer")
orpheus_trainer.train() #inherits from Trainer