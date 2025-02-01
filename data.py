from orpheus.src.orpheus import OrpheusTrainer, OrpheusDataProcessor



dataset_name = "amuvarma/flattened-convos-regzuck"

data_processor = OrpheusDataProcessor()

dataset = data_processor.fast_load_dataset(dataset_name)
dataset = dataset.select(range(100))
processed_dataset = data_processor.adapt_stage_1_to_stage_5_dataset(dataset)

orpheus = OrpheusTrainer(    
    stage = "stage_5",
    dataset = processed_dataset, 
    model_name = "amuvarma/zuck-3bregconvo-automodelcompat" # pass a 🤗 model or local checkpoint folder)
)

orpheus_trainer = orpheus.create_trainer()

orpheus_trainer.train() # subclasses 🤗 Trainer
