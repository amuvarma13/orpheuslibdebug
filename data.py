from orpheus import OrpheusTrainer, OrpheusDataProcessor

dataset_name = "amuvarma/flattened-convos-regzuck"

data_processor = OrpheusDataProcessor()

dataset = data_processor.fast_load_dataset(dataset_name)

processed_dataset = data_processor.adapt_stage_1_to_stage_5_dataset(dataset)

print(processed_dataset)
# orpheus = OrpheusTrainer()


# dataset_name = "amuvarma/orpheus_stage_1"





# orpheus.initialise(
#     stage = "stage_5",
#     dataset = processed_dataset, 
#     model_name = "amuvarma/stage-4-tuned-example-model" # pass a 🤗 model or local checkpoint folder
# )

# orpheus_trainer = orpheus.create_trainer() 

# orpheus_trainer.train() # subclasses 🤗 Trainer