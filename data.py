from orpheus.src.orpheus import OrpheusTrainer, OrpheusDataProcessor
from datasets.features import Audio

# Assuming your dataset has a 'question_audio' column

dataset_name = "amuvarma/flattened-convos-regzuck"

data_processor = OrpheusDataProcessor()

dataset = data_processor.fast_load_dataset(dataset_name)
dataset = dataset.select(range(20))
processed_dataset = data_processor.adapt_stage_1_to_stage_5_dataset(dataset)
processed_dataset = processed_dataset.cast_column('question_audio', Audio())



processed_dataset.push_to_hub("test-ds-kok-proc")

# orpheus = OrpheusTrainer(    
#     stage = "stage_5",
#     dataset = processed_dataset, 
#     model_name = "amuvarma/zuck-3bregconvo-automodelcompat" # pass a 🤗 model or local checkpoint folder)
# )

# orpheus_trainer = orpheus.create_trainer()

# orpheus_trainer.train() # subclasses 🤗 Trainer
