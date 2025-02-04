from orpheus.src.orpheus import OrpheusTrainer, OrpheusDataProcessor
import wandb

wandb.init(project="orpheus-luna-stage5", name="r5")

data_processor = OrpheusDataProcessor()

speech_dataset_name = "amuvarma/canopy-tune-stage_5-luna-snacced"

processed_dataset = data_processor.fast_load_dataset(speech_dataset_name)

# dataset = dataset.select(range(100))

# print("processed dataset",dataset)

# processed_dataset = data_processor.adapt_stage_1_to_stage_5_dataset(dataset)

# print("adapted dataset",processed_dataset)


# processed_dataset.push_to_hub("amuvarma/canopy-tune-stage_5-luna")

# processed_dataset = data_processor.fast_load_dataset("amuvarma/canopy-tune-stage_5-luna-deb")

orpheus = OrpheusTrainer(    
    stage = "stage_5",
    dataset = processed_dataset, 
    processed_dataset = True,
    model_name = "amuvarma/canopy-tune-stage_4-luna" # pass a 🤗 model or local checkpoint folder)
)

orpheus_trainer = orpheus.create_trainer() 

orpheus_trainer.train() # subclasses 🤗 Trainer