from orpheus.src.orpheus import OrpheusTrainer

model_name = "amuvarma/canopy-tune-stage_3-luna" # from stage_2_train.py
#** loading the datasets can take a while, even up to 30 mins **
orpheus = OrpheusTrainer(
    stage = "stage_4",
    model_name = model_name,
    batch_size = 8, # use batch_size * number_of_gpus = 64 for quickest training
)

orpheus_trainer = orpheus.create_trainer( report_to="wandb" ) # subclasses 🤗 Trainer 

orpheus_trainer.train()