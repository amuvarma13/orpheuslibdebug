from orpheus import OrpheusTrainer

orpheus = OrpheusTrainer()

speech_dataset_name = "amuvarma/stage_1_speech_dataset"
text_dataset_name = "amuvarma/stage_1_text_dataset"

orpheus.initialise(
    stage = "stage_1",
    speech_dataset_name = speech_dataset_name,
    speech_dataset = speech_dataset_name, 
    text_dataset_name = text_dataset_name, # optional, defaults to generic QA dataset for LLM tuning
    use_wandb = True, # optional, defaults to False
    wandb_project_name = None, # optional defaults to "orpheus-stage-1"
    wandb_run_name = None, # optional defaults to "r0"
    model_name = None # optional, defaults to Canopy's pretrained model
)

orpheus_trainer = orpheus.create_trainer() # subclasses Trainer 

orpheus_trainer.train() # pass any additional params Trainer accepts in the X.train(**args)