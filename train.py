from orpheus.src.orpheus import OrpheusTrainer
import wandb

wandb.init(project="orpheusdeblib", run_name="s1")

speech_dataset_name = "amuvarma/5k-qa-pairs-tttts"
text_dataset_name = "amuvarma/va-320k-330k-snac-no-identity-QA_TTTTS"

model_name = "amuvarma/3b-10m-pretrain-full"
orpheus = OrpheusTrainer(
    stage = "stage_1",
    speech_dataset_name = speech_dataset_name,
    text_dataset_name = text_dataset_name, # optional, defaults to generic QA dataset for LLM tuning
    model_name = model_name # optional, defaults to Canopy's pretrained model
)

print("finished initialising")
orpheus_trainer = orpheus.create_trainer() # subclasses Trainer 

print("created trainer")
orpheus_trainer.train(
    report_to = "wandb",
) # pass any additional params Trainer accepts in the X.train(**args)