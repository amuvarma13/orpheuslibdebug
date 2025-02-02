from orpheus.src.orpheus import OrpheusUtility
orpheus = OrpheusUtility()

checkpoint_name = "checkpoints/checkpoint-5996" # find <TRAINING STEPS> in checkpoints/
push_name = "canopy-tune-stage_1-luna"
orpheus.fast_push_to_hub(checkpoint=checkpoint_name, push_name=push_name)