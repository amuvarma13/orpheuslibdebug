from orpheus.src.orpheus import OrpheusUtility
orpheus = OrpheusUtility()

checkpoint_name = "checkpoints/checkpoint-2000" # find <TRAINING STEPS> in checkpoints/
push_name = "amuvarma/canopy-tune-stage_5-luna-2000"
orpheus.fast_push_to_hub(checkpoint=checkpoint_name, push_name=push_name)