from orpheus.src.orpheus import OrpheusUtility
orpheus = OrpheusUtility()

checkpoint_name = "checkpoints/checkpoint-3673" # find <TRAINING STEPS> in checkpoints/
push_name = "amuvarma/canopy-tune-stage_4-luna"
orpheus.fast_push_to_hub(checkpoint=checkpoint_name, push_name=push_name)