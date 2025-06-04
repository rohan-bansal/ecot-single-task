# ECoT Single Task

You can change which task to train on by specifying the language instruction on line 43 in `prismatic/vla/datasets/rlds/dataset.py`.

`run_ecot.sh` will launch a training job with 8 a40s, but this is configurable. Remember to change world size, global batch size, and per-device batch size in `prismatic/conf/vla.py`. 

Initial experiment is with 200 epochs but currently observing successful training in only 50-75, this can be configured by `max_steps` variable in above `vla.py` config.