# Few-shot IL

## Update log

05/20:
- Added bunch of experience config files for kitchen environment
- Minor modifications to evaluation of kitchen env
- Added two baselines: 1. normal behavioral cloning 2. goal conditioned behavioral cloning 

Let's look at an example from each:


#### BC
To train the model for 200 epochs on the kitchen dataset:
```
python3 spirl/fewshot_bc_train.py --path spirl/configs/bc_atomic/kitchen/offline_data/no-kettle --val_data_size 160
```

Pay attention to this part in the config file:
```
configuration = {
  ...
  'num_epochs': 200,
  'offline_data': True
}
```

Then you need to finetune it on the few-shot data for say 50 epochs:

```
python3 spirl/fewshot_bc_train.py --path spirl/configs/bc_atomic/kitchen/finetuned/kettle_excluded_demo_microwave_kettle_hinge_slide --val_data_size 160 --resume 199 
```

Pay attention to these parts in the config file, the config file is basically the same as the offline data one with some minor changes:

```
# make sure offline data is False, this will grab the fewshot data as the training data
configuration = {
  ...
  'num_epochs': 50,
  'offline_data': False
}

# make sure your checkpoint from pretrained model is loaded (via --resume 199 from cmd line)
bc_model = AttrDict(
    ...
    checkpt_path=f'{os.environ["EXP_DIR"]}/bc_atomic/kitchen/offline_data/no-kettle',
)
```

After training you have to evaluate, you open the same config file (e.g. `spirl/configs/bc_atomic/kitchen/finetuned/kettle_excluded_demo_microwave_kettle_hinge_slide`)
and comment out the checkpoint file path that goes back to the pre-trained model
```
# make sure for evaluation this is commented out.
bc_model = AttrDict(
    ...
    #checkpt_path=f'{os.environ["EXP_DIR"]}/bc_atomic/kitchen/offline_data/no-kettle',
)
```

Then you run the same script, but this time you load the checkpoint from the finetuned model and also put the the script in evaluation mode:
```
python3 spirl/fewshot_bc_train.py --path spirl/configs/bc_atomic/kitchen/finetuned/kettle_excluded_demo_microwave_kettle_hinge_slide --val_data_size 160 --resume 49 --eval 1
```

You need to write a new eval function in other environments similar to what we had for FIST already.

#### Goal-BC
This is very similar to BC, the difference is that in the config files you should change the bc_model type from `BCMdl` to `GoalBCMdl`.
Look at the config files for examples.


05/10:
- `contrastive_reachability.py`: demo file can now be not given.
- `d4rl.kitchen_env` is updated to account for order of tasks and accepting list of tasks as kwarg.
- fixed bugs in `spirl/components/fsil.py`
- fixed bugs in `spirl/data/kitchen/src/kitchen_data_loader.py`
- look at `spirl/configs/few_shot_imitation_learning/maze/hierarchical_cl_state_gc_4M_B1024_only_demos_contra/conf.py` for updated parameters.
- implemented `eval()` in `spirl/fewshot_kitchen_train.py` for kitchen env.
- fixed bugs in `spirl/fewshot_train.py` regarding the finetuning
- in `spirl/fewshot_kitchen_train.py` you can now choose to finetune the entire vae + skill prior or just the skill prior alone
- Added several experiment config files for kitchen env according to [this spredsheet](https://docs.google.com/spreadsheets/d/1uLSH7uiFf-_8gX1csu-HDk3H1Q7meofVoiXUz_n1XOE/edit#gid=0)
- For reproducing kitchen environment experiments the following code skeletons should be used:

For finetuning, ckpt_path should be given as the latest checkpoint of the skill extractor run and then run 
```
CUDA_VISIBLE_DEVICES=0 python3 spirl/fewshot_kitchen_train.py --path <PATH> --val_data_size 160 --resume 199
```
For evaluation on finetuned version, comment out the ckpt path and run:
```
CUDA_VISIBLE_DEVICES=0 python3 spirl/fewshot_kitchen_train.py --path <PATH> --val_data_size 160 --resume 49 --eval 1
```

For evaluation on the none finetuned version, keep the ckpt_path and run:
```
CUDA_VISIBLE_DEVICES=0 python3 spirl/fewshot_kitchen_train.py --path <PATH> --val_data_size 160 --resume 199 --eval 1
```

05/04 (part 2):
- Created `fewshot_train.py` to create a custom training loop on top of SPiRL.
- Some other data structures were added for fewshot_data. See `spirl/components/fsil.py`
- few shot imitation configs has been modified to reflect the new script requirements. See `spirl/spirl/configs/few_shot_imitation_learning/maze/hierarchical_cl_state_4M_B1024_only_demos/conf.py`
- Updated instruction on reproducing the results:
Collecting training data and training skills (VAE) is the same as before. 
  You will also need to source `.bashrc`.
  
This is what you need to run for fine-tuning to downstream demos. `--val_data_size` is 
only required because I was lazy to remove the irrelevant codes. 
```bash
CUDA_VISIBLE_DEVICES=0 python spirl/fewshot_train.py \
--path=spirl/configs/few_shot_imitation_learning/maze/hierarchical_cl_state_gc_4M_B1024_only_demos/ \
--val_data_size=160 --resume 199
```

For running the evaluation, we use the same `conf.py` with the caveat that its checkpoint path 
should be commented out and passed through command line interface.

```bash
CUDA_VISIBLE_DEVICES=0 python spirl/fewshot_train.py \
--path=spirl/configs/few_shot_imitation_learning/maze/hierarchical_cl_state_gc_4M_B1024_only_demos/ \
--val_data_size=160 --resume weights_ep9 --eval 1
```

This command will create a video folder in the corresponding experiment folder that has an example 
video rollout and a `summary.yaml` which summarizes the evaluation performance.

05/04:
- use `os.environ[‘EXP_DIR’]` to customize experiment directory -> look at `spirl/configs/few_shot_imitation_learning/maze/hierarchical_cl_state_4M_B1024/conf.py` and `.bashrc`
- Avoid regenerating demos every time, just generate them once and save them in file, this will make sure experiments are consistent against baselines —> look at `test/test_get_demos.py` and the method `get_demo_from_file in `spirl.maze_few_demo`
- Added a new SPiRL model to test COM conditioning: look at `spirl/configs/skill_prior_learning/maze/hierarchical_cl_state_gc_com_4M_B1024/conf.py`
- Evaluation is now done by running the `spirl/train.py` with `--eval 1` flag instead of commenting/uncommenting the snippets of code.
- Evaluation is now done using a predefined list of rst_points. 
- New experiment files are added.

## Data
Data for demos is located [here](https://drive.google.com/drive/folders/11WEYuwkKOwihRohabP3y97Ze4dzhiGeW?usp=sharing).
After downloading put it under the correct place e.g. `./data`.

## Sample commands

Collect train-time demonstrations on mazes with masked regions. (We already have a few in this folder).
```
python d4rl/scripts/generation/generate_maze2d_datasets.py \
--noisy --env_name maze2d-large-blr-v1 \
--num_samples 4000000
```

Run goal/future conditioned SPiRL on mazes with masked regions
```
python spirl/train.py \
--path=spirl/configs/skill_prior_learning/maze/hierarchical_cl_state_4M_B1024 \
--val_data_size=1024
```

Fine-tune the learned skills to test-time demonstrations
```
python spirl/train.py \
--path=spirl/configs/few_shot_imitation_learning/maze/hierarchical_cl_state_gc_4M_B1024 \
--val_data_size=4096 --resume 199
```

We have some code for rendering rollouts in train.py line 99-152.
```
python spirl/train.py \
--path=spirl/configs/fsil_visualization/maze/hierarchical_cl_state_4M_B1024 \
--val_data_size=1024 --resume 219
```