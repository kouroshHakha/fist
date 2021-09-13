# Hierarchical Few-shot Imitation with Skill Transition Models

## Overview
Official codebase for [FIST](). It contains the instructions and scripts to reproduce the experiments. 

## Dataset and envs
We have released the datasets used for kitchen and pointmaze environments. To download them, run `python scripts/download_data.py`. A `./data` folder will get created with all the data that is needed. 

The environments used in the paper are based on [D4RL]() and the extensiosn for kitchen env in the [SPiRL]() paper. The environement is also released as part of the repo that can be installed separately. It is recommended to create a separate conda env if you do not want to override your existing d4rl setup.

```
cd d4rl
pip install -e .
```

## Instructions
The experimental setup of FIST has several steps. First the skill prior and contrastive distance models have to be trained using the pretraining data. Then, optionally, they can be finetuned with the demonstration data by loading their pre-trained checkpoints and minimizing the same loss on the demo data for a few more steps (e.g. 20 epochs). 

Each step involves its own combination of script or config file. For minor changes we just modify the config files in-place and run the same script. All the config files are released under `./spirl/configs/`. 

We have also released the checkpoint files for contrastive and skill prior models. They can be downloaded by running `python scripts/download_ckpts.py`. A `./checkpoints` folder with the same folder structure as `./experiments` will get created. The proper checkpoint file can get passed to the script using either command line arguments or config file entries. 

### Example of training/evaluation from scratch


**Pre-training and Fine-tuning the contrastive distance model:** Both the pre-training and fine-tuning of the contrastive encoders are done the same script by passing the PT dataset as well as the demo dataset (The demo dataset can be `None` in which case it will skip fine-tuning) 

```
python scripts/train_contrastive_reachability.py --env kitchen --training-set data/kitchen/kitchen-mixed-no-topknob.hdf5 --demos data/kitchen/kitchen-demo-microwave_kettle_topknob_switch.hdf5 --save-dir experiments/contrastive/kitchen/exact-no-topknob
```

> Note: To produce the exact experimental results in the paper we have created one contrastive encoder with both pre-training and demonstration data from all tasks (instead of one for each each downstream task). To get this checkpoint run:
> ```
> python scripts/train_contrastive_reachability.py --env kitchen --training-set data/kitchen/kitchen-mixed-v0.hdf5 --save-dir experiments/contrastive/kitchen/exact-mixed-all
> ```


**Pre-training the goal-conditioned skill VAE:** The configs for pre-training the skill prior are located under `spril/configs/skil_prior_learning`. We pre-train for ~2000 epochs similar to [SPiRL]().

```
python spirl/train.py --path spirl/configs/skill_prior_learning/kitchen/hierarchical_cl_gc_no_topknob --val_data_size 1024

```

Naming convention:

```
FIST (goal conditioned): hierarchical_cl_gc_xxx
SPiRL: hierarchical_cl_xxx
```

The tensorboard and checkpoint file will get stored under `experiments/skill_prior_learning/<path_to_config>`

**Fine-tuning and Evaluating the goal-conditioned VAE:**
The configs for fine-tuning the skill prior the semi-parametric evaluation are located under `spril/configs/few_shot_imitation_learning`. 

For fine-tuning, the config file should include a checkpoint path referring back to the pre-trained checkpoint path. The `checkpt_path` keyword under `model_config` dictionary in the config file (`conf.py`) is the intended variable for this. The flag `--resume` selects which checkpoint epoch to use for this. 

Other important parameters in the config file include `fewshot_data`,  `fewshot_batch_size`, and `finetune_vae` which determine the settings for fine-tuning. An example command would look like the following:

```
python scripts/fewshot_kitchen_train.py --path spirl/configs/few_shot_imitation_learning/kitchen/hierarchical_cl_gc_demo_topknob2_finetune_vae/ --resume 199 --val_data_size 160
```

For evaluation, we modify the same config file to ignore the pre-training checkpoint path and let the script figure out where to pick-up the fine-tuned model checkpoints. To do so we comment out the `checkpt_path` variable and run the following command on the same config file:

```
python scripts/fewshot_kitchen_train.py --path spirl/configs/few_shot_imitation_learning/kitchen/hierarchical_cl_gc_demo_topknob2_finetune_vae/ --resume weights_ep49 --eval 1
```

Here, `weights_ep49` is referring to the keyword of the checkpoints used in the experiments folder that will get created by running the fine-tuning script. Other important parameters in the config file include `contra_config` and `contra_ckpt` which determines the settings for semi-parameteric lookup. 


> Note: For Maze experiments use `scripts/fewshot_train.py` instead of `scripts/fewshot_kitchen_train.py`. We also found that we do not need to do any fine-tuning for pointmaze. Therefore, we can run the evaluation script by resuming the pre-trained checkpoint directly.

# License 
BSD3


