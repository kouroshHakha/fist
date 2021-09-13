# Hierarchical Few-shot Imitation with Skill Transition Models

## Overview
Official codebase for [FIST](https://arxiv.org/abs/2107.08981). It contains the instructions and scripts for reproducing the experiments. 

## Dataset and envs
We have released the datasets used for kitchen and pointmaze environments. To download them, run `python scripts/download_data.py`. A `./data` folder will get created with all the data that is needed. 

The environments used in the paper are based on [D4RL](https://github.com/rail-berkeley/d4rl) and the extensiosn for kitchen env in the [SPiRL](https://github.com/clvrai/spirl) paper. The environement is also released as part of the repo that can be installed separately. It is recommended to create a separate conda env if you do not want to override your existing d4rl setup.

```
cd d4rl
pip install -e .
```

## Instructions
The experimental setup of FIST has several steps. First the skill prior and contrastive distance models have to be trained using the pretraining data. Then, optionally, they can be finetuned with the demonstration data by loading their pre-trained checkpoints and minimizing the same loss on the demo data for a few more steps (e.g. 20 epochs). 

Each step involves its own combination of script or config file. For minor changes we just modify the config files in-place and run the same script. All the config files are released under `./spirl/configs/`, and the results will be logged in `./experiments` folder.

We have also released the checkpoint files for contrastive encoders and skill prior models. They can be downloaded by running `python scripts/download_ckpts.py`. A `./checkpoints` folder with the same folder structure as `./experiments` will get created. The proper checkpoint file can get passed to the script using either command line arguments or config file entries. 

### Example of training/evaluation from scratch


**Pre-training and Fine-tuning the contrastive distance model:** Both the pre-training and fine-tuning of the contrastive encoders are done with the same script by passing the PT dataset as well as the demo dataset (The demo dataset can be `None` in which case it will skip fine-tuning) 

```
python scripts/train_contrastive_reachability.py --env kitchen --training-set data/kitchen/kitchen-mixed-no-topknob.hdf5 --demos data/kitchen/kitchen-demo-microwave_kettle_topknob_switch.hdf5 --save-dir experiments/contrastive/kitchen/exact-no-topknob
```

> Note: To produce the exact experimental results in the paper we have created one contrastive encoder with both pre-training and demonstration data from all tasks (instead of one for each each downstream task). To get this checkpoint run:
> ```
> python scripts/train_contrastive_reachability.py --env kitchen --training-set data/kitchen/kitchen-mixed-v0.hdf5 --save-dir experiments/contrastive/kitchen/exact-mixed-all
> ```


**Pre-training the goal-conditioned skill VAE:** The configs for pre-training the skill prior are located under `spril/configs/skil_prior_learning`. We pre-train for ~2000 epochs similar to [SPiRL](https://github.com/clvrai/spirl/blob/581db4030989145c32bf0390cd9a1aec0f9cd0dd/spirl/configs/skill_prior_learning/kitchen/hierarchical_cl/conf.py#L16).

```
python spirl/train.py --path spirl/configs/skill_prior_learning/kitchen/hierarchical_cl_gc_no_topknob --val_data_size 1024

```

The following is the naming convention for skill prior learning config files:

```
FIST (goal conditioned): hierarchical_cl_gc_xxx
SPiRL: hierarchical_cl_xxx
```

The tensorboard and checkpoint file will get stored under `experiments/skill_prior_learning/<path_to_config>`.

**Fine-tuning and Evaluating the goal-conditioned VAE:**
The configs for fine-tuning the skill prior and evaluating with the semi-parametric approach are located under `spril/configs/few_shot_imitation_learning`. 

For fine-tuning, the config file should include a checkpoint path referring back to the pre-trained model. The `checkpt_path` keyword under `model_config` dictionary in the config file (`conf.py`) is the intended variable for this. The flag `--resume` selects which checkpoint epoch to use for this. 

Other important parameters in the config file include `fewshot_data`,  `fewshot_batch_size`, and `finetune_vae` which determine the settings for fine-tuning. An example command would look like the following:

```
python scripts/fewshot_kitchen_train.py --path spirl/configs/few_shot_imitation_learning/kitchen/hierarchical_cl_gc_demo_topknob2_finetune_vae/ --resume 199 --val_data_size 160
```


List of fine-tuning configs for kithen env:
| Task (Unseen)                                                  | Performance          | config folder |
|----------------------------------------------------------------|-------------------------|---------------|
| Microwave, Kettle, \textbf{Top Burner}, Light Switch           | $\mathbf{3.6 \pm 0.16}$ | spirl/configs/few_shot_imitation_learning/kitchen/hierarchical_cl_demo_topknob2_finetune_vae |
| \textbf{Microwave}, Bottom Burner, Light Switch, Slide Cabinet | $\mathbf{2.3 \pm 0.5}$  | spirl/configs/few_shot_imitation_learning/kitchen/hierarchical_cl_demo_microwave_finetune_vae |
| Microwave, \textbf{Kettle}, Slide Cabinet, Hinge Cabinet       | $\mathbf{3.5 \pm 0.3}$  | spirl/configs/few_shot_imitation_learning/kitchen/hierarchical_cl_gc_demo_kettle2_finetune_vae |
| Microwave, Kettle, \textbf{Slide Cabinet}, Hinge Cabinet       | $\mathbf{4.0 \pm 0.0}$  | spirl/configs/few_shot_imitation_learning/kitchen/hierarchical_cl_gc_demo_slide_finetune_vae |


List of fine-tuning configs for pointmaze env:
| Section | FIST                  | | config folder|
|------------|--------------------------------------|---|--|
|             | Episode length | SR | |
| Left     | $363.87 \pm 18.73$ |$0.99 \pm 0.03$  | spirl/configs/few_shot_imitation_learning/maze_left/hierarchical_cl_state_gc_4M_B1024_only_demos_contra|
| Right    | $571.21 \pm 38.82$ | $0.91 \pm 0.07$ |spirl/configs/few_shot_imitation_learning/maze_right/hierarchical_cl_state_gc_4M_B1024_only_demos_contra |
| Bottom | $359.82 \pm 3.62$ | $1.0 \pm 0.0$      |spirl/configs/few_shot_imitation_learning/maze_bottom (TODO) | 


For evaluation, we modify the same config file to ignore the pre-training checkpoint path and let the script figure out where to pick-up the fine-tuned model checkpoints. To do so we comment out the `checkpt_path` variable and run the following command on the same config file:

```
python scripts/fewshot_kitchen_train.py --path spirl/configs/few_shot_imitation_learning/kitchen/hierarchical_cl_gc_demo_topknob2_finetune_vae/ --resume weights_ep49 --eval 1
```

Here, `weights_ep49` is referring to the keyword of the checkpoints used in the experiments folder that will get created by running the fine-tuning script. Other important parameters in the config file include `contra_config` and `contra_ckpt` which determines the settings for semi-parameteric lookup. 


> Note: For Maze experiments use `scripts/fewshot_train.py` instead of `scripts/fewshot_kitchen_train.py`. We also found that we do not need to do any fine-tuning for pointmaze. Therefore, we can run the evaluation script by resuming the pre-trained checkpoint directly.

# License 
BSD3


