import os
import numpy as np

from spirl.models.closed_loop_spirl_mdl import GoalClSPiRLMdl
from spirl.models.skill_prior_mdl import SkillSpaceLogger
from spirl.utils.general_utils import AttrDict
from spirl.configs.default_data_configs.antmaze import data_spec
from spirl.components.evaluator import TopOfNSequenceEvaluator
from spirl.data.maze.src.maze_data_loader import MazeStateSequenceDataset
from spirl.maze_few_demo import get_demo_from_file, process_demo

from spirl.components.fsil import FewshotDataset

NUM_IL_DEMO = 10
subseq_len = 10
fewshot_dataset = FewshotDataset(
    'data/antmaze/Antmaze_LR.pkl',
    num_demo=NUM_IL_DEMO,
    subseq_len=subseq_len,
)
current_dir = os.path.dirname(os.path.realpath(__file__))

contra_model_cf = AttrDict(
    state_dimension=data_spec.state_dim,
    hidden_size=128,
    feature_size=32,
)

configuration = {
    'model': GoalClSPiRLMdl,
    'logger': SkillSpaceLogger,
    'data_dir': '.',
    'epoch_cycles_train': 10,
    'evaluator': TopOfNSequenceEvaluator,
    'top_of_n_eval': 100,
    'top_comp_metric': 'mse',
    'batch_size': 128,
    'num_epochs': 20,  # Total including pre-trained 200
    'fewshot_data': fewshot_dataset,
    'fewshot_batch_size': 128,
    'finetune_vae': False,
    'contra_config': contra_model_cf,
    'contra_ckpt': './experiments/antmaze/contrastive_LR/exact_fine_tuned_model.pt',
    'rst_data_path': './data/antmaze/ant_resets_LR.npy'
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    n_rollout_steps=10,
    kl_div_weight=1e-2,
    nz_enc=32,
    nz_mid=32,
    n_processing_layers=3,
    cond_decode=True,
    checkpt_path=f'{os.environ["EXP_DIR"]}/skill_prior_learning/maze/hierarchical_cl_state_gc'
    # checkpt_path=f'{os.environ["EXP_DIR"]}/few_shot_imitation_learning/maze/hierarchical_cl_state_gc_4M_B1024'
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config['dataset_spec']['dataset_class'] = MazeStateSequenceDataset
data_config['dataset_spec']['env_name'] = 'antmaze-large-diverse-v0'
data_config['dataset_spec']['dataset_path'] = './data/antmaze/Antmaze_filtered_LR.hdf5'
data_config.dataset_spec.subseq_len = model_config.n_rollout_steps + 1

# import os
# import numpy as np
#
# from spirl.models.closed_loop_spirl_mdl import GoalClSPiRLMdl
# from spirl.models.skill_prior_mdl import SkillSpaceLogger
# from spirl.utils.general_utils import AttrDict
# from spirl.configs.default_data_configs.maze import data_spec
# from spirl.components.evaluator import TopOfNSequenceEvaluator
# from spirl.data.maze.src.maze_data_loader import MazeStateSequenceDataset
# from spirl.maze_few_demo import get_demo_from_file, process_demo
#
# NUM_IL_DEMO = 20
# il_demo_states, il_demo_actions = get_demo_from_file('data/maze/demos.pkl', NUM_IL_DEMO)
# processed_demo_states, processed_demo_actions = process_demo(il_demo_states, il_demo_actions, 10)
#
#
# def sample_il_demo(batch):
#     idxes = np.random.choice(len(processed_demo_states), size=batch)
#     return processed_demo_states[idxes], processed_demo_actions[idxes]
#
#
# current_dir = os.path.dirname(os.path.realpath(__file__))
#
# configuration = {
#     'model': GoalClSPiRLMdl,
#     'logger': SkillSpaceLogger,
#     'data_dir': '.',
#     'epoch_cycles_train': 10,
#     'evaluator': TopOfNSequenceEvaluator,
#     'top_of_n_eval': 100,
#     'top_comp_metric': 'mse',
#     'batch_size': 1024,
#     'num_epochs': 220,  # Total including pre-trained 200
#     'il_demo_sampler': sample_il_demo,
#     'il_demo_batch_size': 128
# }
# configuration = AttrDict(configuration)
#
# model_config = AttrDict(
#     state_dim=data_spec.state_dim,
#     action_dim=data_spec.n_actions,
#     n_rollout_steps=10,
#     kl_div_weight=1e-2,
#     nz_enc=32,
#     nz_mid=32,
#     n_processing_layers=3,
#     cond_decode=True,
#     checkpt_path=f'{os.environ["EXP_DIR"]}/skill_prior_learning/maze/hierarchical_cl_state_gc_4M_B1024'
# )
#
# # Dataset
# data_config = AttrDict()
# data_config.dataset_spec = data_spec
# data_config['dataset_spec']['dataset_class'] = MazeStateSequenceDataset
# data_config['dataset_spec']['env_name'] = 'maze2d-large-v1'
# data_config['dataset_spec']['dataset_path'] = './maze2d-large-blr-v1-noisy-4M.hdf5'
# data_config.dataset_spec.subseq_len = model_config.n_rollout_steps + 1
