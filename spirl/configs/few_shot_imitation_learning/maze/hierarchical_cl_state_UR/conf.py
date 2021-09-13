import os
import numpy as np

from spirl.models.closed_loop_spirl_mdl import ClSPiRLMdl
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
    'data/antmaze/Antmaze_UR.pkl',
    num_demo=NUM_IL_DEMO,
    subseq_len=subseq_len,
)
current_dir = os.path.dirname(os.path.realpath(__file__))

configuration = {
    'model': ClSPiRLMdl,
    'logger': SkillSpaceLogger,
    'data_dir': '.',
    'epoch_cycles_train': 10,
    'evaluator': TopOfNSequenceEvaluator,
    'top_of_n_eval': 100,
    'top_comp_metric': 'mse',
    'batch_size': 128,
    'num_epochs': 220,  # Total including pre-trained 200
    'fewshot_data': fewshot_dataset,
    'fewshot_batch_size': 128,
    'finetune_vae': False,
    'rst_data_path': './data/antmaze/ant_resets_10.npy'
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
    # checkpt_path=f'{os.environ["EXP_DIR"]}/skill_prior_learning/maze/hierarchical_cl_state'
    checkpt_path=f'{os.environ["EXP_DIR"]}/few_shot_imitation_learning/maze/hierarchical_cl_state_4M_B1024'
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config['dataset_spec']['dataset_class'] = MazeStateSequenceDataset
data_config['dataset_spec']['env_name'] = 'antmaze-large-diverse-v0'
data_config['dataset_spec']['dataset_path'] = './data/antmaze/Antmaze_filtered_UR.hdf5'
data_config.dataset_spec.subseq_len = model_config.n_rollout_steps + 1
