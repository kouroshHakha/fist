import os

from spirl.models.closed_loop_spirl_mdl import GoalClSPiRLMdl
from spirl.models.skill_prior_mdl import SkillSpaceLogger
from spirl.utils.general_utils import AttrDict
from spirl.configs.default_data_configs.maze import data_spec
from spirl.components.evaluator import TopOfNSequenceEvaluator
from spirl.data.maze.src.maze_data_loader import MazeStateSequenceDataset


current_dir = os.path.dirname(os.path.realpath(__file__))

data_spec['n_actions'] = 8
data_spec['state_dim'] = 29
configuration = {
    'model': GoalClSPiRLMdl,
    'logger': SkillSpaceLogger,
    'data_dir': '.',
    'epoch_cycles_train': 10,
    'evaluator': TopOfNSequenceEvaluator,
    'top_of_n_eval': 100,
    'top_comp_metric': 'mse',
    'batch_size': 128
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
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config['dataset_spec']['dataset_class'] = MazeStateSequenceDataset
data_config['dataset_spec']['env_name'] = 'antmaze-large-diverse-v0'
data_config['dataset_spec']['dataset_path'] = './data/antmaze/Antmaze_filtered_UL.hdf5'
data_config.dataset_spec.subseq_len = model_config.n_rollout_steps + 1
