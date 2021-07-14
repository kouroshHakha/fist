import os
import numpy as np

from spirl.models.closed_loop_spirl_mdl import ClSPiRLMdl
from spirl.models.skill_prior_mdl import SkillSpaceLogger
from spirl.utils.general_utils import AttrDict
from spirl.configs.default_data_configs.maze import data_spec
from spirl.components.evaluator import TopOfNSequenceEvaluator
from spirl.data.maze.src.maze_data_loader import MazeStateSequenceDataset
from spirl.maze_few_demo import get_demo_from_file, process_demo

NUM_IL_DEMO = 10
il_demo_states, il_demo_actions = get_demo_from_file('data/maze/demos.pkl', NUM_IL_DEMO)
processed_demo_states, processed_demo_actions = process_demo(il_demo_states, il_demo_actions, 10)


def sample_il_demo(batch):
    idxes = np.random.choice(len(processed_demo_states), size=batch)
    return processed_demo_states[idxes], processed_demo_actions[idxes]


current_dir = os.path.dirname(os.path.realpath(__file__))

configuration = {
    'model': ClSPiRLMdl,
    'logger': SkillSpaceLogger,
    'data_dir': '.',
    'epoch_cycles_train': 10,
    'evaluator': TopOfNSequenceEvaluator,
    'top_of_n_eval': 100,
    'top_comp_metric': 'mse',
    'batch_size': 1024,
    'num_epochs': 220,  # Total including pre-trained 200
    'il_demo_sampler': sample_il_demo,
    'il_demo_batch_size': 128
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
    checkpt_path=f'{os.environ["EXP_DIR"]}/few_shot_imitation_learning/maze/hierarchical_cl_state_4M_B1024_only_demos'
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config['dataset_spec']['dataset_class'] = MazeStateSequenceDataset
data_config['dataset_spec']['env_name'] = 'maze2d-large-v1'
data_config['dataset_spec']['dataset_path'] = './maze2d-large-blr-v1-noisy-4M.hdf5'
data_config.dataset_spec.subseq_len = model_config.n_rollout_steps + 1
