import os

from spirl.models.closed_loop_spirl_mdl import ClSPiRLMdl
from spirl.components.logger import Logger
from spirl.utils.general_utils import AttrDict
from spirl.configs.default_data_configs.maze import data_spec
from spirl.components.evaluator import TopOfNSequenceEvaluator
from spirl.components.fsil import FewshotDataset
from spirl.data.maze.src.maze_data_loader import MazeStateSequenceDataset

current_dir = os.path.dirname(os.path.realpath(__file__))

NUM_IL_DEMO = 10
subseq_len = 10
fewshot_dataset = FewshotDataset(
    'data/maze/left/demos.pkl',
    num_demo=NUM_IL_DEMO,
    subseq_len=subseq_len,
)

contra_model_cf = AttrDict(
    state_dimension=data_spec.state_dim,
    hidden_size=128,
    feature_size=32,
)

configuration = {
    'model': ClSPiRLMdl,
    'logger': Logger,
    'data_dir': '.',
    'epoch_cycles_train': 10,
    'evaluator': TopOfNSequenceEvaluator,
    'top_of_n_eval': 100,
    'top_comp_metric': 'mse',
    'batch_size': 1024,
    'num_epochs': 100,
    'fewshot_data': fewshot_dataset,
    'fewshot_batch_size': 1024,
    'offline_data': True,
    'bc_model': None
}
configuration = AttrDict(configuration)

# Included to make the script run, but is not used.
model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    n_rollout_steps=subseq_len,
    kl_div_weight=1e-2,
    nz_enc=32,
    nz_mid=32,
    n_processing_layers=3,
    cond_decode=True
)

bc_model = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    nz_mid=32,
    n_processing_layers=3
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config['dataset_spec']['dataset_class'] = MazeStateSequenceDataset
data_config['dataset_spec']['env_name'] = 'maze2d-large-v1'
data_config['dataset_spec']['dataset_path'] = './data/maze/left/blocked-4M.hdf5'
data_config['dataset_spec']['subseq_len'] = model_config.n_rollout_steps + 1  # flat last action from seq gets cropped
