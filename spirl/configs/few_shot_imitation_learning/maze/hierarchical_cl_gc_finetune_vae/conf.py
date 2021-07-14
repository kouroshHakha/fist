import os
import numpy as np

from spirl.models.closed_loop_spirl_mdl import GoalClSPiRLMdl
from spirl.models.skill_prior_mdl import SkillSpaceLogger
from spirl.utils.general_utils import AttrDict
from spirl.configs.default_data_configs.maze import data_spec
from spirl.components.evaluator import TopOfNSequenceEvaluator
from spirl.data.maze.src.maze_data_loader import MazeStateSequenceDataset

from spirl.components.fsil import FewshotDataset

NUM_IL_DEMO = 10
subseq_len = 10
fewshot_dataset = FewshotDataset(
    'data/maze/demos.pkl',
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
    'epoch_cycles_train': 1,
    'evaluator': TopOfNSequenceEvaluator,
    'top_of_n_eval': 100,
    'top_comp_metric': 'mse',
    'batch_size': 128,
    'num_epochs': 200,
    'fewshot_data': fewshot_dataset,
    'fewshot_batch_size': 128,
    'contra_config': contra_model_cf,
    'contra_ckpt': './experiments/contrastive/maze/exact-2021-05-06-9-56/exact_model.pt',
    'finetune_vae': True,
    'rst_data_path': './data/maze/rsts.npy'
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    n_rollout_steps=subseq_len,
    kl_div_weight=1e-2,
    nz_enc=32,
    nz_mid=32,
    n_processing_layers=3,
    cond_decode=True,
    checkpt_path=f'{os.environ["EXP_DIR"]}/skill_prior_learning/maze/hierarchical_cl_state_gc_4M_B1024'
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config['dataset_spec']['dataset_class'] = MazeStateSequenceDataset
data_config['dataset_spec']['env_name'] = 'maze2d-large-v1'
data_config['dataset_spec']['dataset_path'] = './data/maze/maze2d-large-blr-v1-noisy-4M.hdf5'
data_config.dataset_spec.subseq_len = model_config.n_rollout_steps + 1
