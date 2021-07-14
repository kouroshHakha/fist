import os

from spirl.models.closed_loop_spirl_mdl import GoalClSPiRLMdl
from spirl.components.logger import Logger
from spirl.utils.general_utils import AttrDict
from spirl.configs.default_data_configs.kitchen import data_spec
from spirl.components.evaluator import TopOfNSequenceEvaluator
from spirl.data.kitchen.src.kitchen_data_loader import KitchenStateSeqDataset

current_dir = os.path.dirname(os.path.realpath(__file__))

fewshot_dataset = KitchenStateSeqDataset(
    data_path='data/kitchen/kitchen-demo-topknob_bottomknob_hinge_slide.hdf5',
    num_demo=1,
    subseq_len=10,
)

env = AttrDict(
    task_list = ['bottom burner', 'top burner', 'slide cabinet', 'hinge cabinet']
)

contra_model_cf = AttrDict(
    state_dimension=data_spec.state_dim,
    hidden_size=128,
    feature_size=32,
)

configuration = {
    'model': GoalClSPiRLMdl,
    'logger': Logger,
    'data_dir': '.',
    'epoch_cycles_train': 1,
    'evaluator': TopOfNSequenceEvaluator,
    'top_of_n_eval': 100,
    'top_comp_metric': 'mse',
    'batch_size': 128,
    'num_epochs': 50,
    'fewshot_data': fewshot_dataset,
    'fewshot_batch_size': 128,
    'contra_config': contra_model_cf,
    'contra_ckpt': './experiments/contrastive/kitchen/exact-mixed-all/exact_model.pt',
    'finetune_vae': True,
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    n_rollout_steps=10,
    kl_div_weight=5e-4,
    nz_enc=128,
    nz_mid=128,
    n_processing_layers=5,
    cond_decode=True,
    checkpt_path=f'{os.environ["EXP_DIR"]}/skill_prior_learning/kitchen/hierarchical_cl_gc_all'
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec['dataset_path'] = './data/kitchen/kitchen-mixed-v0.hdf5'
data_config.dataset_spec.subseq_len = model_config.n_rollout_steps + 1  # flat last action from seq gets cropped
