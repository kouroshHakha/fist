import os

from spirl.models.closed_loop_spirl_mdl import ClSPiRLMdl
from spirl.components.logger import Logger
from spirl.utils.general_utils import AttrDict
from spirl.configs.default_data_configs.kitchen import data_spec
from spirl.components.evaluator import TopOfNSequenceEvaluator
from spirl.data.kitchen.src.kitchen_data_loader import KitchenStateSeqDataset
from spirl.models.bc_atomic import BCMdl

current_dir = os.path.dirname(os.path.realpath(__file__))

fewshot_dataset = KitchenStateSeqDataset(
    data_path='data/kitchen/kitchen-demo-microwave_kettle_hinge_slide.hdf5',
    subseq_len=10,
)

env = AttrDict(
    task_list = ['microwave', 'kettle', 'slide cabinet', 'hinge cabinet']
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
    'epoch_cycles_train': 1,
    'evaluator': TopOfNSequenceEvaluator,
    'top_of_n_eval': 100,
    'top_comp_metric': 'mse',
    'batch_size': 128,
    'num_epochs': 50,
    'fewshot_data': fewshot_dataset,
    'fewshot_batch_size': 128,
    'offline_data': False,
    'contra_config': contra_model_cf,
    'contra_ckpt': './experiments/contrastive/kitchen/exact-mixed-all/exact_model.pt',
    'bc_model': BCMdl,
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
)

bc_model = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    nz_mid=128,
    n_processing_layers=5,
    # checkpt_path=f'{os.environ["EXP_DIR"]}/bc_atomic/kitchen/offline_data/no-slide',
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec['dataset_path'] = './data/kitchen/kitchen-mixed-no-slide.hdf5'
data_config.dataset_spec.subseq_len = model_config.n_rollout_steps + 1  # flat last action from seq gets cropped
