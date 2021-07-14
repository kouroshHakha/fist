
from spirl.components.base_model import BaseModel
from spirl.utils.general_utils import AttrDict, ParamDict
from spirl.modules.subnetworks import Predictor
from spirl.modules.variational_inference import MultivariateGaussian
from spirl.modules.losses import NLL
from spirl.modules.layers import LayerBuilderParams
import torch

class BCMdl(BaseModel):
    """Simple recurrent forward predictor network with image encoder and decoder."""
    def __init__(self, params, logger=None):
        BaseModel.__init__(self, logger)
        self._hp = self._default_hparams()
        self._hp.overwrite(params)  # override defaults with config file
        self._hp.builder = LayerBuilderParams(use_convs=False, normalization='none')
        self.device = self._hp.device
        self.build_network()


    def _default_hparams(self):
        # put new parameters in here:
        return super()._default_hparams().overwrite(ParamDict({
            'device': None,
            'state_dim': 1,             # dimensionality of the state space
            'action_dim': 1,            # dimensionality of the action space
            'nz_mid': 128,
            'n_processing_layers': 5,   # number of layers in MLPs
        }))


    def build_network(self):
        self.net = Predictor(self._hp, input_size=self._hp.state_dim, output_size=self._hp.action_dim * 2)

    def forward(self, inputs):
        """
        forward pass at training time
        """
        output = AttrDict()

        output.pred_act = self._compute_output_dist(self._net_inputs(inputs))
        return output

    def loss(self, model_output, inputs):
        losses = AttrDict()

        # reconstruction loss
        losses.nll = NLL()(model_output.pred_act, self._regression_targets(inputs))

        losses.total = self._compute_total_loss(losses)
        return losses

    def _compute_output_dist(self, inputs):
        return MultivariateGaussian(self.net(inputs))

    def _net_inputs(self, inputs):
        if self.training:
            _inputs = torch.cat(list(inputs.states[:, :-1]), 0)
        else:
            _inputs = torch.cat(list(inputs.states), 0)
        return _inputs

    def _regression_targets(self, inputs):
        _targets = torch.cat(list(inputs.actions), 0)
        return _targets


class GoalBCMdl(BCMdl):

    def build_network(self):
        self.net = Predictor(self._hp, input_size=self._hp.state_dim * 2, output_size=self._hp.action_dim * 2)

    def _net_inputs(self, inputs):
        return torch.cat((inputs.states[:, 0], inputs.states[:, -1]), dim=1)

    def _regression_targets(self, inputs):
        return inputs.actions[:, 0]