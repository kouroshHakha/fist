import matplotlib; matplotlib.use('Agg')
import torch
import os
import time
from shutil import copy
import datetime
import imp
from tensorboardX import SummaryWriter
import numpy as np
import random
from torch import autograd
from torch.optim import Adam, RMSprop, SGD
from functools import partial

from spirl.components.data_loader import RandomVideoDataset
from spirl.utils.general_utils import RecursiveAverageMeter, map_dict
from spirl.components.checkpointer import CheckpointHandler, save_cmd, save_git, get_config_path
from spirl.utils.general_utils import dummy_context, AttrDict, get_clipped_optimizer, \
                                                        AverageMeter, ParamDict
from spirl.utils.pytorch_utils import LossSpikeHook, NanGradHook, NoneGradHook, \
                                                        DataParallelWrapper, RAdam
from spirl.components.trainer_base import BaseTrainer
from spirl.components.params import get_args

from spirl.models.contrastive import ContrastiveFutureState
from spirl.utils.pytorch_utils import ar2ten, ten2ar
from torch.utils.data import DataLoader

from spirl.utils.debug import register_pdb_hook
register_pdb_hook()

class ModelTrainer(BaseTrainer):
    def __init__(self, args):
        self.args = args
        self.setup_device()

        # set up params
        self.conf = conf = self.get_config()

        self._hp = self._default_hparams()
        self._hp.overwrite(conf.general)  # override defaults with config file
        self._hp.exp_path = make_path(conf.exp_dir, args.path, args.prefix, args.new_dir)
        self.log_dir = log_dir = os.path.join(self._hp.exp_path, 'events')
        print('using log dir: ', log_dir)
        self.conf = self.postprocess_conf(conf)
        if args.deterministic: set_seeds()

        # set up logging + training monitoring
        self.writer = self.setup_logging(conf, self.log_dir)
        self.setup_training_monitors()
        
        # build dataset, model. logger, etc.
        train_params = AttrDict(logger_class=self._hp.logger,
                                model_class=self._hp.model,
                                n_repeat=self._hp.epoch_cycles_train,
                                dataset_size=-1)
        self.logger, self.model, self.train_loader = self.build_phase(train_params, 'train')

        test_params = AttrDict(logger_class=self._hp.logger if self._hp.logger_test is None else self._hp.logger_test,
                               model_class=self._hp.model if self._hp.model_test is None else self._hp.model_test,
                               n_repeat=1,
                               dataset_size=args.val_data_size)
        self.logger_test, self.model_test, self.val_loader = self.build_phase(test_params, phase='val')

        # set up optimizer + evaluator
        self.optimizer = self._get_optimizer()
        self.evaluator = self._hp.evaluator(self._hp, self.log_dir, self._hp.top_of_n_eval,
                                            self._hp.top_comp_metric, tb_logger=self.logger_test)


        self.fewshot_dloader = DataLoader(self.conf.general['fewshot_data'],
                                          batch_size=self.conf.general['fewshot_batch_size'],
                                          drop_last=True)

        if self._hp.contra_config:
            self.contrastive_mdl = ContrastiveFutureState(**self._hp.contra_config).to(self.device)
            if self._hp.contra_ckpt:
                print(f'Loading checkpoint {self._hp.contra_ckpt} ...')
                self.contrastive_mdl.load_state_dict(torch.load(self._hp.contra_ckpt, map_location=self.device))
                print(f'Checkpoint {self._hp.contra_ckpt} Loaded.')
        else:
            self.contrastive_mdl = None
        
        # load model params from checkpoint
        self.global_step, start_epoch = 0, 0
        if args.resume or conf.ckpt_path is not None:
            start_epoch = self.resume(args.resume, conf.ckpt_path)

        if args.val_sweep:
            self.run_val_sweep()
        elif args.train:
            self.train(start_epoch)
        else:
            self.val()
    
    def _default_hparams(self):
        default_dict = ParamDict({
            'model': None,
            'model_test': None,
            'logger': None,
            'logger_test': None,
            'evaluator': None,
            'data_dir': None,  # directory where dataset is in
            'batch_size': 16,
            'exp_path': None,  # Path to the folder with experiments
            'num_epochs': 10,
            'epoch_cycles_train': 1,
            'optimizer': 'radam',    # supported: 'adam', 'radam', 'rmsprop', 'sgd'
            'lr': 1e-3,
            'gradient_clip': None,
            'momentum': 0,      # momentum in RMSProp / SGD optimizer
            'adam_beta': 0.9,       # beta1 param in Adam
            'top_of_n_eval': 1,     # number of samples used at eval time
            'top_comp_metric': None,    # metric that is used for comparison at eval time (e.g. 'mse')
            'contra_config': None,
            'contra_ckpt': None,
            'finetune_vae': False,
        })
        return default_dict

    def _loss_backward(self, losses):
        if self._hp.finetune_vae:
            losses.total.value.backward()
        else:
            losses.q_hat_loss.value.backward()

    def _get_optimizer(self):
        if self._hp.finetune_vae:
            return self.get_optimizer_class()(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self._hp.lr)

        return self.get_optimizer_class()(filter(lambda p: p.requires_grad, self.model.p.parameters()), lr=self._hp.lr)

    def eval(self, replan_t):
        # TODO: eval is data dependent right now.
        from ruamel.yaml import YAML
        yaml=YAML(typ='safe')
        from pathlib import Path

        # TODO: run eval for only 10 rst points
        rst_points_list = []
        for i in range(1):
            rst_points_list.append(np.load(self.conf.general['rst_data_path'])[i * 10: (i+1) * 10])
        import gym
        from d4rl.pointmaze import maze_model
        from spirl.models.closed_loop_spirl_mdl import GoalClSPiRLMdl
        import imageio

        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('TkAgg')

        # prepare the data structures required for retrieving the reachable target state
        demo_dataset = self.conf.general['fewshot_data']
        demo_states, demo_ids, step_ids = [], [], []
        for i in range(demo_dataset.n_demos):
            states = demo_dataset.get_subseqs_of_seq(i)['states']
            demo_states.append(states)
            demo_ids.append(np.array([i]*states.shape[0]))
            step_ids.append(np.arange(states.shape[0]))

        # for d in demo_states:
        #     plt.plot(d[:, 0], d[:, 1])
        # plt.show()
        demo_states = np.concatenate(demo_states, 0)
        demo_ids = np.concatenate(demo_ids, 0)
        step_ids = np.concatenate(step_ids, 0)

        if self.contrastive_mdl is not None:
            z_states = self.contrastive_mdl.encode(ar2ten(demo_states, self.device).float())
        else:
            z_states = None

        subseq_len = self.conf.model.n_rollout_steps

        from mujoco_py import GlfwContext
        GlfwContext(offscreen=True)

        self.model.eval()
        env = gym.make('maze2d-large-v1')
        maze = env.str_maze_spec
        from spirl.maze_few_demo import TARGET_LOCATIONS
        target = None
        for k in TARGET_LOCATIONS.keys():
            if k in self._hp.exp_path:
                print('##### Using Goal for ' + k)
                target = TARGET_LOCATIONS[k]
                break
        env = maze_model.MazeEnv(maze)
        env.set_target(target + env.np_random.uniform(low=-.1, high=.1, size=env.model.nq))
        env.reset()

        video_path = Path(self._hp.exp_path) / 'videos'
        video_path.mkdir(exist_ok=True)
        max_videos = 20000
        video_done = False

        rst_points_list = rst_points_list * 10
        imgs = []
        summary = []
        for fold_idx, rst_points in enumerate(rst_points_list):
            print('Fold {}:'.format(fold_idx))
            success_cnt = 0
            episode_lens = []
            for rst_idx, rst in enumerate(rst_points):
                print(f'Evaluating episode {rst_idx} ...')
                s = rst
                env.set_state(qpos=rst[:2], qvel=rst[2:])
                done = False
                episode_lens.append(0)
                success = False
                while True:
                    if isinstance(self.model, GoalClSPiRLMdl):
                        if self.contrastive_mdl is None:
                            dists = ((demo_states - s) ** 2).sum(axis=-1)
                        else:
                            z_cur = self.contrastive_mdl.encode(ar2ten(s, self.device)[None].float())
                            dists = ten2ar(((z_states - z_cur) ** 2).sum(-1))

                        min_idx = np.argmin(dists)
                        traj_states = demo_dataset.get_subseqs_of_seq(demo_ids[min_idx])['states']

                        # if self.contrastive_mdl is None:
                        step_idx = min(len(traj_states) - 1, step_ids[min_idx] + subseq_len)
                        # else:
                        #     step_idx = step_ids[min_idx]

                        s_target = traj_states[step_idx]
                        sg = np.concatenate([s, s_target])
                        z = self.model.compute_learned_prior(ar2ten(sg, self.device)[None].float()).sample()
                    else:
                        z = self.model.compute_learned_prior(torch.Tensor([s]).to('cuda')).sample()

                    for i in range(replan_t):
                        ipt = AttrDict({'states': torch.Tensor([[s]]).to('cuda')})
                        a = self.model.decode(z, None, 1, ipt).cpu().detach().numpy()[0, 0]
                        s, _, _, done = env.step(a)
                        episode_lens[-1] += 1

                        if len(imgs) < max_videos:
                            imgs.append(env.render(mode='rgb_array', width=256, height=256))

                        success = np.linalg.norm(s[0:2] - env.get_target()) <= 0.5
                        success_cnt += float(success)

                        last_step = (rst_idx == len(rst_points) - 1) and (episode_lens[-1] > 2000 or success)
                        if ((len(imgs) == max_videos) or last_step) and not video_done:
                            print('Saving the video ...')
                            imageio.mimsave(video_path / 'eval.mp4', imgs, fps=120)
                            print('Saving done.')
                            video_done = True

                        if success or episode_lens[-1] > 2000:
                            done = True
                            break

                    if done:
                        break
                print(f'sucess = {success}')

            print(f'sucess_rate: {success_cnt / len(rst_points)}')
            summary.append(dict(success_rate=success_cnt / len(rst_points), episode_lens=episode_lens,
                                avg_ep_len=sum(episode_lens)/len(episode_lens)))
        with open(video_path / 'summary.yaml', 'w') as f:
            yaml.dump(summary, f)
        self.model.train()

    def train(self, start_epoch):
        if self.args.eval:
            self.eval(2)
            exit(0)

        # if not self.args.skip_first_val:
        #     self.val()
            
        for epoch in range(start_epoch, self._hp.num_epochs):
            self.train_epoch(epoch)
        
            if not self.args.dont_save:
                save_checkpoint({
                    'epoch': epoch,
                    'global_step': self.global_step,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                },  os.path.join(self._hp.exp_path, 'weights'), CheckpointHandler.get_ckpt_name(epoch))

            # if epoch % self.args.val_interval == 0:
            #     self.val()

    def train_epoch(self, epoch):
        self.model.train()
        epoch_len = len(self.train_loader)
        end = time.time()
        batch_time = AverageMeter()
        upto_log_time = AverageMeter()
        data_load_time = AverageMeter()
        self.log_outputs_interval = self.args.log_interval
        self.log_images_interval = int(epoch_len / self.args.per_epoch_img_logs)
        
        print('starting epoch ', epoch)

        for self.batch_idx, sample_batched in enumerate(self.fewshot_dloader):
        # for self.batch_idx, sample_batched in enumerate(self.train_loader):
            data_load_time.update(time.time() - end)
            inputs = AttrDict(map_dict(lambda x: x.to(self.device).float(), sample_batched))
            with self.training_context():
                self.optimizer.zero_grad()
                output = self.model(inputs)
                losses = self.model.loss(output, inputs)
                self._loss_backward(losses)
                self.call_hooks(inputs, output, losses, epoch)
                self.optimizer.step()
                self.model.step()

            if self.args.train_loop_pdb:
                import pdb; pdb.set_trace()
            
            upto_log_time.update(time.time() - end)
            if self.log_outputs_now and not self.args.dont_save:
                self.model.log_outputs(output, inputs, losses, self.global_step,
                                       log_images=self.log_images_now, phase='train', **self._logging_kwargs)
            batch_time.update(time.time() - end)
            end = time.time()
            
            if self.log_outputs_now:
                print('GPU {}: {}'.format(os.environ["CUDA_VISIBLE_DEVICES"] if self.use_cuda else 'none',
                                          self._hp.exp_path))
                print(('itr: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        self.global_step, epoch, self.batch_idx, len(self.train_loader),
                        100. * self.batch_idx / len(self.train_loader), losses.total.value.item())))

                print('avg time for loading: {:.2f}s, logs: {:.2f}s, compute: {:.2f}s, total: {:.2f}s'
                      .format(data_load_time.avg,
                              batch_time.avg - upto_log_time.avg,
                              upto_log_time.avg - data_load_time.avg,
                              batch_time.avg))
                togo_train_time = batch_time.avg * (self._hp.num_epochs - epoch) * epoch_len / 3600.
                print('ETA: {:.2f}h'.format(togo_train_time))

            del output, losses
            self.global_step = self.global_step + 1

    def val(self):
        print('Running Testing')
        if self.args.test_prediction:
            start = time.time()
            self.model_test.load_state_dict(self.model.state_dict())
            losses_meter = RecursiveAverageMeter()
            self.model_test.eval()
            self.evaluator.reset()
            with autograd.no_grad():
                for batch_idx, sample_batched in enumerate(self.val_loader):
                    inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))

                    # run evaluator with val-mode model
                    with self.model_test.val_mode():
                        self.evaluator.eval(inputs, self.model_test)

                    # run non-val-mode model (inference) to check overfitting
                    output = self.model_test(inputs)
                    losses = self.model_test.loss(output, inputs)

                    losses_meter.update(losses)
                    del losses
                
                if not self.args.dont_save:
                    if self.evaluator is not None:
                        self.evaluator.dump_results(self.global_step)

                    self.model_test.log_outputs(output, inputs, losses_meter.avg, self.global_step,
                                                log_images=True, phase='val', **self._logging_kwargs)
                    print(('\nTest set: Average loss: {:.4f} in {:.2f}s\n'
                           .format(losses_meter.avg.total.value.item(), time.time() - start)))
            del output

    def setup_device(self):
        self.use_cuda = torch.cuda.is_available() and not self.args.debug
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
        if self.args.gpu != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)

    def get_config(self):
        conf = AttrDict()

        # paths
        conf.exp_dir = self.get_exp_dir()
        conf.conf_path = get_config_path(self.args.path)

        # general and model configs
        print('loading from the config file {}'.format(conf.conf_path))
        conf_module = imp.load_source('conf', conf.conf_path)
        conf.general = conf_module.configuration
        conf.model = conf_module.model_config

        # data config
        try:
            data_conf = conf_module.data_config
        except AttributeError:
            data_conf_file = imp.load_source('dataset_spec', os.path.join(AttrDict(conf).data_dir, 'dataset_spec.py'))
            data_conf = AttrDict()
            data_conf.dataset_spec = AttrDict(data_conf_file.dataset_spec)
            data_conf.dataset_spec.split = AttrDict(data_conf.dataset_spec.split)
        conf.data = data_conf

        # model loading config
        conf.ckpt_path = conf.model.checkpt_path if 'checkpt_path' in conf.model else None

        if hasattr(conf_module, 'env'):
            conf.env = conf_module.env
        else:
            conf.env = AttrDict()

        return conf

    def postprocess_conf(self, conf):
        conf.model['batch_size'] = self._hp.batch_size if not torch.cuda.is_available() \
            else int(self._hp.batch_size / torch.cuda.device_count())
        conf.model.update(conf.data.dataset_spec)
        conf.model['device'] = conf.data['device'] = self.device.type
        return conf

    def setup_logging(self, conf, log_dir):
        if not self.args.dont_save:
            print('Writing to the experiment directory: {}'.format(self._hp.exp_path))
            if not os.path.exists(self._hp.exp_path):
                os.makedirs(self._hp.exp_path)
            save_cmd(self._hp.exp_path)
            save_git(self._hp.exp_path)
            save_config(conf.conf_path, os.path.join(self._hp.exp_path, "conf_" + datetime_str() + ".py"))
            writer = SummaryWriter(log_dir)
        else:
            writer = None

        # set up additional logging args
        self._logging_kwargs = AttrDict(
        )
        return writer

    def setup_training_monitors(self):
        self.training_context = autograd.detect_anomaly if self.args.detect_anomaly else dummy_context
        self.hooks = []
        self.hooks.append(LossSpikeHook('sg_img_mse_train'))
        self.hooks.append(NanGradHook(self))
        self.hooks.append(NoneGradHook(self))

    def build_phase(self, params, phase):
        if not self.args.dont_save:
            logger = params.logger_class(self.log_dir, summary_writer=self.writer)
        else:
            logger = None
        model = params.model_class(self.conf.model, logger)
        if torch.cuda.device_count() > 1:
            print("\nUsing {} GPUs!\n".format(torch.cuda.device_count()))
            model = DataParallelWrapper(model)
        model = model.to(self.device)
        model.device = self.device
        loader = self.get_dataset(self.args, model, self.conf.data, phase, params.n_repeat, params.dataset_size)
        return logger, model, loader

    def get_dataset(self, args, model, data_conf, phase, n_repeat, dataset_size=-1):
        if args.feed_random_data:
            dataset_class = RandomVideoDataset
        else:
            dataset_class = data_conf.dataset_spec.dataset_class

        dset = dataset_class(self._hp.data_dir, data_conf, resolution=model.resolution,
                             phase=phase, shuffle=phase == "train", dataset_size=dataset_size)
        loader = dset.get_data_loader(self._hp.batch_size, n_repeat)

        return loader

    def resume(self, ckpt, path=None):
        path = os.path.join(self._hp.exp_path, 'weights') if path is None else os.path.join(path, 'weights')
        assert ckpt is not None  # need to specify resume epoch for loading checkpoint
        weights_file = CheckpointHandler.get_resume_ckpt_file(ckpt, path)
        CheckpointHandler.load_weights(weights_file, self.model,
                                       # load_step=True,
                                       # load_opt=True, optimizer=self.optimizer,
                                       strict=self.args.strict_weight_loading)
        self.model.to(self.model.device)
        return 0

    def get_optimizer_class(self):
        optim = self._hp.optimizer
        if optim == 'adam':
            get_optim = partial(get_clipped_optimizer, optimizer_type=Adam, betas=(self._hp.adam_beta, 0.999))
        elif optim == 'radam':
            get_optim = partial(get_clipped_optimizer, optimizer_type=RAdam, betas=(self._hp.adam_beta, 0.999))
        elif optim == 'rmsprop':
            get_optim = partial(get_clipped_optimizer, optimizer_type=RMSprop, momentum=self._hp.momentum)
        elif optim == 'sgd':
            get_optim = partial(get_clipped_optimizer, optimizer_type=SGD, momentum=self._hp.momentum)
        else:
            raise ValueError("Optimizer '{}' not supported!".format(optim))
        return partial(get_optim, gradient_clip=self._hp.gradient_clip)

    def run_val_sweep(self):
        epochs = CheckpointHandler.get_epochs(os.path.join(self._hp.exp_path, 'weights'))
        for epoch in list(sorted(epochs))[::2]:
            self.resume(epoch)
            self.val()
        return

    def get_exp_dir(self):
        return os.environ['EXP_DIR']

    @property
    def log_images_now(self):
        return self.global_step % self.log_images_interval == 0

    @property
    def log_outputs_now(self):
        return self.global_step % self.log_outputs_interval == 0 or self.global_step % self.log_images_interval == 0


def save_checkpoint(state, folder, filename='checkpoint.pth'):
    os.makedirs(folder, exist_ok=True)
    torch.save(state, os.path.join(folder, filename))
    print(f"Saved checkpoint to {os.path.join(folder, filename)}!")


def get_exp_dir():
    return os.environ['EXP_DIR']


def datetime_str():
    return datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")


def make_path(exp_dir, conf_path, prefix, make_new_dir):
    # extract the subfolder structure from config path
    path = conf_path.split('configs/', 1)[1]
    if make_new_dir:
        prefix += datetime_str()
    base_path = os.path.join(exp_dir, path)
    return os.path.join(base_path, prefix) if prefix else base_path


def set_seeds(seed=0, cuda_deterministic=True):
    """Sets all seeds and disables non-determinism in cuDNN backend."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available() and cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def save_config(conf_path, exp_conf_path):
    copy(conf_path, exp_conf_path)

        
if __name__ == '__main__':
    ModelTrainer(args=get_args())
