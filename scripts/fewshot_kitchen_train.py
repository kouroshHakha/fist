import matplotlib; matplotlib.use('Agg')
import torch
import numpy as np
from spirl.utils.general_utils import AttrDict, ParamDict
from spirl.components.params import get_args

from spirl.utils.debug import register_pdb_hook
register_pdb_hook()

from spirl.utils.pytorch_utils import ar2ten, ten2ar
from spirl.fewshot_train import ModelTrainer as FSTrainer
from ruamel.yaml import YAML
yaml=YAML(typ='safe')

from pathlib import Path
import gym
from spirl.models.closed_loop_spirl_mdl import GoalClSPiRLMdl
import imageio
import time

import matplotlib.pyplot as plt


class ModelTrainer(FSTrainer):

    def eval(self, replan_t):
        # TODO: eval is data dependent right now.

        # prepare the data structures required for retrieving the reachable target state
        demo_dataset = self.conf.general['fewshot_data']
        demo_states, demo_ids, step_ids = [], [], []
        for i in range(demo_dataset.n_demos):
            states = demo_dataset.get_subseqs_of_seq(i)['states']
            demo_states.append(states)
            demo_ids.append(np.array([i]*states.shape[0]))
            step_ids.append(np.arange(states.shape[0]))
        demo_states = np.concatenate(demo_states, 0)
        demo_ids = np.concatenate(demo_ids, 0)
        step_ids = np.concatenate(step_ids, 0)

        if self.contrastive_mdl is not None:
            z_states = self.contrastive_mdl.encode(ar2ten(demo_states, self.device).float())
        else:
            z_states = None

        subseq_len = self.conf.model.n_rollout_steps

        self.model.eval()
        env = gym.make('kitchen-mixed-v0', task_elements=self.conf.env.task_list)

        video_path = Path(self._hp.exp_path) / 'videos'
        video_path.mkdir(exist_ok=True)
        max_len = 280
        nrst = 10
        max_total_video_steps = nrst * max_len
        video_done = False
        success_cnt = 0
        episode_lens, rew_list = [], []
        imgs = []
        for rst_idx in range(nrst):
            # if video_done: break
            print(f'Evaluating episode {rst_idx} ...')
            done = False
            episode_lens.append(0)
            rew_list.append(0)
            success = False
            s = env.reset()
            # for now since there is no quantitative FOM just exit the loop once the video is saved
            states = [s]
            while True:
                s1 = time.time()
                if isinstance(self.model, GoalClSPiRLMdl):
                    if self.contrastive_mdl is None:
                        dists = ((demo_states - s) ** 2).sum(axis=-1)
                        # # TODO: remove this plot code
                        # if episode_lens[-1] % 20 == 0:
                        #     plt.plot(dists)
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

                e1 = time.time()
                for i in range(replan_t):
                    ipt = AttrDict({'states': torch.Tensor([[s]]).to('cuda')})
                    s2 = time.time()
                    a = self.model.decode(z, None, 1, ipt).cpu().detach().numpy()[0, 0]
                    e2 = time.time()
                    s3 = time.time()
                    s, rew, d, _ = env.step(a)
                    e3 = time.time()
                    episode_lens[-1] += 1
                    rew_list[-1] += rew
                    states.append(s)
                    print(f'ep_step = {episode_lens[-1]}, rew = {rew_list[-1]}, done={d},'
                          f'sampled z time = {e1 - s1:.6f}, '
                          f'decode z time = {e2 - s2:.6f}, '
                          f'step time = {e3 - s3:.6f}')

                    if len(imgs) < max_total_video_steps:
                        print('rendering ...')
                        imgs.append(env.render(mode='rgb_array'))

                    success = rew_list[-1] == len(env.TASK_ELEMENTS)
                    success_cnt += float(success)

                    last_step = (rst_idx == nrst - 1) and (episode_lens[-1] > max_len or success)
                    if ((len(imgs) == max_total_video_steps) or last_step) and not video_done:
                        print('Saving the video ...')
                        imageio.mimsave(video_path / 'eval.mp4', imgs, fps=25)
                        print('Saving done.')
                        video_done = True
                        # plt.savefig('test.png')

                    if success or episode_lens[-1] > max_len:
                        done = True
                        break

                if done:
                    break
            print(f'sucess = {success}')

        print(f'sucess_rate: {success_cnt / nrst}')
        summary = dict(success_rate=success_cnt / nrst, episode_lens=episode_lens,
                       rewards=rew_list,
                       avg_ep_len=sum(episode_lens)/len(episode_lens))
        with open(video_path / 'summary.yaml', 'w') as f:
            yaml.dump(summary, f)
        self.model.train()

if __name__ == '__main__':
    ModelTrainer(args=get_args())
