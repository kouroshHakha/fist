import d4rl
import gym
import numpy as np
import itertools

from spirl.components.data_loader import Dataset
from spirl.utils.general_utils import AttrDict
from spirl.utils.general_utils import listdict2dictlist
import torch.utils.data as data

class D4RLSequenceSplitDataset(Dataset):
    SPLIT = AttrDict(train=0.99, val=0.01, test=0.0)

    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1):
        self.phase = phase
        self.data_dir = data_dir
        self.spec = data_conf.dataset_spec
        self.subseq_len = self.spec.subseq_len
        self.remove_goal = self.spec.remove_goal if 'remove_goal' in self.spec else False
        self.dataset_size = dataset_size
        self.device = data_conf.device
        self.n_worker = 4
        self.shuffle = shuffle

        env = gym.make(self.spec.env_name)
        try:
            dataset_path = self.spec.dataset_path
        except AttributeError:
            dataset_path = None
        self.dataset = env.get_dataset(h5path=dataset_path)

        # split dataset into sequences
        seq_end_idxs = np.where(self.dataset['terminals'])[0]
        start = 0
        self.seqs = []
        for end_idx in seq_end_idxs:
            if end_idx+1 - start < self.subseq_len: continue    # skip too short demos
            self.seqs.append(AttrDict(
                states=self.dataset['observations'][start:end_idx+1],
                actions=self.dataset['actions'][start:end_idx+1],
            ))
            start = end_idx+1

        # 0-pad sequences for skill-conditioned training
        if 'pad_n_steps' in self.spec and self.spec.pad_n_steps > 0:
            for seq in self.seqs:
                seq.states = np.concatenate((np.zeros((self.spec.pad_n_steps, seq.states.shape[1]), dtype=seq.states.dtype), seq.states))
                seq.actions = np.concatenate((np.zeros((self.spec.pad_n_steps, seq.actions.shape[1]), dtype=seq.actions.dtype), seq.actions))

        # filter demonstration sequences
        if 'filter_indices' in self.spec:
            print("!!! Filtering kitchen demos in range {} !!!".format(self.spec.filter_indices))
            self.seqs = list(itertools.chain.from_iterable(itertools.repeat(x, self.spec.demo_repeats)
                               for x in self.seqs[self.spec.filter_indices[0] : self.spec.filter_indices[1]+1]))
            import random
            random.shuffle(self.seqs)

        self.n_seqs = len(self.seqs)

        if self.phase == "train":
            self.start = 0
            self.end = int(self.SPLIT.train * self.n_seqs)
        elif self.phase == "val":
            self.start = int(self.SPLIT.train * self.n_seqs)
            self.end = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
        else:
            self.start = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
            self.end = self.n_seqs

    def __getitem__(self, index):
        # sample start index in data range
        seq = self._sample_seq()
        start_idx = np.random.randint(0, seq.states.shape[0] - self.subseq_len - 1)
        output = AttrDict(
            states=seq.states[start_idx:start_idx+self.subseq_len],
            actions=seq.actions[start_idx:start_idx+self.subseq_len-1],
            pad_mask=np.ones((self.subseq_len,)),
        )
        if self.remove_goal:
            output.states = output.states[..., :int(output.states.shape[-1]/2)]
        return output

    def _sample_seq(self):
        seqs = self.seqs[self.start:self.end]
        idx = np.random.randint(len(seqs))
        return seqs[idx]

    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size
        return int(self.SPLIT[self.phase] * self.dataset['observations'].shape[0] / self.subseq_len)


class KitchenStateSeqDataset(data.Dataset):
    def __init__(self, data_path, num_demo=10, subseq_len=10):
        super().__init__()
        self.subseq_len = subseq_len

        env = gym.make('kitchen-mixed-v0')
        self.dataset = env.get_dataset(data_path)

        # split dataset into sequences
        seq_end_idxs = np.where(self.dataset['terminals'])[0]
        start = 0
        self.seqs = []
        for end_idx in seq_end_idxs:
            if len(self.seqs) < num_demo:
                self.seqs.append(AttrDict(
                    states=self.dataset['observations'][start:end_idx+1],
                    actions=self.dataset['actions'][start:end_idx+1],
                ))
                start = end_idx+1

        self.n_demos = num_demo
        self.n_unique_seqs = sum([seq.actions.shape[0] - subseq_len for seq in self.seqs])

    def __getitem__(self, item):
        demo_idx = np.random.randint(len(self.seqs))
        # last step of states is useless
        start_idx = np.random.randint(self.seqs[demo_idx].actions.shape[0] - self.subseq_len)
        output = AttrDict(
            states=self.seqs[demo_idx]['states'][start_idx:start_idx+self.subseq_len+1],
            actions=self.seqs[demo_idx]['actions'][start_idx:start_idx+self.subseq_len]
        )
        return output

    def __len__(self):
        return self.n_unique_seqs

    def get_rolling_sub_trajectories(self, item):
        seq = self.seqs[item]

        output = {'states': [], 'actions': []}
        for start in range(seq.actions.shape[0] - self.subseq_len):
            output['states'].append(seq.states[start:start+self.subseq_len+1])
            output['actions'].append(seq.actions[start:start+self.subseq_len])

        output = {k: np.stack(v, 0) for k, v in output.items()}

        return output

    def get_subseqs_of_seq(self, item):
        return self.seqs[item]