from torch.utils.data import Dataset
from spirl.maze_few_demo import get_demo_from_file
from spirl.utils.general_utils import AttrDict
import numpy as np

class FewshotDataset(Dataset):
    def __init__(self, data_path, num_demo=10, subseq_len=10):
        super().__init__()
        self.states, self.actions = get_demo_from_file(data_path, num_demo)
        self.subseq_len = subseq_len
        self.n_unique_seqs = sum([seq.shape[0] - subseq_len for seq in self.actions])
        self.n_demos = len(self.states)

    def __getitem__(self, item):
        demo_idx = np.random.randint(len(self.states))
        # last step of states is useless
        start_idx = np.random.randint(self.actions[demo_idx].shape[0] - self.subseq_len)
        output = AttrDict(
            states=self.states[demo_idx][start_idx:start_idx+self.subseq_len+1],
            actions=self.actions[demo_idx][start_idx:start_idx+self.subseq_len]
        )
        return output

    def __len__(self):
        return self.n_unique_seqs

    def get_subseqs_of_seq(self, item):
        states = self.states[item]
        actions = self.actions[item]
        output = {'states': states, 'actions': actions}
        return output

