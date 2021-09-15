import torch
import h5py
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from spirl.models.contrastive import ContrastiveFutureState
from spirl.components.checkpointer import save_cmd
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env')
    parser.add_argument('--training-set')
    parser.add_argument('--demos')
    parser.add_argument('--mode', default='exact')
    parser.add_argument('--save-dir')
    parser.add_argument('--training-epochs', type=int, default=200)
    parser.add_argument('--finetune-epochs', type=int, default=20)
    parser.add_argument('--num-demos', type=int, default=10)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--finetune-size', type=int, default=512)

    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--feature-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)

    parsed_args = parser.parse_args()
    return parsed_args


def get_demo_states(env, demo_file, num_demos):

    if env == 'maze':
        from spirl.maze_few_demo import get_demo_from_file
        demo_states, _ = get_demo_from_file(args.demos, num_demo=args.num_demos)

    elif env == 'kitchen':
        from spirl.data.kitchen.src.kitchen_data_loader import KitchenStateSeqDataset
        demo_dataset = KitchenStateSeqDataset(demo_file, num_demo=num_demos)
        demo_states = [seq['states'] for seq in demo_dataset.seqs]
    else:
        raise ValueError('Env not supported!')

    demo_starting_idxes = []
    demo_idx_counter = 0
    for i in range(len(demo_states)):
        demo_starting_idxes += list(np.arange(len(demo_states[i]) - args.horizon) + demo_idx_counter)
        demo_idx_counter += len(demo_states[i])
    demo_starting_idxes = np.array(demo_starting_idxes)
    demo_states_flat = np.concatenate(demo_states, axis=0)

    return demo_starting_idxes, demo_states_flat


if __name__ == '__main__':
    args = parse_args()

    finetune = args.demos is not None
    finetune_epochs = args.finetune_epochs if finetune else 0

    if args.mode not in ['exact', 'reachable']:
        raise KeyError('Unsupported mode')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # we are probably closing our eyes on discontinuity in states, would this matter?
    dataset = h5py.File(args.training_set)
    states = np.array(dataset['observations'])

    if finetune:
        demo_starting_idxes, demo_states_flat = get_demo_states(args.env, args.demos, args.num_demos)

    reachability_model = ContrastiveFutureState(state_dimension=states.shape[-1],
                                                hidden_size=args.hidden_size,
                                                feature_size=args.feature_size).to(device)
    optimizer = torch.optim.Adam(reachability_model.parameters(), lr=args.lr)
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    save_cmd(args.save_dir)
    writer = SummaryWriter(log_dir=args.save_dir)

    batch_size = args.batch_size
    total_training = len(states) - args.horizon
    step = 0
    pre_finetune = True
    for epoch in range(args.training_epochs + finetune_epochs):
        idxes = np.arange(total_training)
        np.random.shuffle(idxes)
        for batch in range(total_training // batch_size):
            step += 1

            starting_states = states[idxes[batch * batch_size: (batch + 1) * batch_size]]

            if args.mode == 'exact':
                offset = args.horizon
            else:
                offset = np.random.randint(1, args.horizon + 1, size=batch_size)

            future_states = states[idxes[batch * batch_size: (batch + 1) * batch_size] + offset]

            if epoch >= args.training_epochs:
                if pre_finetune:
                    torch.save(reachability_model.state_dict(), args.save_dir + '/' + args.mode + '_model_prefinetune.pt')
                    pre_finetune = False
                batch_demo_starting_idx = np.random.choice(demo_starting_idxes, size=args.fine_tune_size)
                starting_states[-args.fine_tune_size:] = demo_states_flat[batch_demo_starting_idx]
                if args.mode == 'reachable':
                    offset = offset[-args.fine_tune_size:]
                future_states[-args.fine_tune_size:] = demo_states_flat[batch_demo_starting_idx + offset]

            starting_states = torch.Tensor(starting_states).to(device)
            future_states = torch.Tensor(future_states).to(device)

            contrastive_logits = reachability_model(starting_states, future_states)
            labels = torch.arange(contrastive_logits.shape[0]).long().to(device)
            loss = cross_entropy_loss(contrastive_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                writer.add_scalar('Loss/train', loss, step)
            print('\rEpoch: {}, batch: {}. Loss: {}'.format(epoch, batch, loss), end='')
            if epoch == args.training_epochs - 1:
                torch.save(reachability_model.state_dict(), args.save_dir + '/' + args.mode + '_blocked_model.pt')

    torch.save(reachability_model.state_dict(), args.save_dir + '/' + args.mode + '_fine_tuned_model.pt')
