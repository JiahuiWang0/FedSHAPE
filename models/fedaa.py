import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
import copy
import numpy as np
from utils.args import *
from models.utils.federated_model import FederatedModel

class FedAA(FederatedModel):
    NAME = 'fedaa'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedAA, self).__init__(nets_list, args, transform)
        self.aa_beta = getattr(self.args, 'aa_beta', 0.9)
        self.aa_temp = getattr(self.args, 'aa_temp', 1.0)

        self.client_rewards = np.zeros(self.args.parti_num)

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        online_clients = self.online_clients_sequence[self.epoch_index]
        self.online_clients = online_clients
        print(f"Round {self.epoch_index} Online Clients: {self.online_clients}")

        round_losses = []
        for i in online_clients:
            avg_loss = self._train_net(i, self.nets_list[i], priloader_list[i])
            round_losses.append(avg_loss)

        self.aggregate_adaptive(round_losses)

    def aggregate_adaptive(self, round_losses):
        total_clients = len(self.online_clients)
        if total_clients == 0: return

        epsilon = 1e-6
        current_rewards = 1.0 / (np.array(round_losses) + epsilon)

        for idx, client_idx in enumerate(self.online_clients):
            self.client_rewards[client_idx] = (self.aa_beta * self.client_rewards[client_idx] +
                                               (1 - self.aa_beta) * current_rewards[idx])

        online_rewards = self.client_rewards[self.online_clients]

        if len(online_rewards) > 1:
            online_rewards = (online_rewards - np.mean(online_rewards)) / (np.std(online_rewards) + epsilon)

        weights = F.softmax(torch.tensor(online_rewards / self.aa_temp), dim=0).numpy()

        ref_state = self.global_net.state_dict()
        accumulated_state = {}

        for key in ref_state:
            accumulated_state[key] = torch.zeros_like(ref_state[key], dtype=torch.float).to(self.device)

        for idx, client_idx in enumerate(self.online_clients):
            local_state = self.nets_list[client_idx].state_dict()
            w = weights[idx]
            for key in accumulated_state:
                accumulated_state[key] += local_state[key].to(self.device).float() * w

        final_state = {}
        for key, value in accumulated_state.items():
            if ref_state[key].dtype == torch.int64 or ref_state[key].dtype == torch.long:
                final_state[key] = value.long()
            else:
                final_state[key] = value

        self.global_net.load_state_dict(final_state)

        for net in self.nets_list:
            net.load_state_dict(final_state)

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()

        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss().to(self.device)

        epoch_loss = 0.0
        num_batches = 0

        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)

                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                iterator.desc = "Local Participant %d loss = %0.3f" % (index, loss.item())

        return epoch_loss / max(1, num_batches)
