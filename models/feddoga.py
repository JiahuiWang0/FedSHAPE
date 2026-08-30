import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
import copy
import numpy as np
from utils.args import *
from models.utils.federated_model import FederatedModel

class FedDOGA(FederatedModel):
    NAME = 'feddoga'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedDOGA, self).__init__(nets_list, args, transform)
        self.doga_gamma = getattr(self.args, 'doga_gamma', 2.0)
        self.clip_threshold = getattr(self.args, 'doga_clip', 1.0)

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        online_clients = self.online_clients_sequence[self.epoch_index]
        self.online_clients = online_clients
        print(f"Round {self.epoch_index} Online Clients: {self.online_clients}")

        self.global_net.to(self.device)
        self.global_net.eval()
        for p in self.global_net.parameters():
            p.requires_grad = False

        client_discrepancies = []
        client_updates = []

        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])

            local_state = self.nets_list[i].state_dict()
            global_state = self.global_net.state_dict()

            discrepancy = 0.0
            update_norm = 0.0
            for key in local_state:
                if 'num_batches_tracked' in key: continue
                diff = (local_state[key] - global_state[key]).float()
                discrepancy += torch.norm(diff, p=2).item() ** 2

            discrepancy = np.sqrt(discrepancy)
            client_discrepancies.append(discrepancy)
            client_updates.append(local_state)

        self.aggregate_doga(client_updates, client_discrepancies)

    def aggregate_doga(self, client_updates, discrepancies):
        total_clients = len(client_updates)
        if total_clients == 0: return

        discrepancies = np.array(discrepancies)

        median_disc = np.median(discrepancies)
        clipped_discrepancies = np.clip(discrepancies, 0, median_disc * 3.0)

        exp_disc = np.exp(self.doga_gamma * clipped_discrepancies)
        weights = exp_disc / np.sum(exp_disc)

        first_state = client_updates[0]
        aggregated_state = {}

        for key in first_state:
            aggregated_state[key] = torch.zeros_like(first_state[key]).float().to(self.device)

        for idx, state in enumerate(client_updates):
            w = weights[idx]
            for key in state:
                aggregated_state[key] += state[key].to(self.device) * w

        self.global_net.load_state_dict(aggregated_state)

        for net in self.nets_list:
            net.load_state_dict(aggregated_state)

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()

        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss().to(self.device)

        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(net.parameters(), self.clip_threshold)

                optimizer.step()
                iterator.desc = "Local Participant %d loss = %0.3f" % (index, loss.item())
