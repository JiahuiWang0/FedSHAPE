import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
import copy
import numpy as np
from utils.args import *
from models.utils.federated_model import FederatedModel

class AFL(FederatedModel):
    NAME = 'afl'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(AFL, self).__init__(nets_list, args, transform)
        self.afl_lr = getattr(self.args, 'afl_lr', 0.01)
        self.dynamic_lambdas = torch.ones(self.args.parti_num) / self.args.parti_num

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        online_clients = self.online_clients_sequence[self.epoch_index]
        self.online_clients = online_clients
        print(f"Round {self.epoch_index} Online Clients: {self.online_clients}")

        client_losses = []
        for i in online_clients:
            loss = self._train_net(i, self.nets_list[i], priloader_list[i])
            client_losses.append(loss)

        current_lambdas = self.dynamic_lambdas.clone()
        for idx, client_idx in enumerate(online_clients):
            current_lambdas[client_idx] += self.afl_lr * client_losses[idx]

        self.dynamic_lambdas = F.relu(current_lambdas)
        if self.dynamic_lambdas.sum() > 0:
            self.dynamic_lambdas /= self.dynamic_lambdas.sum()
        else:
            self.dynamic_lambdas = torch.ones(self.args.parti_num) / self.args.parti_num

        self.aggregate_afl(online_clients)

    def aggregate_afl(self, online_clients):
        online_weights = self.dynamic_lambdas[online_clients]

        if online_weights.sum() > 0:
            agg_weights = online_weights / online_weights.sum()
        else:
            agg_weights = torch.ones(len(online_clients)) / len(online_clients)

        print(f"  AFL Weights: {agg_weights.numpy()}")

        ref_state = self.global_net.state_dict()
        accumulated_state = {}

        for key in ref_state:
            accumulated_state[key] = torch.zeros_like(ref_state[key], dtype=torch.float).to(self.device)

        for idx, i in enumerate(online_clients):
            w = agg_weights[idx].item()
            net_para = self.nets_list[i].state_dict()
            for key in accumulated_state:
                accumulated_state[key] += net_para[key].to(self.device).float() * w

        final_state = {}
        for key, value in accumulated_state.items():
            if ref_state[key].dtype in [torch.int64, torch.long]:
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
        batch_cnt = 0
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
                batch_cnt += 1
                iterator.desc = "Local Participant %d loss = %0.3f" % (index, loss.item())
        return epoch_loss / max(1, batch_cnt)
