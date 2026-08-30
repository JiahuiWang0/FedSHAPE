import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
import copy
import numpy as np
from utils.args import *
from models.utils.federated_model import FederatedModel

class FedIDA(FederatedModel):
    NAME = 'fedida'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedIDA, self).__init__(nets_list, args, transform)
        self.ida_lambda = getattr(self.args, 'ida_lambda', 0.1)
        self.ida_beta = getattr(self.args, 'ida_beta', 0.01)

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        online_clients = self.online_clients_sequence[self.epoch_index]
        self.online_clients = online_clients
        print(f"Round {self.epoch_index} Online Clients: {self.online_clients}")

        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])

        self.aggregate_nets()

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()

        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)

        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = net(images)

                losses = criterion(outputs, labels)

                unique_labels, inverse_indices = torch.unique(labels, return_inverse=True)

                group_losses = []
                for idx, label in enumerate(unique_labels):
                    mask = (labels == label)
                    if mask.sum() > 0:
                        g_loss = losses[mask].mean()
                        group_losses.append(g_loss)

                if len(group_losses) > 0:
                    group_losses = torch.stack(group_losses)

                    avg_loss = group_losses.mean()

                    loss_var = torch.var(group_losses)

                    max_loss = torch.max(group_losses)

                    total_loss = avg_loss + self.ida_lambda * loss_var + self.ida_beta * max_loss
                else:
                    total_loss = losses.mean()

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)

                optimizer.step()
                iterator.desc = "Local Participant %d loss = %0.3f" % (index, total_loss.item())
