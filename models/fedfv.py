import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import copy
import numpy as np
from utils.args import *
from models.utils.federated_model import FederatedModel

class FedFV(FederatedModel):
    NAME = 'fedfv'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedFV, self).__init__(nets_list, args, transform)
        self.fv_alpha = getattr(self.args, 'fv_alpha', 0.1)

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        online_clients = self.online_clients_sequence[self.epoch_index]
        self.online_clients = online_clients
        print(f"Round {self.epoch_index} Online Clients: {self.online_clients}")

        client_updates = []
        client_losses = []

        global_w = {k: v.clone().detach() for k, v in self.global_net.state_dict().items()}

        for i in online_clients:
            loss = self._train_net(i, self.nets_list[i], priloader_list[i])
            client_losses.append(loss)

            local_w = self.nets_list[i].state_dict()
            update = {}
            for k in global_w.keys():
                update[k] = (local_w[k] - global_w[k]).cpu()
            client_updates.append(update)

        self.aggregate_nets_fedfv(client_updates, client_losses)

    def aggregate_nets_fedfv(self, client_updates, client_losses):
        num_clients = len(client_updates)
        if num_clients == 0:
            return

        flattened_updates = []
        for update in client_updates:
            flat = torch.cat([v.view(-1) for v in update.values()])
            flattened_updates.append(flat)

        sorted_indices = np.argsort(client_losses)[::-1]

        grads = [flattened_updates[i] for i in sorted_indices]

        final_update_vec = torch.zeros_like(grads[0])

        for i, g_i in enumerate(grads):

            g_i_mod = g_i.clone()
            for j in range(i):
                g_j = grads[j]
                inner_prod = torch.dot(g_i_mod, g_j)
                norm_sq = torch.dot(g_j, g_j)

                if inner_prod < 0 and norm_sq > 1e-12:
                    g_i_mod -= (inner_prod / norm_sq) * g_j

            final_update_vec += g_i_mod

        final_update_vec /= num_clients

        global_w = self.global_net.state_dict()
        offset = 0
        updated_w = {}

        for k, v in global_w.items():
            numel = v.numel()
            update_segment = final_update_vec[offset: offset + numel].view(v.shape)
            updated_w[k] = v + update_segment.to(self.device)
            offset += numel

        self.global_net.load_state_dict(updated_w)

        for net in self.nets_list:
            net.load_state_dict(updated_w)

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
