import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel

class Scaffold(FederatedModel):
    NAME = 'scaffold'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(Scaffold, self).__init__(nets_list, args, transform)
        self.c_global = []
        self.c_local = []

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

        for param in self.global_net.parameters():
            self.c_global.append(torch.zeros_like(param.data))

        for _ in range(self.args.parti_num):
            c_i = []
            for param in self.global_net.parameters():
                c_i.append(torch.zeros_like(param.data))
            self.c_local.append(c_i)

    def loc_update(self, priloader_list):
        online_clients = self.online_clients_sequence[self.epoch_index]
        self.online_clients = online_clients
        print(f"Round {self.epoch_index} Online Clients: {self.online_clients}")

        total_delta_c = [torch.zeros_like(param) for param in self.c_global]

        for i in online_clients:
            delta_c = self._train_net(i, self.nets_list[i], self.global_net, priloader_list[i])
            for j, d_c in enumerate(delta_c):
                total_delta_c[j].add_(d_c.to(total_delta_c[j].device))

        scale = 1.0 / len(online_clients)
        for j, param in enumerate(self.c_global):
            param.add_(total_delta_c[j] * scale)

        self.aggregate_nets()

    def _train_net(self, index, net, global_net, train_loader):
        net = net.to(self.device)
        global_model = copy.deepcopy(global_net).to(self.device)
        for param in global_model.parameters(): param.requires_grad = False

        c_g = [c.to(self.device) for c in self.c_global]
        c_i = [c.to(self.device) for c in self.c_local[index]]

        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss().to(self.device)

        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = net(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()

                for param, c_g_p, c_i_p in zip(net.parameters(), c_g, c_i):
                    if param.grad is not None:
                        param.grad.data.add_(c_g_p - c_i_p)

                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)

                optimizer.step()
                iterator.desc = "Local Participant %d loss = %0.3f" % (index, loss.item())

        delta_c = []
        K = self.local_epoch * len(train_loader)
        lr = self.local_lr
        if K == 0: K = 1

        with torch.no_grad():
            new_c_i = []
            for p_local, p_global, c_g_p, c_i_p in zip(net.parameters(), global_model.parameters(), c_g, c_i):
                term = (p_global.data - p_local.data) / (K * lr)
                c_new = c_i_p - c_g_p + term
                new_c_i.append(c_new)
                delta_c.append((c_new - c_i_p).cpu())
            self.c_local[index] = [c.cpu() for c in new_c_i]

        return delta_c
