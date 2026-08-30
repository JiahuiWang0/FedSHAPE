from typing import Optional
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from models.utils.federated_model import FederatedModel
from utils.conf import data_path

class FedProxMix(FederatedModel):
    NAME = 'fedproxmix'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedProxMix, self).__init__(nets_list, args, transform)

        self.mu = getattr(args, 'mu', 0.01)

        self.public_epoch = getattr(args, 'public_epoch', 1)
        self.public_lr = getattr(args, 'public_lr', 0.001)
        self.fccl_offdiag_weight = getattr(args, 'fccl_offdiag_weight', 0.0051)
        self.temp = getattr(args, 'temp', 1.0)
        self.local_dis_power = getattr(args, 'local_dis_power', 1.0)

        self.public_len = getattr(args, 'public_len', 5000)
        self.public_batch_size = getattr(args, 'public_batch_size', 256)
        self.pub_aug = getattr(args, 'pub_aug', 'weak')

        self.beta = getattr(args, 'beta', 0.5)

        self.prev_nets_list = []
        self.public_loader: Optional[DataLoader] = None
        self._fccl_optimizers = {}

        self.client_update = {}
        self.euclidean_distance = {}
        self.previous_weights = {}
        self.previous_delta_weights = {}

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for net in self.nets_list:
            net.load_state_dict(global_w)

        for j in range(self.args.parti_num):
            self.prev_nets_list.append(copy.deepcopy(self.nets_list[j]))

        self.public_loader = self._prepare_public_loader()

    def _prepare_public_loader(self) -> Optional[DataLoader]:
        try:
            weak_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            train_dataset = CIFAR10(root=data_path(), train=True, download=True, transform=weak_transform)

            n_train = len(train_dataset)
            idxs = np.random.permutation(n_train)
            if self.public_len is not None and self.public_len > 0:
                idxs = idxs[:self.public_len]

            train_sampler = SubsetRandomSampler(idxs)
            return DataLoader(train_dataset, batch_size=self.public_batch_size, sampler=train_sampler, num_workers=4)
        except Exception as e:
            print(f"Warning: Failed to load CIFAR-10: {e}")
            return None

    def col_update(self, communication_idx):
        if self.public_loader is None: return

        for _ in range(self.public_epoch):
            for _, (images, _) in enumerate(self.public_loader):
                images = images.to(self.device)

                all_logits = []
                for net in self.nets_list:
                    net.to(self.device)
                    net.train()
                    all_logits.append(net(images))

                avg_logits = torch.mean(torch.stack(all_logits), dim=0).detach()

                for net_idx, net in enumerate(self.nets_list):
                    if net_idx not in self._fccl_optimizers:
                        self._fccl_optimizers[net_idx] = optim.Adam(net.parameters(), lr=self.public_lr)
                    optimizer = self._fccl_optimizers[net_idx]

                    logits = all_logits[net_idx]

                    z_1_bn = (logits - logits.mean(0)) / (logits.std(0) + 1e-5)
                    z_2_bn = (avg_logits - avg_logits.mean(0)) / (avg_logits.std(0) + 1e-5)
                    c = (z_1_bn.T @ z_2_bn) / len(images)

                    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                    off_diag = self._off_diagonal(c).pow_(2).sum()
                    loss_colla = on_diag + self.fccl_offdiag_weight * off_diag

                    optimizer.zero_grad()
                    loss_colla.backward()
                    optimizer.step()

    def loc_update(self, priloader_list):
        online_clients = self.online_clients_sequence[self.epoch_index]
        self.online_clients = online_clients
        print(f"Round {self.epoch_index} Online Clients: {self.online_clients}")

        global_params = self.global_net.state_dict()

        for i in online_clients:
            self._train_net_prox_fntd(i, self.nets_list[i], self.prev_nets_list[i], priloader_list[i])

            net_params = self.nets_list[i].state_dict()
            param_names = [name for name, _ in self.nets_list[i].named_parameters()]

            update_diff = {key: net_params[key] - global_params[key] for key in global_params}

            self.client_update[i] = update_diff

            self.compute_distance(i, self.client_update[i], param_names)

        freq = self.get_adaptive_weights()
        self.aggregate_nets_weighted(freq)

        for i in online_clients:
            self.prev_nets_list[i].load_state_dict(self.nets_list[i].state_dict())

    def _train_net_prox_fntd(self, index, net, prev_net, train_loader):
        T = self.temp
        net.to(self.device).train()
        prev_net.to(self.device).eval()

        self.global_net.to(self.device)
        global_weight_collector = list(self.global_net.parameters())

        optimizer = optim.Adam(net.parameters(), lr=self.local_lr, weight_decay=1e-5)

        criterionCE = nn.CrossEntropyLoss()
        criterionKL = nn.KLDivLoss(reduction='batchmean')

        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = net(images)
                with torch.no_grad():
                    prev_outputs = prev_net(images)

                bs, class_num = outputs.shape
                mask = torch.ones(bs, class_num).to(self.device).scatter_(1, labels.view(-1, 1), 0)

                soft_out = F.softmax(outputs / T, dim=1)
                soft_prev = F.softmax(prev_outputs / T, dim=1)

                nt_soft_out = soft_out[mask.bool()].view(bs, class_num - 1)
                nt_soft_prev = soft_prev[mask.bool()].view(bs, class_num - 1)

                loss_fntd = criterionKL(torch.log(nt_soft_out + 1e-8), nt_soft_prev) * (T ** 2)

                loss_ce = criterionCE(outputs, labels)

                proximal_term = 0.0
                for param, global_param in zip(net.parameters(), global_weight_collector):
                    proximal_term += (param - global_param).norm(2) ** 2

                loss_prox = (self.mu / 2) * proximal_term

                total_loss = loss_ce + (self.local_dis_power * loss_fntd) + loss_prox

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                iterator.desc = f"Client {index} CE={loss_ce.item():.2f} FNTD={loss_fntd.item():.2f} Prox={loss_prox.item():.2f}"

    def compute_distance(self, index, update_diff, param_names):
        dist = 0
        for key in update_diff:
            if key in param_names:
                dist += torch.norm(update_diff[key]).item()
        self.euclidean_distance[index] = dist

    def get_adaptive_weights(self):
        weight_dict = {}
        total_dist = sum(self.euclidean_distance.values())
        if total_dist == 0: total_dist = 1e-5

        for client in self.online_clients:
            client_distance = self.euclidean_distance[client]

            delta_weight = (1 - self.beta) * (self.previous_delta_weights.get(client, 0)) + \
                           self.beta * ((client_distance) / total_dist)

            new_weight = self.previous_weights.get(client, 1.0 / self.online_num) + delta_weight
            new_weight = max(new_weight, 0.01)

            weight_dict[client] = new_weight
            self.previous_weights[client] = new_weight
            self.previous_delta_weights[client] = delta_weight

        total_weight = sum(weight_dict.values())
        for client in self.online_clients:
            weight_dict[client] /= total_weight

        return weight_dict

    def aggregate_nets_weighted(self, freq):
        global_params = self.global_net.state_dict()
        global_params_new = copy.deepcopy(global_params)

        for param_key in global_params_new:
            weighted_update_sum = 0

            for client_id in self.online_clients:
                update = self.client_update[client_id][param_key]
                weight = freq[client_id]
                weighted_update_sum = weighted_update_sum + (update * weight)

            global_params_new[param_key] = global_params_new[param_key] + weighted_update_sum

        self.global_net.load_state_dict(global_params_new)
        for i in self.online_clients:
            self.nets_list[i].load_state_dict(global_params_new)

    @staticmethod
    def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
