import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import copy
import numpy as np
from utils.args import *
from models.utils.federated_model import FederatedModel

class FedLF(FederatedModel):
    NAME = 'fedlf'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedLF, self).__init__(nets_list, args, transform)
        self.client_residuals = {}
        self.client_has_history = [False] * self.args.parti_num

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        online_clients = self.online_clients_sequence[self.epoch_index]
        self.online_clients = online_clients
        print(f"Round {self.epoch_index} Online Clients: {self.online_clients}")

        global_w = {k: v.clone().detach().cpu() for k, v in self.global_net.state_dict().items()}

        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])

            local_w = self.nets_list[i].state_dict()
            update = {}
            for k in global_w.keys():
                if 'num_batches_tracked' in k:
                    continue
                update[k] = (global_w[k] - local_w[k].cpu())

            self.client_residuals[i] = update
            self.client_has_history[i] = True

        self.aggregate_hybrid()

    def aggregate_hybrid(self):
        valid_clients = [i for i, has in enumerate(self.client_has_history) if has]
        if not valid_clients:
            return

        global_para = self.global_net.state_dict()
        layer_names = list(global_para.keys())

        trainable_layers = []
        stats_layers = []

        for k in layer_names:
            if 'num_batches_tracked' in k:
                continue
            if 'running' in k:
                stats_layers.append(k)
            else:
                trainable_layers.append(k)

        for layer in trainable_layers:
            layer_grads = []
            for client_idx in valid_clients:
                g = self.client_residuals[client_idx][layer]
                layer_grads.append(g.view(-1))

            if not layer_grads: continue

            grads_stack = torch.stack(layer_grads).double().to(self.device)

            if len(valid_clients) == 1:
                alphas = np.array([1.0])
            else:
                alphas = self._min_norm_solver_fw(grads_stack)

            agg_grad = torch.zeros_like(self.client_residuals[valid_clients[0]][layer])
            for idx, client_idx in enumerate(valid_clients):
                w = float(alphas[idx])
                if w > 1e-8:
                    agg_grad += w * self.client_residuals[client_idx][layer]

            current_weight = global_para[layer].to(self.device)
            update_step = agg_grad.to(self.device)
            global_para[layer] = current_weight - update_step

        if len(self.online_clients) > 0:
            for layer in stats_layers:
                agg_stat = torch.zeros_like(global_para[layer]).float().to(self.device)
                for client_idx in self.online_clients:
                    agg_stat += self.nets_list[client_idx].state_dict()[layer].to(self.device)

                global_para[layer] = agg_stat / len(self.online_clients)

        self.global_net.load_state_dict(global_para)

        for net in self.nets_list:
            net.load_state_dict(global_para)

    def _min_norm_solver_fw(self, vecs, max_iter=20, tol=1e-5):
        N = vecs.shape[0]
        device = vecs.device

        norms = torch.norm(vecs, dim=1, keepdim=True)
        vecs_normalized = vecs / (norms + 1e-10)

        alpha = torch.zeros(N, dtype=torch.double, device=device)
        min_idx = torch.argmin(torch.sum(vecs * vecs, dim=1))
        alpha[min_idx] = 1.0

        current_vec = vecs[min_idx].clone()

        for _ in range(max_iter):
            current_norm_sq = torch.dot(current_vec, current_vec)

            dots = torch.mv(vecs, current_vec)
            min_idx = torch.argmin(dots)
            dot_val = dots[min_idx]

            if current_norm_sq - dot_val < tol * current_norm_sq:
                break

            s = vecs[min_idx]
            d_vec = s - current_vec
            d_norm_sq = torch.dot(d_vec, d_vec)

            if d_norm_sq < 1e-10:
                gamma = 0.0
            else:
                gamma = -torch.dot(current_vec, d_vec) / d_norm_sq
                gamma = torch.clamp(gamma, 0.0, 1.0)

            alpha = (1 - gamma) * alpha
            alpha[min_idx] += gamma
            current_vec = (1 - gamma) * current_vec + gamma * s

        return alpha.cpu().numpy()

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

                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

                optimizer.step()
                iterator.desc = "Local Participant %d loss = %0.3f" % (index, loss.item())
