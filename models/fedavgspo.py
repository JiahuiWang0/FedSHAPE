import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel

class SPOOptim(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, alpha=0.05, **kwargs):
        defaults = dict(alpha=alpha, **kwargs)
        super(SPOOptim, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def generate_delta(self, zero_grad=False):
        device = self.param_groups[0]["params"][0].device
        grad_norm = torch.norm(
            torch.stack([
                (1.0 * p.grad).norm(p=2).to(device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None]), p=2
        )
        for group in self.param_groups:
            scale = group["alpha"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                delta = 1.0 * p.grad * scale.to(p)
                p.add_(delta)
                self.state[p]["delta"] = delta

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["delta"])

        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

class FedAvgSPO(FederatedModel):
    NAME = 'fedavgspo'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedAvgSPO, self).__init__(nets_list, args, transform)

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

        alpha = getattr(self.args, 'spo_alpha', 0.05)

        optimizer = SPOOptim(
            net.parameters(),
            base_optimizer=optim.SGD,
            alpha=alpha,
            lr=self.local_lr,
            momentum=0.9,
            weight_decay=1e-5
        )

        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)

        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = net(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()

                optimizer.generate_delta(zero_grad=True)

                outputs_p = net(images)
                loss_p = criterion(outputs_p, labels)
                loss_p.backward()

                optimizer.step()

                iterator.desc = "Local Participant %d loss = %0.3f" % (index, loss.item())
