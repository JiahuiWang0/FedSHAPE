import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel

class SPOOptim(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, alpha=0.00005, **kwargs):
        defaults = dict(alpha=alpha, **kwargs)
        super(SPOOptim, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
    @torch.no_grad()
    def generate_delta(self, zero_grad=False):
        device = self.param_groups[0]["params"][0].device
        grad_norm = torch.norm(torch.stack([(1.0 * p.grad).norm(p=2).to(device) for group in self.param_groups for p in group["params"] if p.grad is not None]), p=2)
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

class MoonSPOEMA(FederatedModel):
    NAME = 'moonspoema'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(MoonSPOEMA, self).__init__(nets_list, args, transform)
        self.mu = getattr(self.args, 'mu', 5.0)
        self.temperature = getattr(self.args, 'temperature', 0.5)
        self.previous_nets = {}
        self.client_update = {}
        self.increase_history = {}
        self.mask_dict = {}

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)
            self.previous_nets[_] = copy.deepcopy(net).cpu()
        if not self.increase_history:
            for i in range(self.args.parti_num):
                self.increase_history[i] = {}

    def loc_update(self, priloader_list):
        online_clients = self.online_clients_sequence[self.epoch_index]
        self.online_clients = online_clients
        self.global_net.to(self.device)
        self.global_net.eval()
        for param in self.global_net.parameters(): param.requires_grad = False

        for i in online_clients:
            prev_net = self.previous_nets[i].to(self.device)
            prev_net.eval()
            for param in prev_net.parameters(): param.requires_grad = False

            self._train_net(i, self.nets_list[i], self.global_net, prev_net, priloader_list[i])
            self.previous_nets[i] = copy.deepcopy(self.nets_list[i]).cpu()
            prev_net.cpu()

            net_params = self.nets_list[i].state_dict()
            global_params = self.global_net.state_dict()
            update_diff = {key: global_params[key] - net_params[key] for key in global_params}

            mask = self.consistency_mask(i, update_diff)
            self.mask_dict[i] = mask
            self.client_update[i] = {key: update_diff[key] * mask[key] for key in update_diff}

        for param in self.global_net.parameters(): param.requires_grad = True
        self.aggregate_nets_parameter(freq=None)

    def _train_net(self, index, net, global_net, prev_net, train_loader):
        net = net.to(self.device)
        net.train()
        alpha = getattr(self.args, 'spo_alpha', 0.05)
        optimizer = SPOOptim(net.parameters(), base_optimizer=optim.SGD, alpha=alpha, lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss().to(self.device)
        cos = nn.CosineSimilarity(dim=-1)

        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                with torch.no_grad():
                    g_f = global_net.features(images).detach()
                    p_f = prev_net.features(images).detach()

                def compute_loss(model_curr):
                    f = model_curr.features(images)
                    out = model_curr(images)
                    loss_sup = criterion(out, labels)
                    pos = cos(f, g_f)
                    neg = cos(f, p_f)
                    logits_con = torch.cat((pos.unsqueeze(1), neg.unsqueeze(1)), dim=1) / self.temperature
                    labels_con = torch.zeros(images.size(0)).long().to(self.device)
                    loss_con = criterion(logits_con, labels_con)
                    return loss_sup + self.mu * loss_con

                optimizer.zero_grad()
                loss = compute_loss(net)
                loss.backward()
                optimizer.generate_delta(zero_grad=True)
                loss_p = compute_loss(net)
                loss_p.backward()
                optimizer.step()
                iterator.desc = "Local Participant %d loss = %0.3f" % (index, loss.item())
