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

class FedProtoSPO(FederatedModel):
    NAME = 'fedprotospo'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedProtoSPO, self).__init__(nets_list, args, transform)
        self.proto_weight = getattr(self.args, 'proto_weight', 1.0)
        self.global_protos = {}

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        online_clients = self.online_clients_sequence[self.epoch_index]
        self.online_clients = online_clients
        print(f"Round {self.epoch_index} Online Clients: {self.online_clients}")

        local_protos_list = {}
        for i in online_clients:
            local_protos = self._train_net(i, self.nets_list[i], priloader_list[i])
            local_protos_list[i] = local_protos

        self.aggregate_nets()
        self.aggregate_protos(local_protos_list)

    def aggregate_protos(self, local_protos_list):
        new_global_protos = {}
        classes_count = {}
        for client_idx, protos in local_protos_list.items():
            for label, proto in protos.items():
                if label not in new_global_protos:
                    new_global_protos[label] = proto.clone()
                    classes_count[label] = 1
                else:
                    new_global_protos[label] += proto
                    classes_count[label] += 1
        for label in new_global_protos:
            new_global_protos[label] /= classes_count[label]
        self.global_protos = new_global_protos

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()

        spo_alpha = getattr(self.args, 'spo_alpha', 0.05)
        optimizer = SPOOptim(net.parameters(), base_optimizer=optim.SGD, alpha=spo_alpha,
                            lr=self.local_lr, momentum=0.9, weight_decay=1e-5)

        criterion = nn.CrossEntropyLoss().to(self.device)
        mse_loss = nn.MSELoss().to(self.device)

        local_protos = {}
        class_counts = {}

        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                def compute_loss(model_curr):
                    feats = model_curr.features(images)
                    out = model_curr(images)
                    loss_ce = criterion(out, labels)

                    loss_proto = 0.0
                    if len(self.global_protos) > 0:
                        features_proto_target = torch.zeros_like(feats)
                        mask = torch.zeros(labels.size(0), dtype=torch.bool).to(self.device)
                        for k, label in enumerate(labels):
                            l = label.item()
                            if l in self.global_protos:
                                features_proto_target[k] = self.global_protos[l].to(self.device)
                                mask[k] = True
                        if mask.any():
                            loss_proto = mse_loss(feats[mask], features_proto_target[mask])

                    return loss_ce + self.proto_weight * loss_proto, feats

                optimizer.zero_grad()
                loss, features_final = compute_loss(net)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)

                optimizer.generate_delta(zero_grad=True)

                loss_p, _ = compute_loss(net)
                loss_p.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)

                optimizer.step()

                with torch.no_grad():
                    features_detached = features_final.detach()
                    for k, label in enumerate(labels):
                        l = label.item()
                        if l not in local_protos:
                            local_protos[l] = features_detached[k].clone()
                            class_counts[l] = 1
                        else:
                            local_protos[l] += features_detached[k]
                            class_counts[l] += 1

                iterator.desc = "Local Participant %d loss = %0.3f" % (index, loss.item())

        final_local_protos = {}
        for label in local_protos:
            final_local_protos[label] = (local_protos[label] / class_counts[label]).cpu()

        return final_local_protos
