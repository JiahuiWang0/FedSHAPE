import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel

class FedDynEMA(FederatedModel):
    NAME = 'feddynema'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedDynEMA, self).__init__(nets_list, args, transform)
        self.dyn_alpha = getattr(self.args, 'dyn_alpha', 0.01)
        self.server_state = {}
        self.local_grads = {}
        self.client_update = {}
        self.increase_history = {}
        self.mask_dict = {}

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

        for name, param in self.global_net.named_parameters():
            self.server_state[name] = torch.zeros_like(param.data)
        for i in range(self.args.parti_num):
            self.local_grads[i] = {}
            for name, param in self.global_net.named_parameters():
                self.local_grads[i][name] = torch.zeros_like(param.data)

        if not self.increase_history:
            for i in range(self.args.parti_num):
                self.increase_history[i] = {}

    def loc_update(self, priloader_list):
        online_clients = self.online_clients_sequence[self.epoch_index]
        self.online_clients = online_clients
        old_global_net = copy.deepcopy(self.global_net)
        old_global_w = old_global_net.state_dict()

        for i in online_clients:
            self._train_net(i, self.nets_list[i], self.global_net, priloader_list[i])
            net_params = self.nets_list[i].state_dict()
            global_params = self.global_net.state_dict()
            update_diff = {key: global_params[key] - net_params[key] for key in global_params}

            mask = self.consistency_mask(i, update_diff)
            self.mask_dict[i] = mask
            self.client_update[i] = {key: update_diff[key] * mask[key] for key in update_diff}

        self.aggregate_nets_parameter(freq=None)

        avg_w = copy.deepcopy(self.global_net.state_dict())
        for name, param in self.global_net.named_parameters():
            diff = avg_w[name] - old_global_w[name]
            h = self.server_state[name].to(self.device)
            h.sub_(self.dyn_alpha * diff.to(self.device))
            self.server_state[name] = h.cpu()
            param.data.sub_((1.0 / self.dyn_alpha) * h.to(self.device))

        final_global_w = self.global_net.state_dict()
        for i in online_clients:
            self.nets_list[i].load_state_dict(final_global_w)

    def _train_net(self, index, net, global_net, train_loader):
        net = net.to(self.device)
        global_net.to(self.device)
        global_net.eval()
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss().to(self.device)
        local_grad = self.local_grads[index]

        global_params_dict = dict(global_net.named_parameters())

        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                out = net(images)
                loss_task = criterion(out, labels)
                loss_algo = 0.0
                loss_l2 = 0.0
                for name, param in net.named_parameters():
                    if name in local_grad:
                        loss_algo -= torch.sum(local_grad[name].to(self.device) * param)
                    if name in global_params_dict:
                        loss_l2 += torch.sum((param - global_params_dict[name]) ** 2)
                loss = loss_task + loss_algo + (self.dyn_alpha / 2.0) * loss_l2
                loss.backward()
                optimizer.step()
                iterator.desc = "Local Participant %d loss = %0.3f" % (index, loss.item())

        with torch.no_grad():
            new_local_grad = {}
            for name, param in net.named_parameters():
                global_param = global_params_dict[name]
                diff = param - global_param
                new_grad = local_grad[name].to(self.device) - self.dyn_alpha * diff
                new_local_grad[name] = new_grad.cpu()
            self.local_grads[index] = new_local_grad
