import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel

class MoonDIV(FederatedModel):
    NAME = 'moondiv'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(MoonDIV, self).__init__(nets_list, args, transform)
        self.mu = getattr(self.args, 'mu', 5.0)
        self.temperature = getattr(self.args, 'temperature', 0.5)
        self.previous_nets = {}
        self.client_update = {}
        self.mask_dict = {}
        self.euclidean_distance = {}
        self.previous_weights = {}
        self.previous_delta_weights = {}

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)
            self.previous_nets[_] = copy.deepcopy(net).cpu()

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
            param_names = [name for name, _ in self.nets_list[i].named_parameters()]
            update_diff = {key: global_params[key] - net_params[key] for key in global_params}

            self.mask_dict[i] = {key: torch.ones_like(val) for key, val in update_diff.items()}
            self.client_update[i] = update_diff
            self.compute_distance(i, self.client_update[i], param_names)

        for param in self.global_net.parameters(): param.requires_grad = True
        freq = self.get_params_diff_weights()
        self.aggregate_nets_parameter(freq)

    def _train_net(self, index, net, global_net, prev_net, train_loader):
        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss().to(self.device)
        cos = nn.CosineSimilarity(dim=-1)

        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                with torch.no_grad():
                    g_f = global_net.features(images).detach()
                    p_f = prev_net.features(images).detach()

                f = net.features(images)
                out = net(images)

                loss_sup = criterion(out, labels)
                pos = cos(f, g_f)
                neg = cos(f, p_f)
                logits_con = torch.cat((pos.unsqueeze(1), neg.unsqueeze(1)), dim=1) / self.temperature
                labels_con = torch.zeros(images.size(0)).long().to(self.device)
                loss_con = criterion(logits_con, labels_con)

                loss = loss_sup + self.mu * loss_con
                loss.backward()
                optimizer.step()
                iterator.desc = "Local Participant %d loss = %0.3f" % (index, loss.item())
