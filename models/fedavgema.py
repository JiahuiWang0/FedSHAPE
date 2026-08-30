import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel

class FedAvgEMA(FederatedModel):
    NAME = 'fedavgema'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedAvgEMA, self).__init__(nets_list, args, transform)
        self.client_update = {}
        self.increase_history = {}
        self.mask_dict = {}

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)
        if not self.increase_history:
            for i in range(self.args.parti_num):
                self.increase_history[i] = {}

    def loc_update(self, priloader_list):
        online_clients = self.online_clients_sequence[self.epoch_index]
        self.online_clients = online_clients

        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])

            net_params = self.nets_list[i].state_dict()
            global_params = self.global_net.state_dict()
            update_diff = {key: global_params[key] - net_params[key] for key in global_params}

            mask = self.consistency_mask(i, update_diff)
            self.mask_dict[i] = mask
            self.client_update[i] = {key: update_diff[key] * mask[key] for key in update_diff}

        self.aggregate_nets_parameter(freq=None)

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
                outputs = net(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iterator.desc = "Local Participant %d loss = %0.3f" % (index, loss.item())
