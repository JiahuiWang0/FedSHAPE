import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel

class FedBN(FederatedModel):
    NAME = 'fedbn'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedBN, self).__init__(nets_list, args, transform)

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

        self.aggregate_nets_fedbn()

    def aggregate_nets_fedbn(self):
        total_clients = len(self.online_clients)
        if total_clients == 0:
            return

        first_net_state = self.nets_list[self.online_clients[0]].state_dict()
        aggregate_keys = []

        for key in first_net_state.keys():
            if 'bn' in key or 'downsample.1' in key or 'running_' in key or 'num_batches_tracked' in key:
                continue
            else:
                aggregate_keys.append(key)

        avg_w = {}
        for key in aggregate_keys:
            avg_w[key] = torch.zeros_like(first_net_state[key])

        for i in self.online_clients:
            net_state = self.nets_list[i].state_dict()
            for key in aggregate_keys:
                avg_w[key] += net_state[key].to(self.device)

        for key in aggregate_keys:
            avg_w[key] = (avg_w[key] / total_clients).cpu()

        global_state = self.global_net.state_dict()
        global_state.update(avg_w)
        self.global_net.load_state_dict(global_state)

        for i in range(self.args.parti_num):
            local_state = self.nets_list[i].state_dict()
            for key in aggregate_keys:
                local_state[key] = avg_w[key].clone()

            self.nets_list[i].load_state_dict(local_state)

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

                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)

                optimizer.step()
                iterator.desc = "Local Participant %d loss = %0.3f" % (index, loss.item())
