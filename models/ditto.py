import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel

class Ditto(FederatedModel):
    NAME = 'ditto'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(Ditto, self).__init__(nets_list, args, transform)
        self.ditto_lambda = getattr(self.args, 'ditto_lambda', 1.0)

        self.personalized_nets = {}

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

        for i in range(self.args.parti_num):
            self.personalized_nets[i] = copy.deepcopy(self.nets_list[0]).cpu()

    def loc_update(self, priloader_list):
        online_clients = self.online_clients_sequence[self.epoch_index]
        self.online_clients = online_clients
        print(f"Round {self.epoch_index} Online Clients: {self.online_clients}")

        self.global_net.to(self.device)
        self.global_net.eval()
        for param in self.global_net.parameters():
            param.requires_grad = False

        for i in online_clients:
            self.nets_list[i].load_state_dict(self.global_net.state_dict())
            self._train_global_branch(i, self.nets_list[i], priloader_list[i])

            p_net = self.personalized_nets[i].to(self.device)
            p_net.train()

            self._train_personalized_branch(i, p_net, self.global_net, priloader_list[i])

            self.personalized_nets[i] = copy.deepcopy(p_net).cpu()

        for param in self.global_net.parameters():
            param.requires_grad = True

        self.aggregate_nets()

    def _train_global_branch(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss().to(self.device)

        iterator = tqdm(range(self.local_epoch), desc=f"Client {index} Global Branch")
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

    def _train_personalized_branch(self, index, p_net, global_net, train_loader):
        optimizer = optim.SGD(p_net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss().to(self.device)

        iterator = tqdm(range(self.local_epoch), desc=f"Client {index} Ditto Branch")
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = p_net(images)

                task_loss = criterion(outputs, labels)

                prox_loss = 0.0
                for w_p, w_g in zip(p_net.parameters(), global_net.parameters()):
                    prox_loss += torch.sum((w_p - w_g) ** 2)

                loss = task_loss + (self.ditto_lambda / 2.0) * prox_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(p_net.parameters(), 10.0)
                optimizer.step()

                iterator.desc = f"Client {index} Ditto Loss={loss.item():.3f}"
