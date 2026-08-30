import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel

class FedProx(FederatedModel):
    NAME = 'fedprox'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedProx, self).__init__(nets_list, args, transform)
        self.mu = getattr(self.args, 'mu', 0.01)

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        online_clients = self.online_clients_sequence[self.epoch_index]
        self.online_clients = online_clients
        print(f"Round {self.epoch_index} Online Clients: {self.online_clients}")

        self.global_net.to(self.device)
        self.global_net.eval()
        for param in self.global_net.parameters():
            param.requires_grad = False

        for i in online_clients:
            self._train_net(i, self.nets_list[i], self.global_net, priloader_list[i])

        for param in self.global_net.parameters():
            param.requires_grad = True

        self.aggregate_nets()

    def _train_net(self, index, net, global_net, train_loader):
        net = net.to(self.device)
        net.train()

        optimizer = optim.SGD(
            net.parameters(),
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

                optimizer.zero_grad()

                outputs = net(images)
                task_loss = criterion(outputs, labels)

                proximal_term = 0.0
                for w, w_t in zip(net.parameters(), global_net.parameters()):
                    proximal_term += torch.norm(w - w_t, p=2) ** 2
                proximal_term = (self.mu / 2.0) * proximal_term

                loss = task_loss + proximal_term

                loss.backward()
                optimizer.step()

                iterator.desc = "Local Participant %d loss = %0.3f (Task: %.3f, Prox: %.3f)" % (
                    index, loss.item(), task_loss.item(), proximal_term.item())
