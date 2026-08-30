import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import copy
import numpy as np
from utils.args import *
from models.utils.federated_model import FederatedModel

class FedFA(FederatedModel):
    NAME = 'fedfa'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedFA, self).__init__(nets_list, args, transform)
        self.fa_weight = getattr(self.args, 'fa_weight', 1.0)

        self.global_anchors = {}

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        online_clients = self.online_clients_sequence[self.epoch_index]
        self.online_clients = online_clients
        print(f"Round {self.epoch_index} Online Clients: {self.online_clients}")

        local_anchors_list = {}

        for i in online_clients:
            local_anchors = self._train_net(i, self.nets_list[i], priloader_list[i])
            local_anchors_list[i] = local_anchors

        self.aggregate_nets()

        self.aggregate_anchors(local_anchors_list)

    def aggregate_anchors(self, local_anchors_list):
        new_global_anchors = {}
        classes_count = {}

        for client_idx, anchors in local_anchors_list.items():
            for label, anchor_vec in anchors.items():
                if label not in new_global_anchors:
                    new_global_anchors[label] = anchor_vec.clone()
                    classes_count[label] = 1
                else:
                    new_global_anchors[label] += anchor_vec
                    classes_count[label] += 1

        for label in new_global_anchors:
            new_global_anchors[label] /= classes_count[label]

        self.global_anchors = new_global_anchors

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()

        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss().to(self.device)
        mse_loss = nn.MSELoss().to(self.device)

        running_anchors = {}
        class_counts = {}

        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                features = net.features(images)
                outputs = net(images)

                loss_ce = criterion(outputs, labels)

                loss_fa = 0.0
                if len(self.global_anchors) > 0:
                    target_features = torch.zeros_like(features)
                    mask = torch.zeros(labels.size(0), dtype=torch.bool).to(self.device)

                    for k, label in enumerate(labels):
                        l = label.item()
                        if l in self.global_anchors:
                            target_features[k] = self.global_anchors[l].to(self.device)
                            mask[k] = True

                    if mask.any():
                        loss_fa = mse_loss(features[mask], target_features[mask])

                loss = loss_ce + self.fa_weight * loss_fa

                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                optimizer.step()

                with torch.no_grad():
                    features_detached = features.detach()
                    for k, label in enumerate(labels):
                        l = label.item()
                        if l not in running_anchors:
                            running_anchors[l] = features_detached[k].clone()
                            class_counts[l] = 1
                        else:
                            running_anchors[l] += features_detached[k]
                            class_counts[l] += 1

                iterator.desc = "Local Participant %d loss = %0.3f" % (index, loss.item())

        final_local_anchors = {}
        for label in running_anchors:
            final_local_anchors[label] = (running_anchors[label] / class_counts[label]).cpu()

        return final_local_anchors
