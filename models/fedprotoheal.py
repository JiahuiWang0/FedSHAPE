import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import copy
import numpy as np
from utils.args import *
from models.utils.federated_model import FederatedModel

class FedProtoHEAL(FederatedModel):
    NAME = 'fedprotoheal'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedProtoHEAL, self).__init__(nets_list, args, transform)
        self.proto_weight = getattr(self.args, 'proto_weight', 1.0)
        self.global_protos = {}

        self.client_update = {}
        self.increase_history = {}
        self.mask_dict = {}
        self.euclidean_distance = {}
        self.previous_weights = {}
        self.previous_delta_weights = {}

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
        print(f"Round {self.epoch_index} Online Clients: {self.online_clients}")

        local_protos_list = {}

        for i in online_clients:
            local_protos = self._train_net(i, self.nets_list[i], priloader_list[i])
            local_protos_list[i] = local_protos

            if self.args.wHEAL == 1:
                net_params = self.nets_list[i].state_dict()
                global_params = self.global_net.state_dict()
                param_names = [name for name, _ in self.nets_list[i].named_parameters()]
                update_diff = {key: global_params[key] - net_params[key] for key in global_params}

                mask = self.consistency_mask(i, update_diff)
                self.mask_dict[i] = mask
                masked_update = {key: update_diff[key] * mask[key] for key in update_diff}
                self.client_update[i] = masked_update
                self.compute_distance(i, self.client_update[i], param_names)

        freq = self.get_params_diff_weights()
        self.aggregate_nets_parameter(freq)

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
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss().to(self.device)
        mse_loss = nn.MSELoss().to(self.device)

        local_protos = {}
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
                loss_proto = 0.0
                if len(self.global_protos) > 0:
                    features_proto_target = torch.zeros_like(features)
                    mask = torch.zeros(labels.size(0), dtype=torch.bool).to(self.device)
                    for k, label in enumerate(labels):
                        l = label.item()
                        if l in self.global_protos:
                            features_proto_target[k] = self.global_protos[l].to(self.device)
                            mask[k] = True
                    if mask.any():
                        loss_proto = mse_loss(features[mask], features_proto_target[mask])

                loss = loss_ce + self.proto_weight * loss_proto
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                optimizer.step()

                with torch.no_grad():
                    features_detached = features.detach()
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
