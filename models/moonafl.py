import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel

class MoonAFL(FederatedModel):
    NAME = 'moonafl'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(MoonAFL, self).__init__(nets_list, args, transform)
        self.mu = getattr(self.args, 'mu', 5.0)
        self.temperature = getattr(self.args, 'temperature', 0.5)
        self.previous_nets = {}

        self.afl_lr = getattr(self.args, 'afl_lr', 0.01)
        self.dynamic_lambdas = torch.ones(self.args.parti_num) / self.args.parti_num

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)
            self.previous_nets[_] = copy.deepcopy(net).cpu()

    def loc_update(self, priloader_list):
        online_clients = self.online_clients_sequence[self.epoch_index]
        self.online_clients = online_clients
        print(f"Round {self.epoch_index} Online Clients: {self.online_clients}")

        self.global_net.to(self.device)
        self.global_net.eval()
        for param in self.global_net.parameters():
            param.requires_grad = False

        client_losses = []
        for i in online_clients:
            prev_net = self.previous_nets[i].to(self.device)
            prev_net.eval()
            for param in prev_net.parameters():
                param.requires_grad = False

            loss = self._train_net(i, self.nets_list[i], self.global_net, prev_net, priloader_list[i])
            client_losses.append(loss)

            self.previous_nets[i] = copy.deepcopy(self.nets_list[i]).cpu()
            prev_net.cpu()

        for param in self.global_net.parameters():
            param.requires_grad = True

        current_lambdas = self.dynamic_lambdas.clone()
        for idx, client_idx in enumerate(online_clients):
            current_lambdas[client_idx] += self.afl_lr * client_losses[idx]

        self.dynamic_lambdas = F.relu(current_lambdas)
        if self.dynamic_lambdas.sum() > 0:
            self.dynamic_lambdas /= self.dynamic_lambdas.sum()
        else:
            self.dynamic_lambdas = torch.ones(self.args.parti_num) / self.args.parti_num

        self.aggregate_afl(online_clients)

    def aggregate_afl(self, online_clients):
        online_weights = self.dynamic_lambdas[online_clients]
        if online_weights.sum() > 0:
            agg_weights = online_weights / online_weights.sum()
        else:
            agg_weights = torch.ones(len(online_clients)) / len(online_clients)

        ref_state = self.global_net.state_dict()
        accumulated_state = {}
        for key in ref_state:
            accumulated_state[key] = torch.zeros_like(ref_state[key], dtype=torch.float).to(self.device)

        for idx, i in enumerate(online_clients):
            w = agg_weights[idx].item()
            net_para = self.nets_list[i].state_dict()
            for key in accumulated_state:
                accumulated_state[key] += net_para[key].to(self.device).float() * w

        final_state = {}
        for key, value in accumulated_state.items():
            if ref_state[key].dtype in [torch.int64, torch.long]:
                final_state[key] = value.long()
            else:
                final_state[key] = value

        self.global_net.load_state_dict(final_state)
        for net in self.nets_list:
            net.load_state_dict(final_state)

    def _train_net(self, index, net, global_net, prev_net, train_loader):
        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss().to(self.device)
        cos = nn.CosineSimilarity(dim=-1)

        epoch_loss = 0.0
        batch_cnt = 0
        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                pro1 = net.features(images)
                outputs = net(images)
                loss_sup = criterion(outputs, labels)

                with torch.no_grad():
                    pro2 = global_net.features(images)
                    pro3 = prev_net.features(images)

                pos_sim = cos(pro1, pro2) / self.temperature
                neg_sim = cos(pro1, pro3) / self.temperature

                logits_con = torch.cat((pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)), dim=1)
                labels_con = torch.zeros(images.size(0)).long().to(self.device)
                loss_con = criterion(logits_con, labels_con)

                loss = loss_sup + self.mu * loss_con
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                optimizer.step()

                epoch_loss += loss.item()
                batch_cnt += 1
                iterator.desc = "Local Participant %d loss = %0.3f" % (index, loss.item())

        return epoch_loss / max(1, batch_cnt)
