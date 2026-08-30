import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel

class MoonQFFL(FederatedModel):
    NAME = 'moonqffl'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(MoonQFFL, self).__init__(nets_list, args, transform)
        self.mu = getattr(self.args, 'mu', 5.0)
        self.temperature = getattr(self.args, 'temperature', 0.5)
        self.previous_nets = {}

        self.q_param = getattr(self.args, 'q_param', 0.1)

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

        self.aggregate_qffl(client_losses)

    def aggregate_qffl(self, client_losses):
        total_clients = len(self.online_clients)
        if total_clients == 0: return

        losses_tensor = torch.tensor(client_losses).float()
        epsilon = 1e-4
        weights = (losses_tensor + epsilon).pow(self.q_param)
        weights = weights / weights.sum()

        ref_state = self.global_net.state_dict()
        accumulated_state = {}
        for key in ref_state:
            accumulated_state[key] = torch.zeros_like(ref_state[key], dtype=torch.float).to(self.device)

        for idx, i in enumerate(self.online_clients):
            w = weights[idx].item()
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

        return epoch_loss / max(1, batch_cnt)
