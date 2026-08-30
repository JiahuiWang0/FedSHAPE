import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel

class FedDynQFFL(FederatedModel):
    NAME = 'feddynqffl'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedDynQFFL, self).__init__(nets_list, args, transform)
        self.dyn_alpha = getattr(self.args, 'dyn_alpha', 0.01)
        self.q_param = getattr(self.args, 'q_param', 0.1)
        self.server_state = {}
        self.local_grads = {}

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

    def loc_update(self, priloader_list):
        online_clients = self.online_clients_sequence[self.epoch_index]
        self.online_clients = online_clients
        print(f"Round {self.epoch_index} Online Clients: {self.online_clients}")

        old_global_net = copy.deepcopy(self.global_net)
        old_global_w = old_global_net.state_dict()

        client_losses = []
        for i in online_clients:
            loss = self._train_net(i, self.nets_list[i], self.global_net, priloader_list[i])
            client_losses.append(loss)

        self.aggregate_qffl(client_losses)

        avg_w = self.global_net.state_dict()

        for name, param in self.global_net.named_parameters():
            diff = avg_w[name] - old_global_w[name]

            h = self.server_state[name].to(self.device)
            h.sub_(self.dyn_alpha * diff.to(self.device))
            self.server_state[name] = h.cpu()

            param.data.sub_((1.0 / self.dyn_alpha) * h.to(self.device))

        final_state = self.global_net.state_dict()
        for net in self.nets_list:
            net.load_state_dict(final_state)

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

    def _train_net(self, index, net, global_net, train_loader):
        net = net.to(self.device)
        global_net.to(self.device)
        global_net.eval()
        net.train()

        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss().to(self.device)

        local_grad = self.local_grads[index]

        epoch_loss = 0.0
        batch_cnt = 0
        iterator = tqdm(range(self.local_epoch))

        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = net(images)
                loss_task = criterion(outputs, labels)

                loss_algo = 0.0
                loss_l2 = 0.0

                for name, param in net.named_parameters():
                    if name in local_grad:
                        grad_term = local_grad[name].to(self.device)
                        loss_algo -= torch.sum(grad_term * param)

                    if name in dict(global_net.named_parameters()):
                        global_param = dict(global_net.named_parameters())[name]
                        loss_l2 += torch.sum((param - global_param) ** 2)

                loss = loss_task + loss_algo + (self.dyn_alpha / 2.0) * loss_l2

                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                optimizer.step()

                epoch_loss += loss.item()
                batch_cnt += 1
                iterator.desc = "Local Participant %d loss = %0.3f" % (index, loss.item())

        with torch.no_grad():
            new_local_grad = {}
            for name, param in net.named_parameters():
                global_param = dict(global_net.named_parameters())[name]
                diff = param - global_param

                prev_grad = local_grad[name].to(self.device)
                new_grad = prev_grad - self.dyn_alpha * diff
                new_local_grad[name] = new_grad.cpu()

            self.local_grads[index] = new_local_grad

        return epoch_loss / max(1, batch_cnt)
