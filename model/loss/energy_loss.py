import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyLoss(nn.Module):
    def __init__(self, args):
        super(EnergyLoss, self).__init__()
        self.args = args
        if args.ahead_combine:
            from model.loss.automatic_weighted_loss import AutomaticWeightedLoss
            self.awl = AutomaticWeightedLoss(args, 2)
        if args.learnable_margin:
            assert args.ahead_combine
            self.m_in = torch.nn.Parameter(-torch.ones(1, requires_grad=True))
            self.m_out = torch.nn.Parameter(torch.ones(1, requires_grad=True))  # center of zero

    def forward(self, f_klogits, f_ulogits, e_klogits, e_ulogits):
        args = self.args
        energy_loss_scale = 0.1
        if args.ahead_combine:
            # sum
            k_energy_score = -torch.logsumexp(f_klogits, dim=1) # 不同类相似度的交叉熵概率之和 E_f
            u_energy_score = -torch.logsumexp(f_ulogits, dim=1)
            k_logits_score = -torch.logsumexp(e_klogits, dim=1) # E_c
            u_logits_score = -torch.logsumexp(e_ulogits, dim=1)

            k_energy_score = self.awl(k_energy_score, k_logits_score)   # E_c + E_f
            u_energy_score = self.awl(u_energy_score, u_logits_score)   # 越大越好(作为未知类的判别依据)

            if args.learnable_margin:   # margin value whether learnable
                l_energy = (torch.pow(F.relu(k_energy_score - self.m_in), 2).mean() +
                            torch.pow(F.relu(self.m_out - u_energy_score), 2).mean()
                            ) * energy_loss_scale + torch.pow(F.relu(self.m_in - self.m_out), 2)
            else:
                l_energy = (torch.pow(F.relu(k_energy_score - args.m_in), 2).mean() +
                            torch.pow(F.relu(args.m_out - u_energy_score), 2).mean()
                            ) * energy_loss_scale

        else:
            if args.energy_method == "sum": # 只计算 fine-grain 部分
                k_energy_score = -torch.logsumexp(f_klogits, dim=1)
                u_energy_score = -torch.logsumexp(f_ulogits, dim=1)
            else:
                k_energy_score = -torch.log(torch.max(torch.exp(f_klogits), dim=1)[0])
                u_energy_score = -torch.log(torch.max(torch.exp(f_ulogits), dim=1)[0])

            l_energy = (torch.pow(F.relu(k_energy_score - args.m_in), 2).mean() +
                        torch.pow(F.relu(args.m_out - u_energy_score), 2).mean()
                        ) * energy_loss_scale
        energy_score = torch.cat([k_energy_score, u_energy_score], dim=0)   # energy score be used to calculate auroc

        return l_energy, energy_score
