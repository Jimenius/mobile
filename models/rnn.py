import torch.nn
import torch.nn as nn


class GRU_Model(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_units=128, device=None):
        super(GRU_Model, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_units = hidden_units
        self.GRU = nn.GRU(self.obs_dim + self.action_dim, hidden_units, batch_first=True)
        self.device = device

    def forward(self, obs, last_acts, lens, pre_hidden=None):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if last_acts is None:
            last_acts = torch.zeros((1, len(lens), self.action_dim)).to(self.device)
        else:
            last_acts = torch.as_tensor(last_acts, device=self.device, dtype=torch.float32)
        sta_acs = torch.cat([obs, last_acts], dim=-1)
        if len(sta_acs.shape) == 2:
            sta_acs = sta_acs.unsqueeze(dim=1)
        # print(sta_acs.shape)
        packed = torch.nn.utils.rnn.pack_padded_sequence(sta_acs, lens, batch_first=True, enforce_sorted=False)
        if pre_hidden is None:
            pre_hidden = torch.zeros((1,len(lens),self.hidden_units)).to(self.device)
        else:
            pre_hidden = torch.as_tensor(pre_hidden, device=self.device, dtype=torch.float32)
        if len(pre_hidden.shape) == 2:
            pre_hidden = torch.unsqueeze(pre_hidden, dim=0)
        # print(pre_hidden.shape)
        output,_ = self.GRU(packed, pre_hidden)
        output,_ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output
