import torch
import torch.nn as nn
import models.utils as utils


class AudioMeanTextMean(nn.Module):
    
    def forward(self, input):
        sim = input["sim"]

        # sim: [bs, bs, a_len, t_len]
        batch_size, audio_len, text_len = sim.size(0), sim.size(2), sim.size(3)
        sim = sim.reshape(batch_size * batch_size, audio_len, text_len)
        audio_len = torch.as_tensor(input["audio_len"]).unsqueeze(1).expand(
            batch_size, batch_size).reshape(-1)
        sim = utils.mean_with_lens(sim, audio_len)
        # sim: [bs * bs, t_len]
        text_len = torch.as_tensor(input["text_len"]).repeat(batch_size)
        sim = utils.mean_with_lens(sim, text_len)
        sim = sim.reshape(batch_size, batch_size)

        return sim


class AudioMeanTextSum(nn.Module):
    
    def forward(self, input):
        sim = input["sim"]

        # sim: [bs, bs, a_len, t_len]
        batch_size, audio_len, text_len = sim.size(0), sim.size(2), sim.size(3)
        sim = sim.reshape(batch_size * batch_size, audio_len, text_len)
        audio_len = torch.as_tensor(input["audio_len"]).unsqueeze(1).expand(
            batch_size, batch_size).reshape(-1)
        sim = utils.mean_with_lens(sim, audio_len)
        # sim: [bs * bs, t_len]
        text_len = torch.as_tensor(input["text_len"]).repeat(batch_size)
        sim = utils.sum_with_lens(sim, text_len)
        sim = sim.reshape(batch_size, batch_size)

        return sim


class AudioMaxTextMean(nn.Module):

    def forward(self, input):
        sim = input["sim"]
        
        # sim: [bs, bs, a_len, t_len]
        batch_size, audio_len, text_len = sim.size(0), sim.size(2), sim.size(3)
        sim = sim.reshape(batch_size * batch_size, audio_len, text_len)
        audio_len = torch.as_tensor(input["audio_len"]).unsqueeze(1).expand(
            batch_size, batch_size).reshape(-1)
        sim = utils.max_with_lens(sim, audio_len)
        # sim: [bs*bs, t_len]
        text_len = torch.as_tensor(input["text_len"]).repeat(batch_size)
        sim = utils.mean_with_lens(sim, text_len)
        sim = sim.reshape(batch_size, batch_size)

        return sim


class AudioMaxTextMax(nn.Module):

    def forward(self, input):
        sim = input["sim"]
        
        # sim: [bs, bs, a_len, t_len]
        batch_size, audio_len, text_len = sim.size(0), sim.size(2), sim.size(3)
        sim = sim.reshape(batch_size * batch_size, audio_len, text_len)
        audio_len = torch.as_tensor(input["audio_len"]).unsqueeze(1).expand(
            batch_size, batch_size).reshape(-1)
        sim = utils.max_with_lens(sim, audio_len)
        # sim: [bs*bs, t_len]
        text_len = torch.as_tensor(input["text_len"]).repeat(batch_size)
        sim = utils.max_with_lens(sim, text_len)
        sim = sim.reshape(batch_size, batch_size)

        return sim


class AudioMaxTextSum(nn.Module):

    def forward(self, input):
        sim = input["sim"]
        
        # sim: [bs, bs, a_len, t_len]
        batch_size, audio_len, text_len = sim.size(0), sim.size(2), sim.size(3)
        sim = sim.reshape(batch_size * batch_size, audio_len, text_len)
        audio_len = torch.as_tensor(input["audio_len"]).unsqueeze(1).expand(
            batch_size, batch_size).reshape(-1)
        sim = utils.max_with_lens(sim, audio_len)
        text_len = torch.as_tensor(input["text_len"]).repeat(batch_size)
        sim = utils.sum_with_lens(sim, text_len)
        sim = sim.reshape(batch_size, batch_size)

        return sim


class AudioMaxTextMeanSum(nn.Module):

    def forward(self, input):
        sim = input["sim"]
        
        # sim: [bs, bs, a_len, t_len]
        batch_size, audio_len, text_len = sim.size(0), sim.size(2), sim.size(3)
        sim = sim.reshape(batch_size * batch_size, audio_len, text_len)
        audio_len = torch.as_tensor(input["audio_len"]).unsqueeze(1).expand(
            batch_size, batch_size).reshape(-1)
        sim = utils.max_with_lens(sim, audio_len)
        text_len = torch.as_tensor(input["text_len"]).repeat(batch_size)
        sim1 = utils.sum_with_lens(sim, text_len)
        sim2 = utils.mean_with_lens(sim, text_len)
        sim = sim1 + sim2
        sim = sim.reshape(batch_size, batch_size)

        return sim


class AudioLinearSoftTextMean(nn.Module):
    
    def forward(self, input):
        sim = input["sim"]
        # sim: [bs, bs, a_len, t_len]
        batch_size, audio_len, text_len = sim.size(0), sim.size(2), sim.size(3)
        sim = sim.reshape(batch_size * batch_size, audio_len, text_len)
        audio_len = torch.as_tensor(input["audio_len"]).unsqueeze(1).expand(
            batch_size, batch_size).reshape(-1)
        sim = utils.linear_softmax_with_lens(sim, audio_len)
        text_len = torch.as_tensor(input["text_len"]).repeat(batch_size)
        sim = utils.mean_with_lens(sim, text_len)
        sim = sim.reshape(batch_size, batch_size)
        if torch.isnan(sim).any():
            import pdb; pdb.set_trace()
        return sim


class AudioLinearSoftTextSum(nn.Module):
    
    def forward(self, input):
        sim = input["sim"]
        # sim: [bs, bs, a_len, t_len]
        batch_size, audio_len, text_len = sim.size(0), sim.size(2), sim.size(3)
        sim = sim.reshape(batch_size * batch_size, audio_len, text_len)
        audio_len = torch.as_tensor(input["audio_len"]).unsqueeze(1).expand(
            batch_size, batch_size).reshape(-1)
        sim = utils.linear_softmax_with_lens(sim, audio_len) # [bs*bs, t_len]
        text_len = torch.as_tensor(input["text_len"]).repeat(batch_size)
        sim = utils.sum_with_lens(sim, text_len)
        sim = sim.reshape(batch_size, batch_size)
        if torch.isnan(sim).any():
            import pdb; pdb.set_trace()
        return sim


class AudioExpSoftTextMean(nn.Module):
    
    def forward(self, input):
        sim = input["sim"]
        # sim: [bs, bs, a_len, t_len]
        batch_size, audio_len, text_len = sim.size(0), sim.size(2), sim.size(3)
        sim = sim.reshape(batch_size * batch_size, audio_len, text_len)
        audio_len = torch.as_tensor(input["audio_len"]).unsqueeze(1).expand(
            batch_size, batch_size).reshape(-1)
        sim = utils.exp_softmax_with_lens(sim, audio_len) # [bs*bs, t_len]
        text_len = torch.as_tensor(input["text_len"]).repeat(batch_size)
        sim = utils.mean_with_lens(sim, text_len)
        sim = sim.reshape(batch_size, batch_size)
        if torch.isnan(sim).any():
            import pdb; pdb.set_trace()
        return sim


class AudioExpSoftTextSum(nn.Module):
    
    def forward(self, input):
        sim = input["sim"]
        # sim: [bs, bs, a_len, t_len]
        batch_size, audio_len, text_len = sim.size(0), sim.size(2), sim.size(3)
        sim = sim.reshape(batch_size * batch_size, audio_len, text_len)
        audio_len = torch.as_tensor(input["audio_len"]).unsqueeze(1).expand(
            batch_size, batch_size).reshape(-1)
        sim = utils.exp_softmax_with_lens(sim, audio_len) # [bs*bs, t_len]
        text_len = torch.as_tensor(input["text_len"]).repeat(batch_size)
        sim = utils.sum_with_lens(sim, text_len)
        sim = sim.reshape(batch_size, batch_size)
        if torch.isnan(sim).any():
            import pdb; pdb.set_trace()
        return sim


class MultiTextLinearSoft(nn.Module):

    def forward(self, input):
        sim = input["sim"]
        return utils.linear_softmax_with_lens(sim.transpose(1, 2), input["audio_len"])


class MultiTextMax(nn.Module):

    def forward(self, input):
        sim = input["sim"]
        # [bs, n_txt, n_seg]
        return utils.max_with_lens(sim.transpose(1, 2), input["audio_len"])
