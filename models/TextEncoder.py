import torch
import torch.nn as nn

class EmbeddingEncoder(nn.Module):
    def __init__(self, vocabsize, embeddim):
        super(EmbeddingEncoder, self).__init__()
        self.embedding = nn.Embedding(vocabsize, embeddim)
        nn.init.kaiming_uniform_(self.embedding.weight)

    def load_pretrained_embedding(self, weight, tune=True):
        assert weight.shape[0] == self.embedding.weight.size(0), "vocabulary size mismatch!"
        weight = torch.as_tensor(weight).float()
        self.embedding.weight = nn.Parameter(weight)
        for para in self.embedding.parameters():
            para.requires_grad = tune

    def forward(self, *input):
        wordids, lens = input
        lens = torch.as_tensor(lens)
        # print(wordids.shape)
        N, maxlen = wordids.size()
        embeds = self.embedding(wordids)

        idxs = torch.arange(maxlen, device="cpu").repeat(N).view(N, maxlen)
        mask = (idxs < lens.view(-1, 1)).to(embeds.device)

        embeds = embeds * mask.unsqueeze(-1)
        embeds = embeds.sum(1) / lens.unsqueeze(1).to(embeds.device)
        return embeds
