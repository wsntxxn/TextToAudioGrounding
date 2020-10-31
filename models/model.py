import torch
import torch.nn as nn
import torch.nn.functional as F

class Siamese(nn.Module):

    def __init__(self, inputdim, embeddim, audioEncoder, phraseEncoder, use_siamese=True):
        super(Siamese, self).__init__()
        self.audioEncoder = audioEncoder
        self.phraseEncoder = phraseEncoder
        self.use_siamese = use_siamese
        self.projection = nn.Sequential(
            nn.Linear(inputdim, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, embeddim)
        )

    def forward(self, *input):
        if len(input) == 2:
            audio_feats, query_embeds = input
        elif len(input) == 3:
            audio_feats, wordids, phraselens = input
            query_embeds = self.phraseEncoder(wordids, phraselens)
        audio_embeds = self.audioEncoder(audio_feats)
        audio_embeds = F.normalize(audio_embeds, dim=-1)
        query_embeds = F.normalize(query_embeds, dim=-1)
        if self.use_siamese:
            audio_embeds = self.projection(audio_embeds)
            query_embeds = self.projection(query_embeds)
        # audio_embeds: [N, T, E]    query_embeds: [N, E]
        diff = audio_embeds - query_embeds.unsqueeze(1) # diff: [N, T, E]
        dis = torch.exp(- torch.norm(diff, dim=-1)) # dis: [N, T]
        return dis


if __name__ == "__main__":
    from pytorch_model_summary import summary
    from .AudioEncoder import CRNNEncoder
    encoder = CRNNEncoder(64, 256)
    model = Siamese(256, 1024, encoder)
    print(model)
    print(summary(model, torch.randn(4, 501, 256), torch.randn(4, 256), show_input=True))
