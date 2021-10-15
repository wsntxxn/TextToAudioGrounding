import torch
import torch.nn as nn
import torch.nn.functional as F


class GroundingModel(nn.Module):

    def __init__(self, audio_encoder, text_encoder, similarity_fn):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.similarity_fn = similarity_fn

    def forward(self, input_dict):
        raise NotImplementedError


class NaiveModel(GroundingModel):

    def __init__(self, audio_encoder, text_encoder, similarity_fn):
        super().__init__(audio_encoder, text_encoder, similarity_fn)
        assert audio_encoder.embed_dim == text_encoder.embed_dim

    def forward(self, input_dict):
        audio_out = self.audio_encoder(input_dict) # [N, T, E]
        audio = audio_out["audio"]
        audio_len = audio_out["audio_len"]
        text = self.text_encoder(input_dict) # [N, E]
        score = self.similarity_fn(audio, text) # [N, T]
        return {
            "score": score,
            "length": audio_len
        }


class SiameseModel(GroundingModel):

    def __init__(self, audio_encoder, text_encoder, similarity_fn, proj_dim):
        super().__init__(audio_encoder, text_encoder, similarity_fn)
        assert audio_encoder.embed_dim == text_encoder.embed_dim
        embed_dim = audio_encoder.embed_dim
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, proj_dim)
        )

    def forward(self, input_dict):
        audio_out = self.audio_encoder(input_dict) # [N, T, E]
        audio = audio_out["audio"]
        audio_len = audio_out["audio_len"]
        text = self.text_encoder(input_dict) # [N, E]
        audio = self.projection(audio)
        text = self.projection(text)
        score = self.similarity_fn(audio, text) # [N, T]

        return {
            "score": score,
            "length": audio_len
        }


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    from models.audio_encoder import CrnnEncoder
    from models.text_encoder import EmbeddingMeanEncoder
    from models.score import ExponentialNegativeL2
    vocab_size = 1500
    audio_encoder = CrnnEncoder(64, 256)
    text_encoder = EmbeddingMeanEncoder(vocab_size, 256)
    similarity_fn = ExponentialNegativeL2()
    model = SiameseModel(audio_encoder, text_encoder, similarity_fn, 256, 1024)
    print(model)
    input_dict = {
        "audio_feat": torch.randn(4, 501, 64),
        "audio_feat_len": torch.tensor([501, 501, 451, 401]),
        "text": torch.empty(4, 3).random_(vocab_size),
        "text_len": torch.tensor([2, 3, 3, 2])
    }
    output = model(input_dict)
    print("score: ", output["score"].shape, "length: ", output["length"].shape)
