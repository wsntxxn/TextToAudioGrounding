from collections import OrderedDict
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from audio_text_retrieval_models.net_vlad import NetVLAD, NeXtVLAD, NetRVLAD
from audio_text_retrieval_models.net_vf import NetVF
from audio_text_retrieval_models.base import BaseModel
from audio_text_retrieval_models.transformer import Transformer


class Net(BaseModel):
    def __init__(self,
                 audio_dims,
                 text_dim,
                 feat_aggregation,
                 vlad_clusters,
                 max_position_embeddings,
                 hidden_size,
                 transformer_width,
                 transformer_heads,
                 cross_num_hidden_layers,
                 ghost_clusters,
                 freeze_weights,
                 l2renorm,
                 verbose,
                 test_caption_mode,
                 num_layers,
                 text_agg):
        super().__init__()

        # sanity checks on the features that may be vladded, add feature dimension
        # if new feature being evaluated
        pre_vlad_feat_sizes = {
            "lms": 128, "vggish": 128, "vggsound": 512,
            "panns_cnn10": 512, "panns_cnn14": 2048, "panns_wavegram_logmel_cnn14": 2048
        }
        pre_vlad_feat_sizes = {
            key: val for key, val in pre_vlad_feat_sizes.items()
            if feat_aggregation[key]["temporal"] in ["vlad", "rvlad", "vf"]
        }

        vlad_feat_sizes = {key: val for key, val in vlad_clusters.items()}

        self.audio_pooling = nn.ModuleDict()
        self.frame_position_embeddings = nn.ModuleDict()

        self.feat_aggregation = feat_aggregation
        self.modalities = list(audio_dims.keys())
        for mod in audio_dims.keys():
            if feat_aggregation[mod]["temporal"] in ["vlad", "rvlad", "vf"]:
                feature_size = audio_dims[mod][0] // vlad_clusters[mod]
                msg = f"expected {pre_vlad_feat_sizes[mod]} for {mod} features atm"
                assert feature_size == pre_vlad_feat_sizes[mod], msg
                if feat_aggregation[mod]["temporal"] == "vlad":
                    self.audio_pooling[mod] = NetVLAD(
                        feature_size = feature_size,
                        cluster_size = vlad_clusters[mod],
                    )
                elif feat_aggregation[mod]["temporal"] == "rvlad":
                    self.audio_pooling[mod] = NetRVLAD(
                        feature_size = feature_size,
                        cluster_size = vlad_clusters[mod],
                    )
                
                elif feat_aggregation[mod]["temporal"] == "vf":
                    self.audio_pooling[mod] = NetVF(
                        feature_size = feature_size,
                        cluster_size = vlad_clusters[mod],
                    )
            
            elif feat_aggregation[mod]["temporal"] in ["meanP", "maxP"]:
                pass
            
            elif feat_aggregation[mod]["temporal"] == "seqLSTM":
                self.audio_pooling[mod] = nn.LSTM(
                    input_size=audio_dims[mod][1],
                    hidden_size=hidden_size[mod],
                    batch_first=True,
                    bidirectional=False,
                    num_layers=num_layers)
                self.hidden_size = hidden_size
                self.num_layers = num_layers
            elif feat_aggregation[mod]["temporal"] == "seqTransf":
                self.frame_position_embeddings[mod] = nn.Embedding(
                    max_position_embeddings[mod],
                    transformer_width[mod])
                self.fc_t = nn.Linear(audio_dims[mod][1], transformer_width[mod])
                self.transformer = Transformer(width = transformer_width[mod],
                                               layers = cross_num_hidden_layers,
                                               heads = transformer_heads)
                self.audio_pooling[mod] = self.transformer


        self.text_pooling = nn.Sequential()
        self.text_agg = text_agg
        if text_agg == "vlad":
            self.text_pooling = NetVLAD(
                feature_size=text_dim,
                cluster_size=vlad_clusters["text"],
                ghost_clusters=ghost_clusters["text"],
            )
            text_dim = self.text_pooling.out_dim
        
        elif text_agg == "rvlad":
            self.text_pooling = NetRVLAD(
                feature_size=text_dim,
                cluster_size=vlad_clusters["text"],
            )
            text_dim = self.text_pooling.out_dim
        
        elif text_agg == "vf":
            self.text_pooling = NetVF(
                                feature_size=text_dim,
                                cluster_size=vlad_clusters["text"],
            )
            text_dim = self.text_pooling.out_dim
 
        elif text_agg in ["meanP", "maxP"]:
            self.text_pooling = nn.Sequential()

        elif text_agg == "seqLSTM":
            self.text_pooling = nn.LSTM(input_size=text_dim,
                                        hidden_size=hidden_size["text"],
                                        batch_first=True,
                                        bidirectional=False,
                                        num_layers=num_layers)
            text_dim = hidden_size["text"]

        elif text_agg == "seqTransf":
            self.text_position_embeddings = nn.Embedding(
                                            max_position_embeddings["text"],
                                            transformer_width["text"])
            self.fc_text = nn.Linear(text_dim, transformer_width["text"])
            self.transformer = Transformer(width = transformer_width["text"],
                                           layers = cross_num_hidden_layers,
                                           heads = transformer_heads)
            self.text_pooling = self.transformer
            text_dim = transformer_width["text"]
        self.match = MatchModule(
            audio_dims=audio_dims,
            text_dim=text_dim,
            hidden_size=hidden_size,
            feat_aggregation=feat_aggregation,
            freeze_weights=freeze_weights,
            l2renorm=l2renorm,
            verbose=verbose,
            test_caption_mode = test_caption_mode,
        )
 
    def _max_pooling_for_similarity(self, expert):
        expert_out, _ = torch.max(expert, dim=1)
        return expert_out

    def _mean_pooling_for_similarity(self, expert, expert_mask):
        expert_mask_un = expert_mask.to(dtype=torch.float).unsqueeze(-1)
        expert = expert * expert_mask_un
        expert_mask_un_sum = torch.sum(expert_mask_un, dim=1, dtype=torch.float)
        expert_mask_un_sum[expert_mask_un_sum == 0.] = 1.
        expert_out = torch.sum(expert, dim=1) / expert_mask_un_sum
        return expert_out

    def forward(self, experts, text, expert_masks, text_token_masks, ind):

        
        aggregated_experts = OrderedDict()
        for mod in self.modalities:
            if self.feat_aggregation[mod]['temporal'] == "meanP":
                aggregated_experts[mod] = experts[mod]
            
            elif self.feat_aggregation[mod]['temporal'] == "maxP":
                aggregated_experts[mod] = self._max_pooling_for_similarity(experts[mod])

            elif self.feat_aggregation[mod]['temporal'] in ["vlad", "rvlad", "vf"]:
                aggregated_experts[mod] = self.audio_pooling[mod](experts[mod]) 
            
            elif self.feat_aggregation[mod]['temporal'] == "seqLSTM":
                audio_original = experts[mod]
                device = audio_original.device
                lengths = torch.sum(expert_masks[mod],dim=-1).cpu()
                lengths[lengths == 0.] = 1.
                audio_output = pack_padded_sequence(audio_original,
                                                    lengths,
                                                    batch_first=True,
                                                    enforce_sorted=False)
                audio_output, _ = self.audio_pooling[mod](audio_output)
                if self.training: 
                    self.audio_pooling[mod].flatten_parameters()
                audio_output, _ = pad_packed_sequence(audio_output, batch_first=True)
                I = torch.zeros(audio_output.size(0),
                    audio_original.size(1) - audio_output.size(1), audio_output.size(2)).to(device)
            
                aggregated_experts[mod] = torch.cat((audio_output, I), dim=1)

            elif self.feat_aggregation[mod]['temporal'] == "seqTransf":
                # Sequential type: Transformer Encoder
                audio_original = experts[mod]
                audio_original = self.fc_t(audio_original)
                seq_length = audio_original.size(1)
                position_ids = torch.arange(seq_length, dtype=torch.long, 
                                            device = audio_original.device)
                position_ids = position_ids.unsqueeze(0).expand(audio_original.size(0), -1)
                frame_position_embeddings = self.frame_position_embeddings[mod](position_ids)
                aggregated_experts[mod] = audio_original + frame_position_embeddings

                extended_audio_mask = (1.0 - expert_masks[mod].unsqueeze(1))*-1000000.0
                extended_audio_mask = extended_audio_mask.expand(-1, expert_masks[mod].size(1), -1)
                aggregated_experts[mod] = aggregated_experts[mod].permute(1, 0, 2) # NLD -> LND
                aggregated_experts[mod] = self.audio_pooling[mod](aggregated_experts[mod], extended_audio_mask)
                aggregated_experts[mod] = aggregated_experts[mod].permute(1, 0, 2) # LND -> NLD
            if len(aggregated_experts[mod].size()) > 2:
                aggregated_experts[mod] = aggregated_experts[mod] / (aggregated_experts[mod].norm(p=2, dim=-1, keepdim = True) + 1e-8)
                aggregated_experts[mod] = self._mean_pooling_for_similarity(aggregated_experts[mod], expert_masks[mod])
                aggregated_experts[mod] = aggregated_experts[mod] / (aggregated_experts[mod].norm(p=2,dim = -1, keepdim =True) + 1e-8)
 
        B, captions_per_audio, max_words, text_feat_dim = text.size()
        text = text.view(B * captions_per_audio, max_words, text_feat_dim)
        text_token_masks = text_token_masks.view(B * captions_per_audio, -1)
        
        if isinstance(self.text_pooling, NetVLAD):
            kwargs = {"mask": text_token_masks}
        else:
            kwargs = {}
        if self.text_agg  == "meanP":
            aggregated_text  = text
        elif self.text_agg == "maxP":
            aggregated_text = self._max_pooling_for_similarity(text)
        elif self.text_agg in ["vlad", "rvlad", "vf"]:
            aggregated_text = self.text_pooling(text, **kwargs)
        elif self.text_agg == "seqLSTM":
            text_original = text
            lengths = torch.sum(text_token_masks,dim=-1).cpu()
            lengths[lengths == 0.] = 1.
            text_output = pack_padded_sequence(text_original,
                                               lengths,
                                               batch_first=True,
                                               enforce_sorted=False)
            text_output, _ = self.text_pooling(text_output)
            if self.training: self.text_pooling.flatten_parameters()
            text_output, _ = pad_packed_sequence(text_output, batch_first=True)
            I = torch.zeros(text_output.size(0),
                text.size(1)-text_output.size(1), text_output.size(2)).to(device)
            
            aggregated_text = torch.cat((text_output,I), dim=1)
        elif self.text_agg == "seqTransf":
            # Sequential type: Transformer Encoder
            text_original = text
            text_original = self.fc_text(text)
            seq_length = text_original.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device = text_original.device)
            position_ids = position_ids.unsqueeze(0).expand(text_original.size(0), -1)
            text_position_embeddings = self.text_position_embeddings(position_ids)
            aggregated_text = text_original + text_position_embeddings

            extended_text_mask = (1.0 - text_token_masks.unsqueeze(1)) * -1000000.0
            extended_text_mask = extended_text_mask.expand(-1, text_token_masks.size(1), -1)
            aggregated_text = aggregated_text.permute(1, 0, 2) # NLD -> LND
            aggregated_text = self.text_pooling(aggregated_text, extended_text_mask)
            aggregated_text = aggregated_text.permute(1, 0, 2) # LND -> NLD
             
        if len(aggregated_text.size()) > 2:
            aggregated_text = aggregated_text / (aggregated_text.norm(
                p=2, dim=-1, keepdim=True) + 1e-8) 
            aggregated_text = self._mean_pooling_for_similarity(aggregated_text,
                text_token_masks)
            aggregated_text = aggregated_text / (aggregated_text.norm(
                p=2, dim=-1, keepdim=True) + 1e-8)

        aggregated_text = aggregated_text.view(B, captions_per_audio, -1)
        return self.match(aggregated_text, aggregated_experts, ind)


class AudioTextModel(BaseModel):
    def __init__(self,
                 audio_encoders,
                 text_encoder,
                 shared_dims,
                 freeze_weights,
                 l2renorm,
                 verbose,
                 test_caption_mode):
        super().__init__()

        self.audio_encoders = nn.ModuleDict()
        self.text_encoder = text_encoder

        dummy_audio = torch.randn(1, 32000)
        audio_dims = {}
        for encoder_id, audio_encoder in audio_encoders.items():
            self.audio_encoders[encoder_id] = audio_encoder
            audio_dims[encoder_id] = audio_encoder(dummy_audio)["clip_emb"].size(-1)
        dummy_text = torch.randn(2, 1, 300)
        text_dim = self.text_encoder(dummy_text)["clip_emb"].size(-1)

        # self.match = MatchModule2(
            # audio_dims=audio_dims,
            # text_dim=text_dim,
            # shared_dims=shared_dims,
            # freeze_weights=freeze_weights,
            # l2renorm=l2renorm,
            # verbose=verbose,
            # test_caption_mode=test_caption_mode,
        # )
 
    def forward(self, waveform, text, chunk_size=128):

        batch_size = text.shape[0]
        num_captions = text.shape[1]

        if batch_size <= chunk_size:
            audio_embs = OrderedDict()
            for encoder_id, audio_encoder in self.audio_encoders.items():
                audio_embs[encoder_id] = audio_encoder(waveform)["clip_emb"]
     
            text = text.reshape(batch_size * num_captions, -1, text.size(-1))
            text_emb = self.text_encoder(text)["clip_emb"]
            text_emb = text_emb.view(batch_size, num_captions, -1)
        else:
            partitions = math.ceil(batch_size / chunk_size)
            audio_embs = {encoder_id: None for encoder_id in self.audio_encoders}
            text_emb_chunks = []
            for chunk_idx in range(partitions):
                chunk_start = chunk_idx * chunk_size
                chunk_stop = (chunk_idx + 1) * chunk_size
                for encoder_id, audio_encoder in self.audio_encoders.items():
                    audio_emb_chunk = audio_encoder(waveform[chunk_start: chunk_stop])["clip_emb"]
                    if audio_embs[encoder_id] is None:
                        audio_embs[encoder_id] = audio_emb_chunk
                    else:
                        audio_embs[encoder_id] = torch.cat((audio_embs[encoder_id], audio_emb_chunk))
                text_chunk = text[chunk_start: chunk_stop]
                cur_chunk_size = text_chunk.shape[0]
                text_emb_chunk = self.text_encoder(text_chunk)["clip_emb"]
                text_emb_chunk = text_emb_chunk.view(cur_chunk_size, num_captions, -1)
                text_emb_chunks.append(text_emb_chunk)
            text_emb = torch.cat(text_emb_chunks)
            del text_emb_chunks
                
        return {
            "audio_embs": audio_embs,
            "text_emb": text_emb
        }
        # ind = {encoder_id: torch.ones(1) for encoder_id in self.audio_encoders.keys()}
        # print("text embedding: ", text_emb.shape)
        # print("audio embedding: ", audio_embs["wavegram_logmel_cnn14"].shape)
        # output = self.match(text_emb, audio_embs, ind)
        # print("similarity: ", output["cross_view_conf_matrix"].shape)
        # return output


class MatchModule(nn.Module):

    def __init__(self,
                 audio_dims,
                 text_dim,
                 feat_aggregation,
                 hidden_size,
                 freeze_weights,
                 l2renorm,
                 verbose,
                 test_caption_mode):
        super().__init__()
        self.modalities = list(audio_dims.keys())
        self.text_dim = text_dim
        self.feat_aggregation = feat_aggregation
        self.freeze_weights = freeze_weights
        self.l2renorm = l2renorm
        self.verbose = verbose
        self.test_caption_mode = test_caption_mode
        num_mods = len(audio_dims)
        if freeze_weights == False:
            self.moe_fc = nn.Linear(text_dim, num_mods)
        self.moe_weights = torch.ones(1,num_mods) / num_mods
        use_bns = [True for modality in self.modalities]
        self.repeat_temporal = {}
        for mod in self.modalities:
            self.repeat_temporal[mod] = 1
        in_dims = []
        agg_dims = []
        for mod in self.modalities:
            if feat_aggregation[mod]['temporal'] in ["vlad", "rvlad", "vf"]:
                in_dims.append(audio_dims[mod][0] * self.repeat_temporal[mod])
                agg_dims.append(audio_dims[mod][1] * self.repeat_temporal[mod])
            elif feat_aggregation[mod]['temporal'] in ["seqLSTM", "meanP","maxP"]:
                in_dims.append(hidden_size[mod])
                agg_dims.append(audio_dims[mod][1])
            elif feat_aggregation[mod]['temporal'] == "seqTransf":
                in_dims.append(audio_dims[mod][0])
                agg_dims.append(audio_dims[mod][0])
        
        gated_audio_embds = [GatedEmbeddingUnit(in_dim, out_dim, use_bn) for
            in_dim, out_dim, use_bn in zip(in_dims, agg_dims, use_bns)]
        text_out_dims = agg_dims
        self.audio_GU = nn.ModuleList(gated_audio_embds)
        gated_text_embds = [GatedEmbeddingUnit(text_dim, out_dim, use_bn=True) for
                           out_dim in text_out_dims]
        self.text_GU = nn.ModuleList(gated_text_embds)

    def compute_moe_weights(self, text, ind):
        # compute weights for all captions (including when assigned K captions to
        # the same video)
        B, K, D = text.shape
        M = len(self.modalities)
        msg = f"expected between 1 and 10 modalities, found {M} ({self.modalities})"
        assert 1 <= M <= 10, msg

        # Treat each caption independently in the softmax (which runs over modalities)
        text = text.view(B * K, D)
        if self.freeze_weights:
            moe_weights = self.moe_weights.repeat(B, K, 1)
            if text.is_cuda:
                moe_weights = moe_weights.cuda()
        else:
            moe_weights = self.moe_fc(text)  # BK x D -> BK x M
            moe_weights = F.softmax(moe_weights, dim=1)
            moe_weights = moe_weights.view(B, K, M)

        if self.verbose:
            print("--------------------------------")
            for idx, key in enumerate(self.modalities):
                msg = "{}: mean: {:.3f}, std: {:.3f}, min: {:.3f}, max: {:.3f}"
                msg = msg.format(
                    key,
                    moe_weights[:, :, idx].mean().item(),
                    moe_weights[:, :, idx].std().item(),
                    moe_weights[:, :, idx].min().item(),
                    moe_weights[:, :, idx].max().item(),
                )
                print(msg)
        return moe_weights
    
    def forward(self, text, experts, ind):
       
        text_embd = {}

        # Unroll repeated captions into present minibatch
        B, captions_per_audio, feat_dim = text.size()
        text = text.view(B * captions_per_audio, feat_dim)
        for modality, layer in zip(self.modalities, self.text_GU):
           # NOTE: Due to the batch norm, the gated units are sensitive to passing
           # in a lot of zeroes, so we do the masking step after the forwards pass
           text_ = layer(text)

           # We always assume that text is available for retrieval
           text_ = text_.view(B, captions_per_audio, -1)

           text_embd[modality] = text_
        text = text.view(B, captions_per_audio, -1)

        # MOE weights computation + normalization 
        # - note that we use the first caption
        # sample to predict the weights
        moe_weights = self.compute_moe_weights(text, ind=ind)

        if self.l2renorm:
            for modality in self.modalities:
                norm = experts[modality].norm(p=2, dim=-1, keepdim=True)
                experts[modality] = experts[modality].div(norm + 0.0000001).clip(-1000, 1000)
        for modality, layer in zip(self.modalities, self.audio_GU):
            experts[modality] = layer(experts[modality])
        if self.training:
            merge_caption_similiarities = "avg"
        else:
            merge_caption_similiarities = self.test_caption_mode
        
        cross_view_conf_matrix = sharded_cross_view_inner_product(
            ind=ind,
            vid_embds=experts,
            text_embds=text_embd,
            l2renorm=self.l2renorm,
            text_weights=moe_weights,
            subspaces=self.modalities,
            merge_caption_similiarities=merge_caption_similiarities,
            # loss_func = "MaxMarginRankingLoss2"
        )
 
        return {
            "modalities": self.modalities,
            "cross_view_conf_matrix": cross_view_conf_matrix,
            "text_embds": text_embd,
            "audio_embds": experts
        }


class MatchModule2(nn.Module):

    def __init__(self,
                 audio_dims,
                 text_dim,
                 shared_dims,
                 freeze_weights,
                 l2renorm,
                 verbose,
                 test_caption_mode):
        super().__init__()
        assert audio_dims.keys() == shared_dims.keys()
        self.text_dim = text_dim
        self.freeze_weights = freeze_weights
        self.l2renorm = l2renorm
        self.verbose = verbose
        self.test_caption_mode = test_caption_mode
        self.modalities = list(audio_dims.keys())
        num_mods = len(audio_dims)
        if freeze_weights == False:
            self.moe_fc = nn.Linear(text_dim, num_mods)
        self.moe_weights = torch.ones(1, num_mods) / num_mods
        
        gated_audio_embeds, gated_text_embeds = [], []
        for encoder_id in self.modalities:
            in_dim = audio_dims[encoder_id]
            out_dim = shared_dims[encoder_id]
            gated_audio_embeds.append(GatedEmbeddingUnit(in_dim, out_dim, True))
            gated_text_embeds.append(GatedEmbeddingUnit(text_dim, out_dim, True))
        self.audio_GU = nn.ModuleList(gated_audio_embeds)
        self.text_GU = nn.ModuleList(gated_text_embeds)

    def compute_moe_weights(self, text, ind):
        # compute weights for all captions (including when assigned K captions to
        # the same video)
        B, K, D = text.shape
        M = len(self.modalities)
        msg = f"expected between 1 and 10 modalities, found {M} ({self.modalities})"
        assert 1 <= M <= 10, msg

        # Treat each caption independently in the softmax (which runs over modalities)
        text = text.view(B * K, D)
        if self.freeze_weights:
            moe_weights = self.moe_weights.repeat(B, K, 1)
            if text.is_cuda:
                moe_weights = moe_weights.cuda()
        else:
            moe_weights = self.moe_fc(text)  # BK x D -> BK x M
            moe_weights = F.softmax(moe_weights, dim=1)
            moe_weights = moe_weights.view(B, K, M)

        if self.verbose:
            print("--------------------------------")
            for idx, key in enumerate(self.modalities):
                msg = "{}: mean: {:.3f}, std: {:.3f}, min: {:.3f}, max: {:.3f}"
                msg = msg.format(
                    key,
                    moe_weights[:, :, idx].mean().item(),
                    moe_weights[:, :, idx].std().item(),
                    moe_weights[:, :, idx].min().item(),
                    moe_weights[:, :, idx].max().item(),
                )
                print(msg)
        return moe_weights
    
    def forward(self, text, experts):
        ind = {encoder_id: torch.ones(1) for encoder_id in experts.keys()}
        text_embd = {}

        # Unroll repeated captions into present minibatch
        B, captions_per_audio, feat_dim = text.size()
        text = text.view(B * captions_per_audio, feat_dim)
        for modality, layer in zip(self.modalities, self.text_GU):
           # NOTE: Due to the batch norm, the gated units are sensitive to passing
           # in a lot of zeroes, so we do the masking step after the forwards pass
           text_ = layer(text)

           # We always assume that text is available for retrieval
           text_ = text_.view(B, captions_per_audio, -1)

           text_embd[modality] = text_
        text = text.view(B, captions_per_audio, -1)

        # MOE weights computation + normalization 
        # - note that we use the first caption
        # sample to predict the weights
        moe_weights = self.compute_moe_weights(text, ind=ind)

        if self.l2renorm:
            for modality in self.modalities:
                norm = experts[modality].norm(p=2, dim=-1, keepdim=True)
                experts[modality] = experts[modality].div(norm + 0.0000001).clip(-1000, 1000)
        for modality, layer in zip(self.modalities, self.audio_GU):
            experts[modality] = layer(experts[modality])
        if self.training:
            merge_caption_similiarities = "avg"
        else:
            merge_caption_similiarities = self.test_caption_mode
        
        cross_view_conf_matrix = sharded_cross_view_inner_product(
            ind=ind,
            vid_embds=experts,
            text_embds=text_embd,
            l2renorm=self.l2renorm,
            text_weights=moe_weights,
            subspaces=self.modalities,
            merge_caption_similiarities=merge_caption_similiarities,
            # loss_func = "MaxMarginRankingLoss2"
        ).T
 
        return {
            "modalities": self.modalities,
            "cross_view_conf_matrix": cross_view_conf_matrix,
            "text_embds": text_embd,
            "audio_embds": experts
        }

def sharded_cross_view_inner_product(vid_embds, text_embds, text_weights,
        subspaces, l2renorm, ind, merge_caption_similiarities="avg", tol=1E-5,
        loss_func="MaxMarginRankingLoss"):
    """Compute a similarity matrix from sharded vectors.

    Args:
        embds1 (dict[str:th.Tensor]): the set of sub-embeddings that, when
            concatenated, form the whole. The ith shard has shape `B x K x F_i`
            (i.e. they can differ in the last dimension).
        embds2 (dict[str:th.Tensor]): same format.
        weights2 (th.Tensor): weights for the shards in `embds2`.
        l2norm (bool::True): whether to l2 renormalize the full embeddings.

    Returns:
        (th.tensor): similarity matrix of size `BK x BK`.

    NOTE: If multiple captions are provided, we can aggregate their similarities to
    provide a single video-text similarity score.
    """
    B = vid_embds[subspaces[0]].size(0)
    T, num_caps, _ = text_embds[subspaces[0]].size()
    device = vid_embds[subspaces[0]].device
    # unroll separate captions onto first dimension and treat them separately
    sims = torch.zeros(T * num_caps, B, device=device)
    sims_audio = torch.zeros(B, B, device = device)
    sims_text = torch.zeros(T * num_caps, T* num_caps, device = device)
    text_weights = text_weights.view(T * num_caps, -1)
    # if False:
        # mus = [round(x, 3) for x in text_weights.mean(0).detach().cpu().numpy().tolist()]
        # stds = [round(x, 3) for x in text_weights.std(0).detach().cpu().numpy().tolist()]
        # summary = ">>>"
        # for mod, mu, std in zip(subspaces, mus, stds):
            # summary += f"{mod}: {mu} +/- {std} "
        # print(summary)
    
    # mark expert availabilities along the second axis
    available = torch.ones(1, B, len(subspaces), dtype=text_weights.dtype)
    for ii, modality in enumerate(subspaces):
        available[:, :, ii] = ind[modality].squeeze()
    available = available.to(text_weights.device)
    msg = "expected `available` modality mask to only contain 0s or 1s"
    assert set(torch.unique(available).cpu().numpy()).issubset(set([0, 1])), msg
    # set the text weights along the first axis and combine with availabilities to
    # produce a <T x B x num_experts> tensor
    text_weight_tensor = text_weights.view(T * num_caps, 1, len(subspaces)) * available
    # normalise to account for missing experts
    normalising_weights = text_weight_tensor.sum(2).view(T * num_caps, B, 1)
    text_weight_tensor = torch.div(text_weight_tensor, normalising_weights)

    if l2renorm:
#        raise NotImplementedError("Do not use renorm until availability fix is complete")
        l2_mass_vid, l2_mass_text = 0, 0
        for idx, modality in enumerate(subspaces):
            vid_embd_ = vid_embds[modality]
            assert len(vid_embd_.size()) == 2, "expected B x feat_dim format"
            l2_mass_vid += vid_embd_.reshape(B, -1).pow(2).sum(1)
            text_embd_ = text_embds[modality]
            assert len(text_embd_.size()) == 3, "expected B x caps x feat_dim format"
            text_embd_ = text_embd_.reshape(T * num_caps, -1)
            text_embd_ = text_weights[:, idx:idx + 1] * text_embd_
            l2_mass_text += text_embd_.pow(2).sum(1)
        l2_mass_vid = torch.sqrt(l2_mass_vid.clamp(min=1E-6)).unsqueeze(1)
        l2_mass_text = torch.sqrt(l2_mass_text.clamp(min=1E-6)).unsqueeze(1)
    else:
        l2_mass_text, l2_mass_vid = 1, 1

    for idx, modality in enumerate(subspaces):
        vid_embd_ = vid_embds[modality].reshape(B, -1) / l2_mass_vid
        text_embd_ = text_embds[modality].view(T * num_caps, -1)
        msg = "expected weights to be applied to text embeddings"
        assert text_embd_.shape[0] == text_weights.shape[0], msg
        text_embd_ = text_embd_ / l2_mass_text
        weighting = text_weight_tensor[:, :, idx]
        sims += weighting * torch.matmul(text_embd_, vid_embd_.t())  # (T x num_caps) x (B)
        sims_audio += torch.matmul(vid_embd_, vid_embd_.t())
        sims_text += torch.matmul(text_embd_, text_embd_.t())
    
    if l2renorm:
        if not (sims.max() < 1 + tol):
             import ipdb; ipdb.set_trace()
        assert sims.max() < 1 + tol, "expected cosine similarities to be < 1"
        assert sims.min() > -1 - tol, "expected cosine similarities to be > -1"

    if torch.isnan(sims).sum().item():
        import ipdb; ipdb.set_trace()
        raise ValueError("Found nans in similarity matrix!")
    
    if num_caps > 1:
        # aggregate similarities from different captions
        if merge_caption_similiarities == "avg":
            sims = sims.view(B, num_caps, B)
            sims = torch.mean(sims, dim=1)
            sims = sims.view(B, B)
            sims_text = sims_text.view(B, num_caps, B, num_caps)
            sims_text = torch.mean(sims_text,dim=1)
            sims_text = torch.mean(sims_text, dim=-1)
            sims_text = sims_text.view(B, B)
        elif merge_caption_similiarities == "indep":
            pass
        else:
            msg = "unrecognised merge mode: {}"
            raise ValueError(msg.format(merge_caption_similiarities))
    if loss_func == "MaxMarginRankingLoss":
        return sims
    elif loss_func == "MaxMarginRankingLoss2":
        return sims, sims_audio, sims_text


class GatedEmbeddingUnit(nn.Module):
    def __init__(self, input_dimension, output_dimension, use_bn):
        super(GatedEmbeddingUnit, self).__init__()

        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = ContextGating(output_dimension, add_batch_norm=use_bn)

    def forward(self, x):
        x = self.fc(x)
        x = self.cg(x)
        x = F.normalize(x)
        return x


class ContextGating(nn.Module):
    def __init__(self, dimension, add_batch_norm=True):
        super(ContextGating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(dimension)

    def forward(self, x):
        x1 = self.fc(x)
        if self.add_batch_norm:
            x1 = self.batch_norm(x1)
        x1 = torch.sigmoid(x1)
        x = torch.mul(x, x1)
        return x
#        x = torch.cat((x, x1), 1)
#        return F.glu(x, 1)


