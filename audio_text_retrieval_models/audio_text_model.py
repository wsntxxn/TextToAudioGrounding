import numpy as np
import torch
import torch.nn as nn
from audio_text_retrieval_models.base import BaseModel


class GradientReversalFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha * grad_output
        return grad_input, None


class GradientClip(nn.Module):

    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = torch.tensor(-alpha, requires_grad=False)

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class AudioTextClip(BaseModel):
    def __init__(self,
                 audio_encoder,
                 text_encoder,
                 audio_dim,
                 text_dim,
                 shared_dim,
                 audio_forward_keys=["waveform", "wave_length"],
                 text_forward_keys=["input_ids", "token_type_ids",
                                    "attention_mask"],
                 gradient_clip=1):
        super().__init__()

        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.audio_proj = nn.Linear(audio_dim, shared_dim)
        self.text_proj = nn.Linear(text_dim, shared_dim)
        self.audio_forward_keys = audio_forward_keys
        self.text_forward_keys = text_forward_keys
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.gradient_clip = gradient_clip
        if gradient_clip != 1:
            self.audio_gradient_clip = GradientClip(gradient_clip)
            self.text_gradient_clip = GradientClip(gradient_clip)
 
    def forward(self, input_dict):

        batch_size = input_dict["waveform"].size(0)
        num_captions = input_dict["num_captions"]

        audio_input = {k: input_dict[k] for k in self.audio_forward_keys}
        audio_emb = self.audio_encoder(**audio_input)["clip_emb"]
        if self.gradient_clip != 1:
            audio_emb = self.audio_gradient_clip(audio_emb)
        audio_emb = self.audio_proj(audio_emb)
        norm = audio_emb.norm(p=2, dim=-1, keepdim=True)
        audio_emb = audio_emb.div(norm + 1e-7).clip(-1e3, 1e3)
 
        text_input = {}
        for k in self.text_forward_keys:
            v = input_dict[k]
            text_input[k] = v.reshape(batch_size * num_captions, *v.size()[2:])
        text_emb = self.text_encoder(**text_input)["clip_emb"]
        if self.gradient_clip != 1:
            text_emb = self.text_gradient_clip(text_emb)
        text_emb = self.text_proj(text_emb)
        norm = text_emb.norm(p=2, dim=-1, keepdim=True)
        text_emb = text_emb.div(norm + 1e-7).clip(-1e3, 1e3)
        text_emb = text_emb.view(batch_size, num_captions, -1)
                
        output = {
            "audio_emb": audio_emb,
            "text_emb": text_emb,
            "logit_scale": self.logit_scale.exp()
        }

        return output

    def evaluate_retrieval(self, inputs):
        return self.forward(inputs)
    
    def encode_audio(self, waveform, wave_length):
        audio_emb = self.audio_encoder(waveform, wave_length)["clip_emb"]
        audio_emb = self.audio_proj(audio_emb)
        norm = audio_emb.norm(p=2, dim=-1, keepdim=True)
        audio_emb = audio_emb.div(norm + 1e-7).clip(-1e3, 1e3)
        return audio_emb

    def encode_text(self, **text):
        text_emb = self.text_encoder(**text)["clip_emb"]
        text_emb = self.text_proj(text_emb)
        norm = text_emb.norm(p=2, dim=-1, keepdim=True)
        text_emb = text_emb.div(norm + 1e-7).clip(-1e3, 1e3)
        return text_emb


class AudioSingleTextClip(AudioTextClip):

    def forward(self, input_dict):
        audio_input = {k: input_dict[k] for k in self.audio_forward_keys}
        audio_emb = self.audio_encoder(**audio_input)["clip_emb"]
        if self.gradient_clip != 1:
            audio_emb = self.audio_gradient_clip(audio_emb)
        audio_emb = self.audio_proj(audio_emb)
        norm = audio_emb.norm(p=2, dim=-1, keepdim=True)
        audio_emb = audio_emb.div(norm + 1e-7).clip(-1e3, 1e3)
 
        text_input = {k: input_dict[k] for k in self.text_forward_keys}
        text_emb = self.text_encoder(**text_input)["clip_emb"]
        if self.gradient_clip != 1:
            text_emb = self.text_gradient_clip(text_emb)
        text_emb = self.text_proj(text_emb)
        norm = text_emb.norm(p=2, dim=-1, keepdim=True)
        text_emb = text_emb.div(norm + 1e-7).clip(-1e3, 1e3)
                
        output = {
            "audio_emb": audio_emb,
            "text_emb": text_emb,
            "logit_scale": self.logit_scale.exp()
        }

        return output

    def evaluate_retrieval(self, input_dict):
        if "num_captions" in input_dict:
            return super().forward(input_dict)
        else:
            return self.forward(input_dict)


# class AudioTextDataParallel(nn.DataParallel):

    # def __init__(self, embed_module, match_module, device_ids=None, output_device=None, dim=0):
        # super(nn.DataParallel, self).__init__()
        # device_type = _get_available_device_type()
        # if device_type is None:
            # self.embed_module = embed_module
            # self.match_module = match_module
            # self.device_ids = []
            # return

        # if device_ids is None:
            # device_ids = _get_all_device_indices()

        # if output_device is None:
            # output_device = device_ids[0]

        # self.dim = dim
        # self.embed_module = embed_module
        # self.match_module = match_module
        # self.device_ids = [_get_device_index(x, True) for x in device_ids]
        # self.output_device = _get_device_index(output_device, True)
        # self.src_device_obj = torch.device(device_type, self.device_ids[0])

        # _check_balance(self.device_ids)

        # if len(self.device_ids) == 1:
            # self.embed_module.to(self.src_device_obj)
            # self.match_module.to(self.src_device_obj)

    # def forward(self, *inputs, **kwargs):
        # with torch.autograd.profiler.record_function("DataParallel.forward"):
            # if not self.device_ids:
                # return self.match_module(**self.embed_module(*inputs, **kwargs))

            # for module in [self.embed_module, self.match_module]:
                # for t in chain(module.parameters(), module.buffers()):
                    # if t.device != self.src_device_obj:
                        # raise RuntimeError("module must have its parameters and buffers "
                                           # "on device {} (device_ids[0]) but found one of "
                                           # "them on device: {}".format(self.src_device_obj, t.device))
            # inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            # if not inputs and not kwargs:
                # inputs = ((),)
                # kwargs = ({},)
            # if len(self.device_ids) == 1:
                # return self.match_module(**self.embed_module(*inputs[0], **kwargs[0]))
            # replicas = self.replicate(self.embed_module, self.device_ids[:len(inputs)])
            # outputs = self.parallel_apply(replicas, inputs, kwargs)
            # text_emb = self.gather([out["text_emb"] for out in outputs], self.output_device)
            # text_emb = nn.parallel.comm.broadcast(text_emb, self.device_ids)
            # replicas = self.replicate(self.match_module, self.device_ids[:len(inputs)])
            # kwargs = tuple([{} for _ in range(len(self.device_ids))])
            # audio_embs = [out["audio_embs"] for out in outputs]
            # inputs = tuple([(text_emb[i], audio_embs[i]) for i in range(len(self.device_ids))])
            # outputs = self.parallel_apply(replicas, inputs, kwargs)
            # outputs = self.gather(outputs, self.output_device)
            # outputs["cross_view_conf_matrix"] = outputs["cross_view_conf_matrix"].T
            # return outputs


