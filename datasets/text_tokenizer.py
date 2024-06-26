import pickle
import numpy as np

from utils.build_vocab import Vocabulary
from utils.train_util import pad_sequence


class DictTokenizer:

    def __init__(self,
                 vocabulary) -> None:
        self.vocabulary = Vocabulary()
        state_dict = pickle.load(open(vocabulary, "rb"))
        self.vocabulary.load_state_dict(state_dict)

    def __call__(self, texts):
        assert isinstance(texts, list), "the input must be List[str] or List[List[str]]"
        if isinstance(texts[0], str): # List[str]
            tokens = []
            for text in texts:
                tokens.append(np.array([self.vocabulary(token) for token in text.split()]))
            tokens, token_lens = pad_sequence(tokens)
        elif isinstance(texts[0], list):
            text_num = len(texts[0])
            batch_size = len(texts)
            for text_list in texts:
                assert len(text_list) == text_num, "the text number in each list must be the same"
            flattened_texts = sum(texts, [])
            tokens = []
            for text in flattened_texts:
                tokens.append(np.array([self.vocabulary(token) for token in text.split()]))
            tokens, token_lens = pad_sequence(tokens)
            tokens = tokens.reshape(batch_size, text_num, -1)
            token_lens = np.array(token_lens)
            token_lens = token_lens.reshape(batch_size, text_num)
        return {
            "text": tokens,
            "text_len": token_lens
        }

    def inverse_transform(self, texts):
        output = []
        for text in texts:
            tokens = []
            for word_idx in text:
                if word_idx != 0:
                    tokens.append(self.vocabulary.idx2word[word_idx])
                else:
                    break
            output.append(" ".join(tokens))
        return output
            

class HuggingFaceTokenizer:
    
    def __init__(self, model_type, max_length=30) -> None:
        from transformers import AutoTokenizer
        model_type_to_cache = {
            "bert-base-uncased": "/hpc_stor03/sjtu_home/xuenan.xu/tools/hf_cache/tokenizer/bert-base-uncased"
        }
        try:
            self.core = AutoTokenizer.from_pretrained(model_type, proxies={
                "http": "http://d6-hpc-debuggpu-001:14227",
                "https": "http://d6-hpc-debuggpu-001:14227",
            })
        except:
            self.core = AutoTokenizer.from_pretrained(model_type_to_cache[model_type])
        self.max_length = max_length

    def __call__(self, texts):
        assert isinstance(texts, list), "the input must be List[str] or List[List[str]]"
        if isinstance(texts[0], str): # List[str]
            token_dict = self.core(texts, padding="max_length", return_tensors="pt",
                                   truncation=True, max_length=self.max_length)
        elif isinstance(texts[0], list):
            text_num = len(texts[0])
            batch_size = len(texts)
            for text_list in texts:
                assert len(text_list) == text_num, "the text number in each list must be the same"
            flattened_texts = sum(texts, [])
            token_dict = self.core(flattened_texts, padding="max_length", return_tensors="pt",
                                   truncation=True, max_length=self.max_length)
            new_dict = {}
            for k, v in token_dict.items():
                v = v.reshape(batch_size, text_num, *v.shape[1:])
                new_dict[k] = v
            token_dict = new_dict
        return token_dict
