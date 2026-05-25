import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        positive_sentenses = [sentense.split(' ') for sentense in positive]
        negative_sentenses = [sentense.split(' ') for sentense in negative]

        vocabulary = []
        max_len = 0
        for sentense in positive_sentenses + negative_sentenses:
            vocabulary += sentense
            max_len = max(max_len, len(sentense))
        vocabulary = set(vocabulary)
        vocabulary = list(vocabulary)
        vocabulary.sort()
        word2id = {word: i+1 for i, word in enumerate(vocabulary)}

        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        
        encoded = []
        for sentense in positive_sentenses + negative_sentenses:
            token_ids = [word2id[word] for word in sentense] + [0] * (max_len - len(sentense))
            encoded.append(token_ids)

        return torch.tensor(encoded).float()
        
