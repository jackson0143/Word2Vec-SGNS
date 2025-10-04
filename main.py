import re
import random
from collections import Counter
from itertools import islice
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from convokit import Corpus, download

#hyperparams
SEED = 42
WINDOW = 2
DYNAMIC_WINDOW = True
BATCH_SIZE =256
EMBEDDING_DIM = 50
LR = 1e-3
EPOCHS = 5
LOG_EVERY = 100
NEIGHBOUR_TOP_K = 5
MAX_UTTERANCES = 10000

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)



def tokenise(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+'[a-z0-9]+|[a-z0-9]+", text.lower(), flags=re.I)

#Note to self: need to drop low frequency words, need a threshold for that
def build_vocab(tokenised_utts: List[List[str]]) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    # Flatten tokenised utterances into a single list of words
    words: List[str] = [token for utt in tokenised_utts for token in utt]
    counts = Counter(words)

    sorted_vocab: List[Tuple[str, int]] = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    idx_to_word: Dict[int, str] = {idx: word for idx, (word, _) in enumerate(sorted_vocab)}
    word_to_idx: Dict[str, int] = {word: idx for idx, word in idx_to_word.items()}
    return words, word_to_idx, idx_to_word


def tokens_to_ids(tokenised_utts: List[List[str]], word_to_idx: Dict[str, int]) -> List[List[int]]:
    return [[word_to_idx[w] for w in utt if w in word_to_idx] for utt in tokenised_utts]



def make_skip_gram_pairs(tokens: Sequence[int], window: int, dynamic_window: bool = False) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for i, center in enumerate(tokens):
        w = random.randint(1, window) if dynamic_window else window
        left_pointer = max(0, i - w)
        right_pointer = min(len(tokens) - 1, i + w)
        for j in range(left_pointer, right_pointer + 1):
            if j == i:
                continue
            pairs.append((center, tokens[j]))
    return pairs


def make_pairs_for_all_utterances(utt_ids: List[List[int]], window: int, dynamic_window: bool = False) -> List[Tuple[int, int]]:
    all_pairs: List[Tuple[int, int]] = []
    for utt in utt_ids:
        if not utt:
            continue
        all_pairs.extend(make_skip_gram_pairs(utt, window, dynamic_window))
    return all_pairs



# Dataset
class Word2VecDataset(Dataset):
    def __init__(self, data: List[Tuple[int, int]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[int, int]:
        return self.data[idx]


# Model
'''
FROM MY UNDERSTANDING:
The model will try to maximise the probability of the context words given the center word
eg: P(context_word | center_word)
This is what we train the model to do

Input embeddings: [vocab, embedding_dim]
Each row is a vector representation of a word when it is a center word (input)

Output embeddings: [vocab, embedding_dim]
Each row is a vector representation of a word when it is a context word (output)


During training, we look at a center word and all the context words around it, and try to maximise the probability of the context words given the center word

So each time the Input embedding will learn to represent the center word as a vector, 
and that way we aren't actually making predictions, we are looking to see how similar the center word is to the other words
We are mapping out the embedding space of the words, and each time we train we are pulling them closer to each other

Im guessing for CBOW we still use the input embeddings to predict the context words,
but we average the input embeddings to get a better representation of the center word
'''
class SkipGram(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim) #embedding matrix for center words
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim) #embedding matrix for context words

    def forward(self, center_indices: torch.Tensor) -> torch.Tensor:
        center_vecs = self.input_embeddings(center_indices) 
        scores = center_vecs @ self.output_embeddings.weight.t() 
        return scores

@torch.no_grad()
def nearest_neighbours(
    model: SkipGram,
    word_to_idx: Dict[str, int],
    idx_to_word: Dict[int, str],
    device: torch.device,
    query_word: str,
    top_k: int = NEIGHBOUR_TOP_K,
):
    model.eval()
    if query_word not in word_to_idx:
        print(f"'{query_word}' not in vocab")
        return []

    '''
    we use input embeddings matrix because in the training, we take the center word get a score for all the context words
    and so in the end, we have a trained input embeddings matrix that is used to find the nearest neighbours
    Input embeddings is what the model uses to represent the words as inputs

    perhaps its oaky to average both input and output embeddings to get a better representation, idk
    '''
    embed = model.input_embeddings.weight 
    q_idx = word_to_idx[query_word]
    q_vec = embed[q_idx].unsqueeze(0)     

    sims = F.cosine_similarity(q_vec, embed, dim=1)  #compare how similar  our query word is to all the other rows
    sims[q_idx] = float("-inf")  # don’t return self
    top_vals, top_inds = torch.topk(sims, k=min(top_k, embed.size(0) - 1)) #get top k values and indices


    return [(idx_to_word[i.item()], top_vals[j].item()) for j, i in enumerate(top_inds)]



def main():
    corpus = Corpus(filename=download("reddit-corpus-small"))
    chosen_utts = [utt.text for utt in islice(corpus.iter_utterances(), MAX_UTTERANCES)]

    # Tokenise and build vocab (word to index and index to word)
    tokenised_utts = [tokenise(utt) for utt in chosen_utts]
    words, word_to_idx, idx_to_word = build_vocab(tokenised_utts)
    utt_ids = tokens_to_ids(tokenised_utts, word_to_idx)

    # Training pairs and loader
    training_data = make_pairs_for_all_utterances(utt_ids, WINDOW, DYNAMIC_WINDOW)
    dataset = Word2VecDataset(training_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    # Model & Optim
    vocab_size = len(word_to_idx)
    model = SkipGram(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Vocab size: {vocab_size}, Using device: {device}")

    # Train (full softmax, note: maybe implement negative sampling later)
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        step = 0
        for centers_tensor, contexts_tensor in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            centers_tensor = centers_tensor.to(device).long()
            contexts_tensor = contexts_tensor.to(device).long()

            logits = model(centers_tensor)            
            loss = criterion(logits, contexts_tensor) 

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_loss += loss.item()
            step += 1
            if step % LOG_EVERY == 0:
                print(f"step {step} | loss {total_loss / step:.4f}")

        print(f"Epoch {epoch+1} done. Avg loss: {total_loss / max(1, step):.4f}")

    example = "day"
    print(nearest_neighbours(model, word_to_idx, idx_to_word, device, example, top_k=NEIGHBOUR_TOP_K))


if __name__ == "__main__":
    main()
