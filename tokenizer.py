# tokenizer.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to ShanghaiTech University, including a link
# to https://i-techx.github.io/iTechX/courses?course_code=CS274A
#
# Attribution Information: The NLP projects were developed at ShanghaiTech University.
# The core projects and autograders were adapted by Haoyi Wu (wuhy1@shanghaitech.edu.cn)


from typing import Dict, Tuple, List
import util

from tokenizers import Tokenizer
import tokenizers.models
import tokenizers.pre_tokenizers
import tokenizers.decoders


def get_gpt2_tokenizer() -> Tokenizer:
    """
    Return a GPT-2 tokenizer.
    """
    #  vocab, merges = tokenizers.models.BPE.read_file(
    #      "data/vocab.json", "data/merges.txt")
    vocab, merges = tokenizers.models.BPE.read_file(
        "data/vocab.json", "data/merges.txt")
    clean_vocab(vocab, merges)
    tokenizer = Tokenizer(tokenizers.models.BPE(vocab, merges))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(
        add_prefix_space=False)
    tokenizer.decoder = tokenizers.decoders.ByteLevel()

#  vocab_items = sorted(vocab.items(), key=lambda x: x[1])
    return tokenizer


def clean_vocab(vocab: Dict[str, int], merges: List[Tuple[str, str]]):
    """
    Question:
        Given the vocabulary and merges of a BPE tokenizer, clean them up to avoid subtokens
        that consist of multiple digits. This would reduce the sparsity problem.

        This function does in-place modifications, so it should not return anything.

    Example:
        >>> vocab = {'Ġ': 0, '1': 1, '2': 2, 'Ġ1': 3, 'Ġ2': 4, '12': 5, 'Ġ12': 6}
        >>> merges = [('Ġ', '1'), ('Ġ', '2'), ('1', '2'), ('Ġ1', '2')]
        >>> clean_vocab(vocab, merges)
        >>> vocab
        {'Ġ': 0, '1': 1, '2': 2, 'Ġ1': 3, 'Ġ2': 4}

    Args:
        vocab (:obj:`Dict[str, int]`):
            A dictionnary of string keys and their ids, e.g.`{"am": 0,...}`

        merges (:obj:`List[Tuple[str, str]]`):
            A list of pairs of tokens (:obj:`Tuple[str, str]`), e.g. `[("a", "b"),...]`
    """

    """YOUR CODE HERE"""
    #  (vocab: Dict[str, int], merges: List[Tuple[str, str]])
    def has_digits(str: str):
        # digit cannot combined with character: only consider digits that appear multiple times.
        return len([char for char in str if char.isdigit()]) > 1

    keys_to_del = []
    for i in vocab:
        if has_digits(i):
            keys_to_del.append(i)

    for i in keys_to_del:
        del vocab[i]

    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    for i, dit in enumerate(sorted_vocab):
        vocab[dit[0]] = i

    valid_merges = []
    for merge in merges:
        if has_digits(merge[0]) or has_digits(merge[1]):
            continue
        if merge[0][-1].isdigit() and merge[1][0].isdigit():
            continue
        valid_merges.append(merge)
    merges[:] = valid_merges

    return


if __name__ == '__main__':

    print("Running tokenizer.py ...")

    tokenizer = get_gpt2_tokenizer()

    sentence = "Is 1029310928407 a multiple of 3?"
    print("      Sentence:", sentence)
    output = tokenizer.encode(sentence)
    print("After encoding:", output.tokens)
