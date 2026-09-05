# problem: Implement TF-IDF (Term Frequency-Inverse Document Frequency)


import torch
import math
from typing import List


def compute_tf_idf(corpus: List[List[str]], query: List[str]) -> torch.Tensor:
    """
    Compute TF-IDF scores for a query against a corpus of documents using PyTorch.

    :param corpus: List of documents, where each document is a list of words
    :param query: List of words in the query
    :return: torch.Tensor of shape (num_docs, num_query_words) with TF-IDF scores
             rounded to five decimal places
    """
    num_docs = len(corpus)
    num_query_words = len(query)
    res = torch.zeros(num_docs, num_query_words, dtype=torch.float)
    for i, q in enumerate(query):
        occur_cnts = 0
        for j, c in enumerate(corpus):
            cnts = c.count(q)
            if cnts != 0:
                occur_cnts += 1
                res[j, i] = c.count(q) / len(c)
        res[:, i] *= math.log((num_docs + 1) / (occur_cnts + 1)) + 1
    return res


def main():
    corpus = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "chased", "the", "cat"],
        ["the", "bird", "flew", "over", "the", "mat"]
        ]
    query = ["cat"]
    print(compute_tf_idf(corpus, query))


if __name__ == "__main__":
    main()

#
# Created By jing At 2026-09-05 14:58:01
#
