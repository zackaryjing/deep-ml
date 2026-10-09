# problem: Decision Tree Learning

import torch
import math
from collections import Counter
from typing import List, Dict, Any, Union
from collections import defaultdict


def calculate_entropy(labels: List[Any]) -> float:
    """
    Compute the Shannon entropy of the list of labels.
    labels: list of any hashable items.
    Returns a Python float.
    """
    # Your implementation here
    n = len(labels)
    cnts = Counter(labels)
    res = 0
    for key, val in cnts.items():
        p = val / n
        res += p * math.log2(p)
    return -res


def calculate_information_gain(
        examples: List[Dict[str, Any]],
        attr: str,
        target_attr: str
) -> float:
    """
    Compute information gain for splitting `examples` on `attr` w.r.t. `target_attr`.
    Returns a Python float.
    """
    # Your implementation here
    temp = []
    for example in examples:
        temp.append(example[target_attr])
    HS = calculate_entropy(temp)
    n = len(examples)
    subgroup = defaultdict(list)
    for item in examples:
        subgroup[item[attr]].append(item[target_attr])
    HS_after = 0.0
    for key, val in subgroup.items():
        HS_after += len(val) / n * calculate_entropy(val)
    return HS - HS_after


def majority_class(
        examples: List[Dict[str, Any]],
        target_attr: str
) -> Any:
    """
    Return the most common value of `target_attr` in `examples`.
    In case of a tie, return the class that comes first alphabetically.
    """
    # Your implementation here
    occured = defaultdict(int)
    for example in examples:
        occured[example[target_attr]] += 1
    data = [(-val, key) for key, val in occured.items()]
    data.sort()
    return data[0][1]


def learn_decision_tree(
        examples: List[Dict[str, Any]],
        attributes: List[str],
        target_attr: str
) -> Union[Dict[str, Any], Any]:
    """
    Learn a decision tree using the ID3 algorithm.
    Returns either a nested dict representing the tree or a class label at the leaves.
    """
    # Your implementation here
    labels = [e[target_attr] for e in examples]
    if len(set(labels)) == 1:
        return labels[0]

    if len(attributes) == 0:
        return majority_class(examples, target_attr)

    best_gain = -1
    best_attr = attributes[0]
    for attr in attributes:
        gain = calculate_information_gain(examples, attr, target_attr)
        if gain > best_gain:
            best_gain  = gain
            best_attr = attr
    subgroups = defaultdict(list)
    for example in examples:
        subgroups[example[best_attr]].append(example)
    res = {best_attr:{}}
    remaining_attrs = [a for a in attributes if a != best_attr]
    for value,subset in subgroups.items():
        res[best_attr][value] = learn_decision_tree(subset,remaining_attrs,target_attr)
    return res


def main():
    examples = [
        {'Outlook': 'Sunny', 'Wind': 'Weak', 'PlayTennis': 'No'},
        {'Outlook': 'Overcast', 'Wind': 'Strong', 'PlayTennis': 'Yes'},
        {'Outlook': 'Rain', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
        {'Outlook': 'Sunny', 'Wind': 'Strong', 'PlayTennis': 'No'},
        {'Outlook': 'Overcast', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
        {'Outlook': 'Rain', 'Wind': 'Strong', 'PlayTennis': 'No'},
        {'Outlook': 'Rain', 'Wind': 'Weak', 'PlayTennis': 'Yes'}
    ]
    attributes = ['Outlook', 'Wind']
    target_attr = 'PlayTennis'
    print(learn_decision_tree(examples,attributes,target_attr))


if __name__ == "__main__":
    main()

#
# Created By jing At 2026-09-11 12:48:34
#
