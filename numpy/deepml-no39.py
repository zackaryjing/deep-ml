# Implementation of Log Softmax Function

import numpy as np

def log_softmax(scores: list) -> np.ndarray:
    s_a = np.array(scores)
    scores_norm = s_a - max(s_a)
    lg_part = np.log(sum(np.exp(scores_norm)))
    return scores_norm - lg_part

def main():
    print(log_softmax([1,2,3]))
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-08-26 09:45:14
#

