# Divide Dataset Based on Feature Threshold

import numpy as np

def divide_on_feature(X, feature_i, threshold):
    condition = X[:,feature_i] >= threshold
    return [X[condition],X[~condition]]

def main():
    print(divide_on_feature(np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),0,5))   
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-08-26 10:44:29
#
