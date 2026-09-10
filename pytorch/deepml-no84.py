# Phi Transformation for Polynomial Features

import torch

def phi_transform(data: list[float], degree: int) -> torch.Tensor:
    """
    Perform a Phi Transformation to map input features into a higher-dimensional space by generating polynomial features.

    Args:
        data (list[float]): A list of numerical values to transform.
        degree (int): The degree of the polynomial expansion.

    Returns:
        torch.Tensor: A 2D tensor where each row represents the transformed features of a data point,
                      containing powers from 0 to degree.
    """
    # Your code here
    data_t = torch.tensor(data).reshape(-1,1)
    height = data_t.shape[0]
    if (degree > 0):
        res= torch.ones((height,1))
    else:
        return torch.tensor([])
    cur = data_t
    for i in range(degree):
        res = torch.cat([res,cur],dim=1)
        cur = cur * data_t
    return res

def main():
    print( phi_transform([1.0, 2.0], 2))
    print( phi_transform([1.0, 3.0], 3))
    print( phi_transform([1.0, 2.0], -1))
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-09-10 19:47:03
#
