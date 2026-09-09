# problem: Calculate Image Brightness

import torch


def calculate_brightness(img) -> float:
    """
    Calculate the average brightness of a grayscale image using PyTorch.

    Args:
        img: A 2D list where each element represents a pixel value between 0-255.

    Returns:
        The average brightness rounded to two decimal places, or -1 for invalid inputs.
    """
    # Write your code here
    if len(img) == 0:
        return -1
    length = len(img[0])
    for l in img:
        if len(l) != length:
            return -1
    img_t = torch.tensor(img, dtype=torch.float)
    mn = img_t.min()
    mx = img_t.max()
    if mn < 0 or mx > 255:
        return -1
    return torch.mean(img_t).item()


def main():
    img = [[100, 200], [50, 150]]
    print(calculate_brightness(img))


if __name__ == "__main__":
    main()

#
# Created By jing At 2026-09-07 20:25:56
#
