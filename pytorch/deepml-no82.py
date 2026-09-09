# Grayscale Image Contrast Calculator

import torch

def calculate_contrast(img: torch.Tensor) -> float:
    """
    Calculate the contrast of a grayscale image.
    Args:
        img (torch.Tensor): 2D tensor representing a grayscale image with pixel values between 0 and 255.
    Returns:
        float: Contrast value rounded to 3 decimal places.
    """
    # Your code here
    return (img.max() - img.min()).item()

def main():
    img = torch.tensor([[0, 50], [200, 255]])
    print(img)
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-09-09 19:34:17
#
