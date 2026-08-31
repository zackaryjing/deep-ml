# Leaky ReLU Activation Function

def leaky_relu(z: float, alpha: float = 0.01) -> float|int:
    # Your code here
    return z if z >= 0 else z * alpha


def main():
    print(leaky_relu(0)) 
    print(leaky_relu(1))
    print(leaky_relu(-1)) 
    print(leaky_relu(-2, alpha=0.1))

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-08-26 11:08:43
#
