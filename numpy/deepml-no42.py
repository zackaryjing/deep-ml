# Implement ReLU Activation Function

def relu(z: float) -> float:
    return z if z >= 0 else 0 


def main():
    print(relu(-1))
    print(relu(1))

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-08-26 12:19:33
#
