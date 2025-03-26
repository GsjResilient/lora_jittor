import os

from matplotlib import pyplot as plt


def paint(train_loss_list,path):
    os.makedirs("loss", exist_ok=True)
    filename = os.path.join("loss", f"{path}.png")
    plt.plot(train_loss_list, label="Training Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    # plt.show()

