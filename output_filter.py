#!/usr/bin/env python3

import matplotlib
matplotlib.use('qtagg')

import re
import matplotlib.pyplot as plt

def plot(x, y, title):
    # plt.plot(x, y)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(x, y, markevery=5, mec="1.0")
    plt.show()

def save_to_csv(x, y, xlabel, ylabel, file_name):
    with open(file_name, "w") as file:
        file.write(f"{xlabel},{ylabel}\n")
        for i in range(len(x)):
            file.write(f"{x[i]},{y[i]}\n")

def filter_output_log(input_file, verbose=False):
    """Reads an input file and plots the training and validation loss with matplotlib. Or saves to CSV. Edit the source code for now to choose."""
    expression = r"Epoch\s*(\d+)\s*\[([^\]]+)\]\s*Avg\sLoss:\s([\d.]+)"
    train_x = []
    train_y = []
    valid_x = []
    valid_y = []

    with open(input_file) as file:
        for line in file:
            match = re.match(expression, line)
            if match:
                epoch = match.group(1)
                data_type = match.group(2)
                loss = match.group(3)

                if verbose:
                    print(f"Epoch {epoch} | Type {data_type} | Loss {loss}")

                if data_type == "Train":
                    train_x.append(epoch)
                    train_y.append(loss)

                if data_type == "Val":
                    valid_x.append(epoch)
                    valid_y.append(loss)

    if verbose:
        print(f"  Train x: {train_x}")
        print(f"  Train y: {train_y}")
        print(f"  Valid x: {valid_x}")
        print(f"  Valid y: {valid_y}")

    # plot(train_x, train_y, title="Train Loss vs. Epoch")
    # plot(valid_x, valid_y, title="Validation Loss vs. Epoch")

    save_to_csv(train_x, train_y, "Epoch", "Loss", "train_64_split_vgg.csv")
    save_to_csv(valid_x, valid_y, "Epoch", "Loss", "valid_64_split_vgg.csv")



if __name__ == "__main__":
    filter_output_log("output_split_vgg_64.log", verbose=True)
