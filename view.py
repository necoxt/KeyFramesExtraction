import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys


def on_close(event):
    sys.exit()


def view(path):
    with open(path, 'r') as f:
        data = f.readlines()
    plt.ion()
    fig = plt.figure()
    fig.canvas.mpl_connect('close_event', on_close)
    print("Press enter to continue...")
    for line in data:
        line = line.split(" ")[0]
        image = mpimg.imread(line)
        plt.imshow(image)
        plt.title(line)
        input()
        plt.clf()  # clear the current figure


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', required=True, help='txt file path')
    args = parser.parse_args()
    view(args.path)
