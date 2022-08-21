import matplotlib.pyplot as plt
from IPython import display
from contextlib import contextmanager
import sys, os

plt.ion()


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def plot(scores_labels, x_label, y_label, y_min, filename):
    if "." not in filename:
        filename += ".png"
    
    if not os.path.exists("./Graphs"):
        os.mkdir("./Graphs")
    
    with suppress_stdout():
        display.clear_output(wait=True)
        display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # lista tupleova o ubliku (lista_mjerenja, label)
    for score_label in scores_labels:
        scores = score_label[0]
        label = score_label[1]

        plt.plot(scores, label = label)
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))

    plt.ylim(ymin=y_min)
    plt.legend(loc="lower left")
    plt.show(block=False)
    plt.pause(.1)
    plt.savefig("./Graphs/" + filename)