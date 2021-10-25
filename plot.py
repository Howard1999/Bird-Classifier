from matplotlib import pyplot as plt


def show_hist(hist_list, title, legends):
    plt.title(title)
    for i in range(len(hist_list)):
        plt.plot(hist_list[i])
    plt.xlabel('Epoch')
    plt.legend(legends, loc='center right')
    plt.show()
    
def save_hist(hist_list, title, legends):
    plt.title(title)
    for i in range(len(hist_list)):
        plt.plot(hist_list[i])
    plt.xlabel('Epoch')
    plt.legend(legends, loc='center right')
    plt.savefig(title)
    plt.close()
