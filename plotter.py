import matplotlib.pyplot as plt


def plot_fitness_learning_curve(path_to_file):
    """
    Plot learning curve of the players.
    """
    with open(path_to_file, 'r') as f:
        lines = f.readlines()
        lines = [line.split(',') for line in lines]
        lines = [[float(x) for x in line] for line in lines]
        best_fit = [line[0] for line in lines]
        worst_fit = [line[1] for line in lines]
        average_fit = [line[2] for line in lines]
        plt.plot(best_fit, label='best')
        plt.plot(worst_fit, label='worst')
        plt.plot(average_fit, label='average')
        plt.legend()
        plt.show()
        pass
    pass


plot_fitness_learning_curve('learning_info.txt')
