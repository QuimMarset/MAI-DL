import seaborn as sns
import matplotlib.pyplot as plt


def decayed_learning_rate(step, initial_learning_rate, decay_rate, decay_steps):
    return initial_learning_rate * decay_rate ** (step / decay_steps)



if __name__ == '__main__':

    total_steps = 1000

    decay_rate = 0.95
    decay_steps = 100

    learning_rates = []

    initial_lr = 0.001

    for step in range(total_steps):
        new_lr = decayed_learning_rate(step, initial_lr, decay_rate, decay_steps)
        learning_rates.append(new_lr)

    print(decay_rate, decay_steps)
    print(f'Initial LR: {initial_lr:.2E}')
    print(f'Last LR: {new_lr:.2E}')    

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates)
    plt.title('Learning rate decaying at each epoch')
    plt.show()

