import seaborn as sns
import matplotlib.pyplot as plt


def decayed_learning_rate(step, initial_learning_rate, decay_rate, decay_steps):
    return initial_learning_rate * decay_rate ** (step / decay_steps)



if __name__ == '__main__':

    steps_per_epoch = 158
    epochs = 50

    total_steps = steps_per_epoch * epochs

    decay_rate = 0.98
    decay_epochs = 1

    learning_rates = []

    initial_lr = 5e-4

    for step in range(total_steps):
        new_lr = decayed_learning_rate(step, initial_lr, decay_rate, decay_epochs*steps_per_epoch)
        learning_rates.append(new_lr)

    print(decay_rate, decay_epochs)
    print(f'Initial LR: {initial_lr:.2E}')
    print(f'Last LR: {new_lr:.2E}')

    for i in range(0, epochs, 10):
        if i > 0:
            index = steps_per_epoch * i - 1
            print(f'LR at epoch {i}: {learning_rates[index]:.2e}')
    

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates)
    plt.title('Learning rate decaying at each epoch')
    plt.show()

