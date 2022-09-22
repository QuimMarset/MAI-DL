import seaborn as sns
import matplotlib.pyplot as plt


def decayed_learning_rate(step, initial_learning_rate, decay_rate, decay_steps):
    return initial_learning_rate * decay_rate ** (step / decay_steps)



if __name__ == '__main__':

    steps_per_epoch = 158
    epochs = 70

    total_steps = steps_per_epoch * epochs

    decay_rate = 0.95

    learning_rates = []

    initial_lr = 1e-3

    for step in range(total_steps):

        new_lr = decayed_learning_rate(step, initial_lr, decay_rate, 1.5*steps_per_epoch)
        learning_rates.append(new_lr)


    print(f'Initial LR: {initial_lr:.2E}')
    print(f'Last LR: {new_lr:.2E}')
    print(f'LR at epoch 10: {learning_rates[1579]:.2E}')
    print(f'LR at epoch 20: {learning_rates[3159]:.2E}')
    print(f'LR at epoch 30: {learning_rates[4739]:.2E}')
    print(f'LR at epoch 40: {learning_rates[6319]:.2E}')
    print(f'LR at epoch 50: {learning_rates[7899]:.2E}')
    print(f'LR at epoch 60: {learning_rates[9479]:.2E}')
    print(f'LR at epoch 70: {learning_rates[11059]:.2E}')
    #print(f'LR at epoch 80: {learning_rates[12639]:.2E}')
    

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates)
    plt.title('Learning rate decaying at each epoch')
    plt.show()

