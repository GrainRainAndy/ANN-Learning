import os
import pickle

import matplotlib.pyplot as plt

from dataset.mnist.mnist import load_mnist
from models.FCNN import *
from utils.earlystopping import *
from utils.optimizer import *

if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    continueTraining = True
    modelsSavePath = ".//saved_params"
    modelsAutoSavePath = modelsSavePath + "//autosave"
    diagramsSavePath = ".//diagrams//" + "diagram.png"

    if continueTraining:
        (network, base_loss) = pickle.load(open(modelsAutoSavePath + "//best_model.pkl", 'rb'))
    else:
        # network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10, reg_lambda = 1e-3)
        network = TwoHiddenLayersNet(input_size=784, hidden_size1=100, hidden_size2=50, output_size=10, reg_lambda=1e-3)
        base_loss = None

    iters_num = 50000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1
    optimizer = SGD(learning_rate)
    # optimizer = AdaGrad(learning_rate)
    train_loss_list = []
    train_loss_show_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)
    epochs_num = int(iters_num // iter_per_epoch) + 1
    patience = iters_num * 0.0002
    min_delta = network.reg_lambda * 0.1
    early_stopping = AvgEarlyStopping(patience, min_delta)

    for i in range(iters_num):
        epoch = int(i // iter_per_epoch) + 1
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        # Dynamic learning rate
        if i % 1000 == 0:
            learning_rate *= 0.9
            optimizer.update_lr(learning_rate)

        grads = network.gradient(x_batch, t_batch)
        params = network.params
        optimizer.update(params, grads)

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            # loss = network.loss(x_test, t_test)
            train_loss_show_list.append(loss)
            print(f"=== Epoch: [{epoch}/{epochs_num}], Train acc: {train_acc:.04f}, Test acc: {test_acc:.04f}, Loss: {loss:.04f} ===")

            with open(f"{modelsAutoSavePath}//autosave_{epoch}.pkl", "wb") as f:
                pickle.dump((network, loss), f)

            saved_models = sorted(os.listdir(modelsAutoSavePath),
                                  key=lambda x: os.path.getctime(os.path.join(modelsAutoSavePath, x)))
            autosave_models = [model for model in saved_models if model.startswith("autosave_")]
            if len(autosave_models) > 3:
                for model in autosave_models[:-3]:
                    os.remove(os.path.join(modelsAutoSavePath, model))

            if base_loss is None:
                base_loss = loss
                with open(modelsAutoSavePath + "//best_model.pkl", "wb") as f:
                    pickle.dump((network, loss), f)
            elif loss < base_loss:
                with open(modelsAutoSavePath + "//best_model.pkl", "wb") as f:
                    pickle.dump((network, loss), f)

            loss = network.loss(x_test, t_test)
            early_stopping(loss)
            if early_stopping.early_stop:
                print("=== Early stopped ===")
                break
    print("=== Training finished ===")

    epochs = np.arange(len(train_acc_list))

    fig, ax1 = plt.subplots()

    ax1.plot(epochs, train_acc_list, label='Train accuracy', color='b')
    ax1.plot(epochs, test_acc_list, label='Test accuracy', color='g')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0.8, 1)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.plot(epochs, train_loss_show_list, label='Loss', color='r')
    ax2.set_ylabel('Loss')
    ax2.tick_params(axis='y')
    ax2.set_ylim(0, max(train_loss_show_list) * 1.3)

    fig.tight_layout()
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(0, 1))
    plt.savefig(diagramsSavePath)
    plt.show()
