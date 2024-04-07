import numpy as np
import matplotlib.pyplot as plt

from predict import *
from sigmoidGradient import *
from nnCost import *

# a function that implements stochastic gradient descent using back propagation
def sGD(input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lmbda, alpha, MaxEpochs):
    Theta1 = np.random.uniform(-0.18, 0.18, (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.random.uniform(-0.18, 0.18, (num_labels, hidden_layer_size + 1))
    
    Theta1 = np.transpose(Theta1) # il+1 x hl
    Theta2 = np.transpose(Theta2) # hl+1 x l

    y_k = np.zeros((y_train.shape[0], 3))
    for i in range(y_train.shape[0]):
        y_k[i, y_train[i]-1] = 1
    
    cost = []

    # Theta1 = np.transpose(Theta1) # il+1 x hl
    # Theta2 = np.transpose(Theta2) # hl+1 x l
    for i in range(MaxEpochs):
        for j in range(X_train.shape[0]):
            X_sample = np.reshape(X_train[j][:], (1, X_train.shape[1]))
            y_sample = np.reshape(y_k[j][:], (1, y_k.shape[1]))

            # forward propagation
            # m x n+1
            a_1 = np.insert(X_sample, 0, 1, axis=1)
            # m x hl
            z_2 = np.matmul(a_1, Theta1)
            # m x hl+1
            a_2 = np.insert(sigmoid(z_2), 0, 1, axis=1)
            # m x l
            z_3 = np.matmul(a_2, Theta2)
            # m x l
            a_3 = sigmoid(z_3)
            
            # backward propagation
            # m x l
            e_3 = np.subtract(a_3, y_sample)
            # (m x l * l x hl) .* m x hl
            Theta2_temp = np.transpose(Theta2)
            e_2 = np.multiply(np.matmul(e_3, Theta2_temp[:, 1:]), sigmoidGradient(z_2))
            # n+1 x m * m x hl
            delta_1 = np.matmul(np.transpose(a_1), e_2)
            for k in range(1, delta_1.shape[1]):
                delta_1[:][k] += lmbda * Theta1[:][k]
            # hl+1 x m * m x l
            delta_2 = np.matmul( np.transpose(a_2), e_3)
            for k in range(1, delta_2.shape[1]):
                delta_2[:][k] += lmbda * Theta2[:][k]

            Theta1 = np.subtract(Theta1, alpha * delta_1)
            Theta2 = np.subtract(Theta2, alpha * delta_2)
            print('Epoch #', i, 'Sample #', j)
        cost.append(nnCost(np.transpose(Theta1), np.transpose(Theta2), X_train, y_train, num_labels, lmbda))
        print('Completed Epoch #', i)

    # fig2 = plt.figure(2)
    # plt.plot(range(MaxEpochs), cost, c='tab:blue')
    # plt.xlabel('Iteration')
    # plt.ylabel('Cost')
    # plt.title('Training Cost vs Iteration Number')
    # plt.savefig('output/ps7-4-e-1.png')
    # plt.show()
    return [Theta1, Theta2]