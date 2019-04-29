# -*- coding: utf-8 -*-
"""
@author: Paritosh
"""

# Movie Recommendation System using Stacked Autoencoders

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# importing the module of torch to implement neural networks
import torch.nn as nn

# importing the module of torch for parallel processing
import torch.nn.parallel

# importing the module of torch for the optimizer
import torch.optim as optim

# importing the tools which we'll use
import torch.utils.data

# importing for stochastic gradient descent
from torch.autograd import Variable
import matplotlib.pyplot as plt
import csv

# Importing the dataset

# here I have specified the separator, the header = None (since the default header is infer if there's no header in the csv file,
# engine as pyhton to ensure that the csv file is loaded correctly and encoding as latin-1 to support all special characters in the movie names)
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python',encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python',encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python',encoding = 'latin-1')

#extra part
movies.columns = ['MovieID', 'Title', 'Genres']
ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
users.columns = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip Code']



# Preparing the training set and test set(UserID::MovieID::Rating::Timestamp)
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
number_of_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
number_of_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into a format with users in lines and movies in columns
def convert(data):
    new_data = []
    for user_id in range(1, number_of_users + 1):
        movie_id = data[:,1][data[:,0] == user_id]
        rating_id = data[:,2][data[:,0] == user_id]
        ratings = np.zeros(number_of_movies)
        ratings[movie_id - 1] = rating_id
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

#Converting the data into torch tensor
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network for Stacked Autoencoders
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        # building 2 layers for encoding and decoding respectively
        # encoding layers
        self.fc1 = nn.Linear(number_of_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        # decoding layers 
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, number_of_movies)
        # using the sigmoid activation function
        self.activation = nn.Sigmoid()
        
    # pass the input vector    
    def forward(self, i):
        
        # encoding
        i = self.activation(self.fc1(i))
        i = self.activation(self.fc2(i))
        
        # decoding
        i = self.activation(self.fc3(i))
        i = self.fc4(i)
        return i
    
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)  

# Training the SAE
nb_epochs = 200
for epoch in range(1, nb_epochs+1):
    train_loss = 0
    # stores the number of users who rated at least one movie 
    s = 0.
    for user_id in range(number_of_users):
        input = Variable(training_set[user_id]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            # memory optimization 
            # the optimization will not be apllied to target and
            # vectors corresponding to user who did not rate any movies
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            # mean_corrector is needed to compute only the average of the loss for which the user provided ratings
            # added 1e-10 to avoid dividing by 0 which would lead to infinite computations
            mean_corrector = number_of_movies/float(torch.sum(target.data > 0) + 1e-10)
            # decides the direction of weight updates
            loss.backward()
            train_loss +=  np.sqrt(loss.data[0] * mean_corrector)
            s += 1.
            # decides the intensity of the weight updates
            optimizer.step()
    print('epoch: '+ str(epoch) + 'loss: ' + str(train_loss/s))


# Testing the SAE
test_loss = 0
# stores the number of users who rated at least one movie 
s = 0.
for user_id in range(number_of_users):
    input = Variable(training_set[user_id]).unsqueeze(0)
    target = Variable(test_set[user_id])
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        # memory optimization 
        # the optimization will not be apllied to target and
        # vectors corresponding to user who did not rate any movies
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        # mean_corrector is needed to compute only the average of the loss for which the user provided ratings
        # added 1e-10 to avoid dividing by 0 which would lead to infinite computations
        mean_corrector = number_of_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss +=  np.sqrt(loss.data[0] * mean_corrector)
        s += 1.
print('test loss: ' + str(test_loss/s))


# Tuning and Evaluation of the model

data_500 = pd.read_csv('epoch500.txt', sep = ':', header = None)
training_loss = data_500[:500]

data_test_loss_lr = pd.read_csv('test_loss_lr.txt', sep = ' ', header = None)
learning_rates = data_test_loss_lr[0]
test_loss = data_test_loss_lr[1]

# function to plot training loss vs Number of iterations
def generate_plot_trainin_loss_vs_iterations(x,y):
    plt.yticks(np.arange(0.85, 1.1, 0.025))
    # plotting training loss vs Number of iterations
    plt.scatter(x, y)

    # setting the title, X and Y label for the plot
    plt.title('Plotting Training loss vs Number of iterations')
    plt.xlabel('Number of iterations', fontsize=14)
    plt.ylabel('Training loss', fontsize=14)

    # plotting the line
    plt.plot(x, y, color='red', linestyle='dashed')

    plt.show()
    
generate_plot_trainin_loss_vs_iterations(range(0,500), training_loss[2])


# function to plot test loss vs learning rate
def generate_plot_test_loss_vs_lr(x,y):
    plt.yticks(np.arange(0.85, 1, 0.025))
    # plotting test loss vs learning rate
    plt.scatter(x, y)

    # setting the title, X and Y label for the plot
    plt.title('Plotting Test loss vs Learning Rate')
    plt.xlabel('Learning Rate', fontsize=14)
    plt.ylabel('Test loss', fontsize=14)

    # plotting the line
    plt.plot(x, y, color='red', linestyle='dashed')

    plt.show()


generate_plot_test_loss_vs_lr(learning_rates, test_loss)








    
        
        