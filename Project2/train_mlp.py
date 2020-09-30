"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import cifar10_utils
import torch
import torch.nn as nn

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100 100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = "./data/"

FLAGS = None


class MLP(nn.Module):
  def __init__(self, n_inputs, n_hidden, n_classes, neg_slope):
    super(MLP, self).__init__()
    layer_list = []
    if n_hidden:
      for nr_nodes in n_hidden:
        layer_list.append(nn.Linear(n_inputs, nr_nodes))
        layer_list.append(nn.LeakyReLU(neg_slope))
        n_inputs = nr_nodes
    layer_list += [nn.Linear(n_inputs, n_classes)]

    self.layers = nn.ModuleList(layer_list)

    print(self.layers)


  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    out = x

    return out




def accuracy(predictions, targets):

  predictions_np = predictions.cpu().detach().numpy()
  targets_np = targets.cpu().detach().numpy()

  prediction_labels = np.argmax(predictions_np, axis=1)
  targets = np.argmax(targets_np, axis=1)
  accuracy = np.mean(prediction_labels == targets)

  return accuracy

def train():

  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  # Get negative slope parameter for LeakyReLU
  neg_slope = FLAGS.neg_slope

  # use GPU if available
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Device :", device)

  # load data and sample the first batch
  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
  x_np, y_np = cifar10['train'].next_batch(FLAGS.batch_size)
  x_np = x_np.reshape(FLAGS.batch_size, -1) # batchsize * pixels per image

  # initialize MLP
  nn = MLP(x_np.shape[1], dnn_hidden_units, 10, neg_slope).to(device)
  crossEntropy = torch.nn.CrossEntropyLoss()

  # initialize optimizer
  optimizer = torch.optim.SGD(nn.parameters(), lr=FLAGS.learning_rate)


  # initialization for plotting and metrics
  test_accuracies = []
  training_losses = []
  training_accuracies = []
  test_losses = []

  # extract test data
  x_test, y_test_np = cifar10['test'].images, cifar10['test'].labels
  x_test = x_test.reshape(x_test.shape[0], -1)
  x_test = torch.from_numpy(x_test).to(device)
  y_test = torch.from_numpy(y_test_np).to(device)


  # perform the forward step, backward step and updating of weights max_steps number of times,
  for step in range(FLAGS.max_steps):
    if (step+1)%100 == 0:
      print(step+1, "/", FLAGS.max_steps, "\n")

    optimizer.zero_grad()

    x = (torch.autograd.Variable(torch.from_numpy(x_np), requires_grad=1)).to(device)
    y = (torch.autograd.Variable(torch.from_numpy(y_np), requires_grad=1)).to(device)

    pred = nn(x).to(device)

    train_acc = accuracy(pred, y)

    # compute cross entropy loss
    labels = torch.max(y, 1)[1]
    loss = crossEntropy(pred, labels)

    training_accuracies.append(train_acc)
    training_losses.append(loss)

    # evaluation on test set
    if step % FLAGS.eval_freq == 0:
      test_accuracies, test_losses = eval_on_test(nn, crossEntropy, x_test, y_test, test_accuracies, test_losses)

    # get a next batch
    x_np, y_np = cifar10['train'].next_batch(FLAGS.batch_size)
    x_np = x_np.reshape(FLAGS.batch_size, -1) # batchsize * pixels per image

    loss.backward()
    optimizer.step()


  # compute loss and accuracy on the test set a final time
  test_accuracies, test_losses = eval_on_test(nn, crossEntropy, x_test, y_test, test_accuracies, test_losses)
  print("Maximum accuracy :", max(test_accuracies), "after %d steps\n"%(np.argmax(test_accuracies) * FLAGS.eval_freq))


def eval_on_test(nn, crossEntropy, x_test, y_test, test_accuracies, test_losses):
  """
  Find the accuracy and loss on the test set, given the current weights
  """
  test_pred = nn(x_test)
  true_labels = torch.max(y_test, 1)[1]
  test_acc = accuracy(test_pred, y_test)
  test_loss = crossEntropy(test_pred, true_labels)
  print("Test accuracy is:", test_acc, "\n")
  test_accuracies.append(test_acc)
  test_losses.append(test_loss)

  return test_accuracies, test_losses




def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  parser.add_argument('--neg_slope', type=float, default=NEG_SLOPE_DEFAULT,
                      help='Negative slope parameter for LeakyReLU')


  FLAGS, unparsed = parser.parse_known_args()
  FLAGS.amsgrad = bool(FLAGS.amsgrad)

  main()


