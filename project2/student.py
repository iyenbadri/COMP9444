"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning


The program has been broadly classified into two stages:
    1. Data Processing
    2. Data Modelling and Training
    
1. In the Data Processing stage, the model is transformed and processed so as to eliminate any unnecessary charecters. This was made using a simple regex command that eliminates all charectes except ASCII values.
   An additional regex command is added to eliminate the punctuations. This string is converted into a list and returned to the main function.
2. In the Data Modelling and Training stage, the actual model is implemented and the actual dataset is trained through this model. The model used for this training is LSTM. This model has been proven to provide more accuracy as opposed to any other  models
   The Loss Function used is CrossEntropyLoss from the Pytorch library. No user defined loss function was needed to be implemented.
   The Model uses default parameters at places where the value is left ambiguous. All other parameters that have been changed and used for this experimentation have been mentioned and done for the purpose of attaining greater accuracy.
   
   The modified parameters for testing are:
       Learning rate value: 1.0
       Dimensions= 200 (Post testing with 50,100,300 and finding the most optimal and accurate dimension for testing)
       
The details of each and every step is specified throughout the code via commenting so as to help the assessor have a better understanding of the code itself.
   
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
import re
from torchtext.vocab import GloVe
# import numpy as np
# import sklearn


###########################################################################
### The following determines the processing of input data (review text) ###
###########################################################################


def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    
    This Function accepts inputs in a string format and eliminates any unnecessary charecters using Regex
    and outputs the processed data in a List format.
    """
    input = " ".join(sample)
    rw_ip = re.sub(r"[^\x00-\x7F]+", " ", input)        # Eliminate Non-ASCII Charecters
    ip=   re.sub(r"[^\w\s]", " ", rw_ip)                # Eliminate Punctuations 
    proc_op = ip.split(" ")                             # Remove additional spaces from the end
    proc_op = list(filter(lambda x: x != '', proc_op))

    return proc_op


def postprocessing(batch, vocab):
    """
    Called after numericalisation but before vectorisation.
    """
    
    return batch

stopWords = {}
wordVectors = GloVe(name='6B', dim=200)

###########################################################################
##### The following determines the processing of label data (ratings) #####
###########################################################################


def convertLabel(datasetLabel):
    """
    Labels (product ratings) from the dataset are provided to you as
    floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    You may wish to train with these as they are, or you you may wish
    to convert them to another representation in this function.
    Consider regression vs classification.
    """
    
    # The Labels of float data type are converted into integers with range 0-4
    
    val= datasetLabel.long()
    out = val - 1
    
    return out


def convertNetOutput(netOutput):
    """
    Your model will be assessed on the predictions it makes, which
    must be in the same format as the dataset labels.  The predictions
    must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    If your network outputs a different representation or any float
    values other than the five mentioned, convert the output here.
    """
    
    # Post Computation, the labels are converted back into float types having range 1.0-5.0
    
    maxval = torch.argmax(netOutput, dim=1)
    out = (maxval + 1).float()
    
    return out


###########################################################################
################### The following determines the model ####################
###########################################################################


class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network
    will be a batch of reviews (in word vector form).  As reviews will
    have different numbers of words in them, padding has been added to the
    end of the reviews so we can form a batch of reviews of equal length.
    """
    
# Model parameters used for the experimentation
# hidden_dim = 100
# num_layers = 1
# out_dim = 5
    
# One LSTM network and one Linear Layer,
    
    def __init__(self):
        super(network, self).__init__()
        # Initialising model
        self.lstm = tnn.LSTM(input_size=200, hidden_size=200, num_layers=1, batch_first=True)       #Initialise the layer with specified i/p and o/p parameters
        self.linear = tnn.Linear(200,5)                                                            

    def forward(self, input, length):

        output, (hidden, cell) = self.lstm(input)               # Implement LSTM Network
        outputs = self.linear(hidden[-1])                       # Implement Linear Layer
        return outputs

net = network()

lossFunc = tnn.CrossEntropyLoss()

###########################################################################
################ The following determines training options ################
###########################################################################

# Hyper parameters
trainValSplit = 0.8
batchSize = 32
epochs = 10
optimiser = toptim.SGD(net.parameters(), lr=1.0)
