import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import types
import math
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize

def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    if isinstance(z,float):
        z=1/(1+math.exp(-1*z))
        return z
    else:
        z=np.multiply(z,-1)
        z=np.exp(z)
        z=np.add(z,1)
        z=np.divide(1,z)
        return z
    #your code here
    
    

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
   
    #Creating train data
    train= mat.get('train0')
    
    
    vec=np.zeros(len(train))[...,None]
    #print vec.shape
    for i in range(1,10):
        traint= mat.get('train'+str(i))
        train=np.append(train,traint,axis=0)

        vect=np.zeros(len(traint))[...,None]
        vect=vect+i
        vec=np.append(vec,vect,axis=0)

    vec=vec.astype(np.int64)
    #print vec
    #Test data
    test= mat.get('test0')
    vectest=np.zeros(len(test))[...,None]
  
    for i in range(1,10):
        testt= mat.get('test'+str(i))
        test=np.append(test,testt,axis=0)

        vectestt=np.zeros(len(testt))[...,None]
        vectestt=vectestt+i
        vectest=np.append(vectest,vectestt,axis=0)

    vectest=vectest.astype(np.int64)

    #Normalize
    train=train.astype(np.float64)
    test=test.astype(np.float64)

    train_rows= train.shape[0]
    test_rows= test.shape[0]

    for i in range(0,train_rows):
        temp=train[i,:]
        temp_max=temp.max()
        train[i,:]=temp/temp_max
        
    for i in range(0,test_rows):
        temp=test[i,:]
        temp_max=temp.max()
        test[i,:]=temp/temp_max
    
   
    #Randomly split
    

    combined=np.append(train,vec,axis=1)
    # print (combined.shape)
    
    #train_comb, valid_comb = train_test_split(combined, test_size = 0.16666)
    
    a = range(combined.shape[0])
    #print (a)
    aperm = np.random.permutation(a)
    train_comb = combined[aperm[0:50000],:]
    valid_comb = combined[aperm[50000:],:]
    #print (train_comb.shape)
    #print (valid_comb.shape)
    disint=np.split(train_comb,[784],1)
    train_data=disint[0]
    train_label=disint[1]

    disint=np.split(valid_comb,[784],1)
    validation_data=disint[0]
    validation_label=disint[1]
    test_data = test
    test_label = vectest

    ##print (train_data.shape)
    #print (train_label.shape)

    # print (validation_data.shape) 
    # print (validation_label.shape)
    
    #Feature Selection
    variance= np.var(train,0).astype(np.int64)

    
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    
    
    

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, the training data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


    
    #print w1.shape
    #print w2.shape

    #print training_data.shape
    #ACTUALLY NEED THE EXACT BVALUES HERE; INSTEAD OF THE THRESHOLDED. SO WRITE THE METHOD HERE AGAIN OR WRITE A NEW METHOD
    data=(training_data)
    w1temp=np.transpose(w1)
    w2temp=np.transpose(w2)

    temp=np.ones(len(data))[...,None]  #adding 1s to data
    data=np.append(data,temp,axis=1)
    #print w1
    
    a=np.dot(data,w1temp) #getting first sum-product at hidden node
    #print res
    #print res
    z=sigmoid(a)  #applying sigma on every entry
    #print z
    #columns=z.shape[0]
    temp=np.ones(len(z))[...,None]  #adding 1s to hidden node values
    z=np.append(z,temp,axis=1)
    
    b=np.dot(z,w2temp) #getting final sum-product at output node
    #print res1
    o=sigmoid(b) #applying sigma on every entry

    print (o.shape)
    oneOfK=label_binarize(training_label, classes=[0,1,2,3,4,5,6,7,8,9])
    print (oneOfK.shape)
    obj_val = 0  
    
    #Your code here
    #
    #
    #
    #
    #
    
    
    
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    
    return (obj_val,obj_grad)



def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    #data=np.array([[1,2,3],[3,2,3],[2,3,4]]) #data=3 training with 4 attributes each


    #w1=np.array([[0.1,0.2,0.1,0.2],[0.2,0.3,0.1,0.3]]) #two hidden nodes so two rows
    #w2=np.array([[0.1,0.1,0.2],[0.2,0.2,0.2],[0.1,0.2,0.3],[0.2,0.1,0.2]]) #4 output nodes so 4 rows
    
    #print w1.shape
    temp=np.ones(len(data))[...,None]  #adding 1s to data
    data=np.append(data,temp,axis=1)
    #print w1
    w1t=np.transpose(w1)
    a=np.dot(data,w1t) #getting first sum-product at hidden node
    #print res
    #print res
    z=sigmoid(a)  #applying sigma on every entry
    #print z
    #columns=z.shape[0]
    temp=np.ones(len(z))[...,None]  #adding 1s to hidden node values
    z=np.append(z,temp,axis=1)
    w2t=np.transpose(w2)
    b=np.dot(z,w2t) #getting final sum-product at output node
    #print res1
    o=sigmoid(b) #applying sigma on every entry
    #print l
    #print l
    labels = np.amax(o, axis=1) # using maximum out of all output values
    #print (labels.shape)
    
    return labels
    



"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);


# WE NEED FLOAT WEIGHTS!

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.
#nnPredict(initial_w1,initial_w2,train_data)

#nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
#w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
#w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
#print w1.shape
#print w2.shape

#Test the computed parameters

# predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

# print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

# predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

# print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


# predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Test Dataset

# print('\n Test set Accuracy:' + + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
