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
    a = range(combined.shape[0])
    aperm = np.random.permutation(a)
    train_comb = combined[aperm[0:50000],:]
    valid_comb = combined[aperm[50000:],:]
    disint=np.split(train_comb,[784],1)
    train_data=disint[0]
    train_label=disint[1]
    disint=np.split(valid_comb,[784],1)
    validation_data=disint[0]
    validation_label=disint[1]  
    test_data = test
    test_label = vectest

    threshold=0
    variance= np.var(train_data,0).astype(np.float64)[...,None]
    print ("len of variance", len(variance))
    # print(variance)
    global idx
    idx = variance[:,0] > threshold
    # print (idx)
    # variance[idx,0] = 0
    print (np.bincount(idx)[0])
    reqdCols = np.zeros(np.bincount(idx)[0])[...,None]
    # reqdCols = np.empty([1])
    counter2=0

    for count in range (0, len(variance)):
        # print(variance[count])
        if(variance[count]<= 0):
            # print(variance[count])
            # print(threshold)
            reqdCols[counter2,0] = count
            counter2 = counter2 + 1
    # print (reqdCols)

    train_data = np.delete(train_data, reqdCols,1)
    validation_data = np.delete(validation_data,reqdCols, 1)
    test_data = np.delete(test_data, reqdCols, 1)
   
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

    data=(training_data)
    w1temp=np.transpose(w1)
    w2temp=np.transpose(w2)

    temp=np.ones(len(data))[...,None]  #adding 1s to data
    x=np.append(data,temp,axis=1)
    a=np.dot(x,w1temp) #getting first sum-product at hidden node
    z=sigmoid(a)  #applying sigma on every entry
    temp=np.ones(len(z))[...,None]  #adding 1s to hidden node values
    z=np.append(z,temp,axis=1)
    b=np.dot(z,w2temp) #getting final sum-product at output node
    o=sigmoid(b) #applying sigma on every entry
    
    oneOfK = np.zeros((len(training_label), 10))
    for label in range(0,len(training_label)):
        oneOfK[label][math.floor(training_label[label])] = 1

    y=oneOfK
    onesarray=np.ones(y.shape)
    obj_val = 0 

    delta1= y-o
    delta2= onesarray-o
    delta=delta1*delta2
    delta=delta*o
    Jpw2=np.dot(np.transpose(delta),z)
    Jpw2=-Jpw2
    Jpw2=Jpw2+ w2*lambdaval
    grad_w2=Jpw2/len(y)
    grad_w2=grad_w2[...,None]

    newones=np.ones(z.shape)
    delta1=newones-z
    delta1=delta1*z
    delta1=-delta1
    delta2=np.dot(delta,w2)
    delta=delta1*delta2
    delta=np.delete(delta,delta.shape[1]-1,1)
    Jpw1=np.dot(np.transpose(delta),x)
    Jpw1=Jpw1+lambdaval*w1
    grad_w1=Jpw1/len(y)

    sca1=(y-o)*(y-o)
    sca2=np.sum(sca1,axis=1)[...,None]
    sca2=sca2/2
    sca3=np.sum(sca2,axis=0)[...,None]
    scalar1=np.asscalar(sca3)/50000
    
    sca1=w1*w1
    sca2=np.sum(sca1,axis=1)[...,None]
    sca3=np.sum(sca2,axis=0)[...,None]
    scalar2=np.asscalar(sca3)
   
    sca1=w2*w2
    sca2=np.sum(sca1,axis=1)[...,None]
    sca3=np.sum(sca2,axis=0)[...,None]
    scalar3=np.asscalar(sca3)

    scasum=scalar2+scalar3
    scasum=lambdaval*scasum
    scasum=scasum/(2*len(y))

    scalar=scalar1+scasum
    obj_val=scalar
    
    gr1=grad_w1.flatten()
    gr2=grad_w2.flatten()
    gr1=gr1[...,None]
    gr2=gr2[...,None]
    
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    
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
    
    temp=np.ones(len(data))[...,None]  #adding 1s to data
    x=np.append(data,temp,axis=1)
    w1t=np.transpose(w1)
    a=np.dot(x,w1t) #getting first sum-product at hidden node
    z=sigmoid(a)  #applying sigma on every entry
    temp=np.ones(len(z))[...,None]  #adding 1s to hidden node values
    z=np.append(z,temp,axis=1)
    w2t=np.transpose(w2)
    b=np.dot(z,w2t) #getting final sum-product at output node
    #print res1
    o=sigmoid(b) #applying sigma on every entry
    #print l
    #print l
    labels=np.argmax(o,axis=1)
    lene=len(labels)
    labels=np.asarray(labels)
    labels=np.reshape(labels,(-1,lene))
    labels=np.transpose(labels)
    
    return labels
    



"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 



# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 20;
# set the regularization hyper-parameter
lambdaval = 2;
n_class = 10;                  

initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)


opts = {'maxiter' : 50}    # Preferred value.
# while n_hidden<51:
lambdaval=0
while lambdaval<1.1:
        # set the number of nodes in output unit
    print (n_hidden)
    print (lambdaval)
    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)
    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

    w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


    predicted_label = nnPredict(w1,w2,train_data)
        
    print('\n Training set Accuracy for n_hidden= '+str(n_hidden)+' and lambdaval= '+str(lambdaval)+' is ' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

    predicted_label = nnPredict(w1,w2,validation_data)

    print('\n Validation set Accuracy for n_hidden= ' +str(n_hidden)+' and lambdaval= '+str(lambdaval)+' is '+ str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


    predicted_label = nnPredict(w1,w2,test_data)

    print('\n Test set Accuracy for n_hidden=' +str(n_hidden)+' and lambdaval= '+str(lambdaval)+' is '+   str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

    lambdaval=lambdaval+0.2
    # n_hidden=n_hidden+5
    
				   
