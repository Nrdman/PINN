#!/usr/bin/env python
# coding: utf-8

# In[47]:


import tensorflow as tf
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
import pandas as pd
import math


# In[49]:


#generates exact data

data  = np.zeros([101, 101]) 
x = np.zeros([101])
y = np.zeros([101])
pi = math.pi
const = -0.5*pi
for i in range(101):
    x[i] = (i)/100
    for j in range(101):
        if i==0:
            y[j]=j/100
        data[i,j] = math.sin(x[i]*const)*(math.e)**(y[j]*const)
data = data.T


# In[58]:


#makes a list of every x y combination and a list of the values
X, Y = np.meshgrid(x,y) 
X_star=np.hstack((X.flatten()[:,None],Y.flatten()[:,None])) 
u_star = data.flatten()[:,None] 

xx1=np.hstack((X[0:1,:].T,Y[0:1,:].T))  # coords for initial boundary condition  y=0
xx2 = np.hstack((X[-1:,:].T,Y[-1:,:].T)) #coords for initial bc y=1
xx3=np.hstack((X[:,0:1],Y[:,0:1]))      # coords for boundary condition for x=0 
xx4=np.hstack((X[:,-1:],Y[:,-1:]))      # coords for boundary condition for x=1
        
uu1 = data[0:1,:].T #data for y=0, this transpose is also present in the other code
uu2 = data[-1:, :].T #data for y =1
uu3 = data[:,0:1] #data for x =0
uu4 = data[:,-1:] #data for x=1
actual_outputs_1  = data[25:26,:].T
actual_outputs_2  = data[50:51,:].T
actual_outputs_3  = data[75:76,:].T

lb = X_star.min(axis=0)    # (1,2)   [-1 0]
ub = X_star.max(axis=0)    # (1,2)   [1 0.99]
no_of_interior_points=2500 #8000
no_of_collocation_points=200 #100

X_u_train = np.vstack([xx1, xx2, xx3, xx4])                           #  stacking up all data related to boundary conditions 
X_f_train = lb + (ub-lb)*lhs(2, no_of_interior_points)           # lhs is used to generate random sample of points. 2 is no of variables(x,t).
X_f_train = np.vstack((X_f_train, X_u_train))    
u_train = np.vstack([uu1, uu2, uu3, uu4])                             #  values correspoing to X_u_train
print(u_train.shape)
print(u_train)
idx = np.random.choice(X_u_train.shape[0], no_of_collocation_points , replace=False) # random sample of collocation points from boundary coords
X_u_train = X_u_train[idx, :]     # Those collocation points chosen from boundary conditions data
u_train = u_train[idx,:]     # Output corresponding to collocation points


# subscript u denotes collocation/boundary points and subscript f denotes interior points
x_u = X_u_train[:,0:1]       #Separating x,t from X_u_train
y_u = X_u_train[:,1:2]
x_f = X_f_train [:,0:1]
y_f = X_f_train [:,1:2]

x_u_tf=tf.Variable(x_u)       #Converting to tensor variable. Essential for calculating gradients later on.
y_u_tf=tf.Variable(y_u)
x_f_tf=tf.Variable(x_f)
y_f_tf=tf.Variable(y_f)
X_f_train_tf=tf.Variable(X_f_train)


# In[51]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input
def get_model():
    model=Sequential([Dense(20,activation='tanh',
                            kernel_initializer=tf.keras.initializers.GlorotNormal(),
                            input_shape=(2,),name='H1')])
    for i in range(6):
        model.add(Dense(20,activation='tanh',kernel_initializer=tf.keras.initializers.GlorotNormal(),name='H'+str(i+2)))
    model.add(Dense(1,name='output_layer'))
    return model


# In[52]:


model=get_model()
print(model.summary())


# In[53]:


#function for calculating loss wrt to interior points
def interior_loss():
    with tf.GradientTape() as tape:
        tape.watch(X_f_train_tf)
        with tf.GradientTape() as tape2:
            u_predicted=model(X_f_train_tf)
        grad=tape2.gradient(u_predicted,X_f_train_tf)
        du_dx=grad[:, 0]
        du_dy=grad[:, 1]
    j = tape.gradient(grad, X_f_train_tf)
    d2u_dx2 = j[:,0]
    d2u_dy2 = j[:,1]
    #u_predicted=tf.cast(u_predicted, dtype=tf.float64)
    d2u_dx2=tf.reshape(d2u_dx2, [len(d2u_dx2),1])
    d2u_dy2=tf.reshape(d2u_dy2, [len(d2u_dy2),1])
    #d2u_dx2=tf.cast(d2u_dx2, dtype=tf.float64)
    #d2u_dy2=tf.cast(d2u_dy2, dtype=tf.float64)
    f= -d2u_dx2-d2u_dy2  #want to f to minimize to 0
    f=tf.math.reduce_mean(tf.math.square(f)) #squares elementwise and finds average 
    #f = tf.cast(f, dtype = tf.float64)
    return f


# In[54]:


"""
LBFG-S, which is second order optimizer, has been used to update the weights and biases because conventional first order optimizers Adam, Gradient descent and RMSprop 
are slow to converge.LBFG-S is not available by default in Tensorflow 2.0 and hence a function from Tensorflow Probability has been used. This has not been coded by me
except for a minor addition to loss function(check the loss_value variable). Please refer to the link below to get a better idea.

https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993

"""
import numpy
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot

def function_factory(model, loss, train_x, train_y):
    """A factory to create a function required by tfp.optimizer.lbfgs_minimize.
    Args:
        model [in]: an instance of `tf.keras.Model` or its subclasses.
        loss [in]: a function with signature loss_value = loss(pred_y, true_y).
        train_x [in]: the input part of training data.
        train_y [in]: the output part of training data.
    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """

    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = [] # stitch indices
    part = [] # partition indices

    for i, shape in enumerate(shapes):
        n = numpy.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n
    part = tf.constant(part)
    @tf.function
    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.
        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """

        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

    # now create a function that will be returned by this factory
    @tf.function
    def f(params_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.
        This function is created by function_factory.
        Args:
           params_1d [in]: a 1D tf.Tensor.
        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.
        """

        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        with tf.GradientTape() as tape:
            # update the parameters in the model
            assign_new_model_parameters(params_1d)
            # calculate the loss
            loss_value = loss(model(train_x, training=True), train_y)
            int_loss=interior_loss()
            loss_value=loss_value+int_loss


        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)

        # print out iteration & loss
        f.iter.assign_add(1)
        tf.print("Iter:", f.iter, "loss:", loss_value)

        # store loss value so we can retrieve later
        tf.py_function(f.history.append, inp=[loss_value], Tout=[])
        
        loss_value = tf.cast(loss_value, tf.float32)
        #grads = tf.cast(grads, tf.float64)
        return loss_value, grads

    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = []

    return f


# In[55]:


def plot_helper(inputs, outputs, title, fname):

    pyplot.figure(figsize=(8,4))
    pyplot.tricontourf(inputs[:, 1], inputs[:, 0], outputs.flatten(), 100)
    pyplot.scatter(X_u_train[:, 1], X_u_train[:, 0],marker='x',s=100,c='k')
    pyplot.xlabel("y")
    pyplot.ylabel("x")
    pyplot.title(title)
    pyplot.colorbar()
    pyplot.savefig(fname)



# In[56]:


func = function_factory(model, tf.keras.losses.MeanSquaredError() ,X_u_train , u_train)
init_params = tf.dynamic_stitch(func.idx, model.trainable_variables)
results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func, initial_position=init_params, max_iterations=5000)


# In[ ]:


func.assign_new_model_parameters(results.position)
p=np.vstack([X_f_train,X_u_train])
q=np.vstack([model.predict(X_f_train),model.predict(X_u_train)])[:,0]
plot_helper(p, q, "u (x,y)", "pred_soln3.pdf")


# In[ ]:


splicenum = 0
plt.plot(data[splicenum,:], label = "Y ="+str(y[splicenum]))
plt.plot(data[:,splicenum], label = "X = "+str(x[splicenum]))

splice1  = np.array([x[splicenum]*np.ones(len(data[splicenum, :])),y ]).T
splice2 = np.array([x, y[splicenum]*np.ones(len(data[:, splicenum])) ]).T

plt.plot(model.predict(splice1), label = "X Approx")
plt.plot(model.predict(splice2), label = "Y Approx")
plt.legend()


# In[ ]:


print(X_f_train[0, :])


# In[ ]:


len(X_f_train[:,0])


# In[ ]:


plot_helper(X_star, data, "exact", "exact.pdf")


# In[ ]:


print(data)


# In[ ]:





# In[ ]:





# In[ ]:




