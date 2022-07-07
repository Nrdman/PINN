#!/usr/bin/env python
# coding: utf-8

# Equation: - div(del u ) = f in [0,pi] x [0,1]
# 
# equivalent: -uxx-uyy = f
# 

# u(x,y) = g on boundary

# u(x,y) =sin(cx)sin(cy)

# ux  = c*cos(cx)sin(cy), uxx = -c^2sin(cx)sin(cy)
# 
# uy = c*sin(cx)cos(cy), uyy = -c^2sin(cx)sin(cy)

# uxx+uyy = -2c^2sin(cx)sin(cy) so f should be 2c^2sin(cx)sin(cy)

# In[17]:


import tensorflow as tf
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
import pandas as pd
import math


# In[18]:


PINN=True #toggle between PINN and plain NN


# In[19]:


#generates exact data
data  = np.zeros([101, 101]) 
x = np.zeros([101])
y = np.zeros([101])
data_coords = np.zeros([101*101, 2])
output_for_L2  = np.zeros([101*101, 1])

pi = math.pi
const = pi
for i in range(101):
    x[i] = i/100
    for j in range(101):
        if i==0:
            y[j]=j/100 
        k = 101*i +j
        data_coords[k] = [i/100,j/100]
        data[i,j] = (math.sin(x[i]*const))*math.sin(y[j]*const)
        output_for_L2[k] = data[i,j]
data = data.T


# In[20]:


#makes a list of every x y combination and a list of the values
X, Y = np.meshgrid(x,y) 
X_star=np.hstack((X.flatten()[:,None],Y.flatten()[:,None])) 
u_star = data.flatten()[:,None] 


# In[21]:


xx1=np.hstack((X[0:1,:].T,Y[0:1,:].T))  # coords for initial boundary condition  y=0
xx2=np.hstack((X[:,0:1],Y[:,0:1]))      # coords for boundary condition for x=0 
xx3=np.hstack((X[:,-1:],Y[:,-1:]))      # coords for boundary condition for x=1
xx4=np.hstack((X[-1:,:].T, Y[-1:,:].T)) #coords for boundary condition y=1

uu1 = data[0:1,:].T #data for y=0, this transpose is also present in the other code
uu2 = data[:,0:1] #data for x =0
uu3 = data[:,-1:] #data for x=pi, should technically be all 0, but the error in the pi causes it to be just very small
uu4 = data[-1:,:].T #data for y=1
actual_outputs_1  = data[25:26,:].T
actual_outputs_2  = data[50:51,:].T
actual_outputs_3  = data[75:76,:].T


# In[22]:


lb = X_star.min(axis=0)    # (1,2)   [-1 0]
ub = X_star.max(axis=0)    # (1,2)   [1 0.99]
no_of_interior_points=10000 #defaulut 8000
no_of_collocation_points=200 #default 100, do multiple of 4

X_u_train = np.vstack([xx1, xx2, xx3, xx4])                           #  stacking up all data related to boundary conditions 
X_f_train = lb + (ub-lb)*lhs(2, no_of_interior_points)           # lhs is used to generate random sample of points. 2 is no of variables(x,t).
u_train = np.vstack([uu1, uu2, uu3, uu4])                             #  values correspoing to X_u_train


# In[23]:


idxx1 = np.random.choice(xx1.shape[0], int(no_of_collocation_points/4) , replace=False)  # random sample of collocation points from boundary coords
idxx2 = np.random.choice(xx2.shape[0], int(no_of_collocation_points/4) , replace=False)  # random sample of collocation points from boundary coords
idxx3 = np.random.choice(xx3.shape[0], int(no_of_collocation_points/4) , replace=False) # random sample of collocation points from boundary coords
idxx4 = np.random.choice(xx4.shape[0], int(no_of_collocation_points/4) , replace=False) # random sample of collocation points from boundary coords
xx1 = xx1[idxx1, :]
xx2 = xx2[idxx2, :]
xx3 = xx3[idxx3, :]
xx4 = xx4[idxx4, :]
uu1 = uu1[idxx1,:]
uu2 = uu2[idxx2,:]
uu3 = uu3[idxx3,:]
uu4 = uu4[idxx4,:]


# In[24]:


#idx = np.random.choice(X_u_train.shape[0], no_of_collocation_points , replace=False) # random sample of collocation points from boundary coords
#X_u_train = X_u_train[idx, :]                                                        # Those collocation points chosen from boundary conditions data
#u_train = u_train[idx,:]     # Output corresponding to collocation points

X_u_train = np.vstack([xx1, xx2, xx3, xx4])                           #  stacking up all data related to boundary conditions 
u_train = np.vstack([uu1, uu2, uu3, uu4])                             #  values correspoing to X_u_train


# In[25]:


# subscript u denotes collocation/boundary points and subscript f denotes interior points
x_u = X_u_train[:,0:1]       #Separating x,t from X_u_train
y_u = X_u_train[:,1:2]
x_f = X_f_train [:,0:1]
y_f = X_f_train [:,1:2]


# In[26]:


x_u_tf=tf.Variable(x_u)       #Converting to tensor variable. Essential for calculating gradients later on.
y_u_tf=tf.Variable(y_u)
x_f_tf=tf.Variable(x_f)
y_f_tf=tf.Variable(y_f)
#X_all_train = np.vstack((X_f_train, X_u_train))     #puts together boundary points and interior points
X_f_train_tf= tf.Variable(X_f_train)
#X_all_train_tf=tf.Variable(X_all_train)


# In[27]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input
def get_model():
  model=Sequential([
                    Dense(20,activation='tanh',kernel_initializer=tf.keras.initializers.GlorotNormal(),input_shape=(2,),name='H1')
  ])
  
  for i in range(6):
    model.add(Dense(20,activation='tanh',kernel_initializer=tf.keras.initializers.GlorotNormal(),name='H'+str(i+2)))

  model.add(Dense(1,name='output_layer'))

  return model


# In[28]:


model=get_model()
print(model.summary())


# In[29]:


def f_star(inmatrix):
    xvalues = inmatrix[:,0]
    yvalues = inmatrix[:, 1]
    #f should be 2c^2sin(cx)sin(cy) for this problem
    
    pointnum, varnum = X_f_train_tf.shape
    output = np.zeros([pointnum]) 
    for i in range(pointnum):
        output[i] =2*const*const*math.sin(const*xvalues[i])*np.sin(const*yvalues[i])
        
    #output= 2*const*const*np.multiply(np.sin(const*xvalues), np.sin(const*yvalues))
    return output


# In[ ]:





# In[30]:


#function for calculating loss wrt to interior points
def interior_loss():
    if PINN == True:
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
        u_predicted=tf.cast(u_predicted, dtype=tf.float64)
        xlength = int(len(X_f_train_tf[:,1]))
        d2u_dx2=tf.reshape(d2u_dx2, [xlength,1])
        d2u_dy2=tf.reshape(d2u_dy2, [xlength,1])
        d2u_dx2=tf.cast(d2u_dx2, dtype=tf.float64)
        d2u_dy2=tf.cast(d2u_dy2, dtype=tf.float64)
        laplacian= d2u_dx2+d2u_dy2  
        f = f_star(X_f_train)
        f=f+laplacian #want to minimize to 0
        f=tf.math.reduce_mean(tf.math.square(f)) #squares elementwise and finds average 
    if PINN==False:
        f=0
    return f


# In[31]:


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
            loss_value = tf.cast(loss_value, tf.float64)
            int_loss = tf.cast(int_loss, tf.float64)
            loss_value=loss_value+int_loss


        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)
        # print out iteration & loss
        f.iter.assign_add(1)
        tf.print("Iter:", f.iter, "loss:", loss_value)

        # store loss value so we can retrieve later
        tf.py_function(f.history.append, inp=[loss_value], Tout=[])

        return loss_value, grads

    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = []

    return f


def plot_helper(inputs, outputs, title, fname):

    pyplot.figure(figsize=(8,4))
    pyplot.tricontourf(inputs[:, 1], inputs[:, 0], outputs.flatten(), 100)
    pyplot.scatter(X_u_train[:, 1], X_u_train[:, 0],marker='x',s=100,c='k')
    #pyplot.scatter(X_f_train[:, 1], X_f_train[:, 0],marker='.',s=.1,c='k') # code to turn on the ability to see interior test points
    pyplot.xlabel("y")
    pyplot.ylabel("x")
    pyplot.title(title)
    pyplot.colorbar()
    pyplot.savefig(fname)


# In[32]:


tf.keras.backend.set_floatx("float64")
func = function_factory(model, tf.keras.losses.MeanSquaredError() ,X_u_train , u_train)
init_params = tf.dynamic_stitch(func.idx, model.trainable_variables)
results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func, initial_position=init_params, max_iterations=1000)


# In[33]:


func.assign_new_model_parameters(results.position)


# In[34]:


p=np.vstack([X_f_train,X_u_train])
q=np.vstack([model.predict(X_f_train),model.predict(X_u_train)])[:,0]


# In[35]:


print(q[7888])
print(p[7888])
np.min(q)


# In[36]:


plot_helper(p, q, "u (x,y)", "pred_soln3.pdf")


# In[37]:


x_t_0=np.hstack(( X[0:1,:].T, Y[0:1,:].T))
x_t_25=np.hstack(( X[0:1,:].T, Y[25:26,:].T))
x_t_50=np.hstack(( X[0:1,:].T, Y[50:51,:].T))
x_t_75=np.hstack(( X[0:1,:].T, Y[75:76,:].T))
uu1 = data[0:1,:].T #data for y=0, this transpose is also present in the other code
uu2 = data[:,0:1] #data for x =0
uu3 = data[:,-1:] #data for x=pi, should technically be all 0, but the error in the pi causes it to be just very small
uu4 = data[-1:,:].T #data for y=1


# In[38]:


#graphs for given initial value, should be exact
fig, axs = plt.subplots(1, 2, figsize=(8,4) ,sharey=True)

l1,=axs[0].plot(x, uu1,linewidth=6,color='b')
l2,=axs[0].plot(x_t_0[:,0],model.predict(x_t_0),linewidth=6,linestyle='dashed',color='r')
axs[0].set_title('y=0')
axs[0].set_xlabel('x')
axs[0].set_ylabel('u (x,t)')
line_labels = ['Exact','Predicted']

fig.legend(handles=(l1,l2),labels=('Exact','Predicted'),loc='upper right')
pyplot.savefig('graphs4.pdf')


# In[39]:


fig, axs = plt.subplots(1, 3, figsize=(8,4) ,sharey=True)

l1,=axs[0].plot(x, actual_outputs_1,linewidth=6,color='b')
l2,=axs[0].plot(x_t_25[:,0],model.predict(x_t_25),linewidth=6,linestyle='dashed',color='r')
axs[0].set_title('y=0.25')
axs[0].set_xlabel('x')
axs[0].set_ylabel('u (x,t)')


axs[1].plot(x, actual_outputs_2,linewidth=6,color='b')
l2,=axs[1].plot(x_t_50[:,0],model.predict(x_t_50),linewidth=6,linestyle='dashed',color='r')
axs[1].set_title('y=0.50')
axs[1].set_xlabel('x')

axs[2].plot(x, actual_outputs_3,linewidth=6,color='b')
l2,=axs[2].plot(x_t_75[:,0],model.predict(x_t_75),linewidth=6,linestyle='dashed',color='r')
axs[2].set_title('y=0.75')
axs[2].set_xlabel('x')

line_labels = ['Exact','Predicted']

fig.legend(handles=(l1,l2),labels=('Exact','Predicted'),loc='upper right')
pyplot.savefig('graphs3.pdf')


# In[40]:


int_loss1 = interior_loss()
print("Interior Loss:", int_loss1)
print("Boundary Loss:", 33.045427800894878-int_loss1)#change the 33


# In[41]:


full_predicted = model.predict(data_coords)


# In[42]:


full_predicted.shape


# In[43]:


output_for_L2.shape


# In[44]:


from numpy.linalg import norm 
L2_abs_error= norm(full_predicted-output_for_L2,2)
L2_rel_error = L2_abs_error/norm(output_for_L2,2)


# In[45]:


print(L2_abs_error)
print(L2_rel_error)


# In[ ]:




