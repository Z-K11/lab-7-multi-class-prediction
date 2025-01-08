import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
# Start
plotting_colors = 'ryb'
plot_step =0.2

def decision_boundary(X,Y,Model,iris,two=None):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    '''x is a 2d array contains rows and columns x[rows,columns]x[:,:]selects all rows and columns
x[:,0] selects all rows from the 0th column so x[:,0].min() finds the minimum value from all
the rows in column 0 and x[:,0].max() finds the maximum value from all the rows in 0th Column'''
    y_min,y_max=Y[:,0].min()-1,Y[:,0].max()+1
    # this is the same as above explanation
    xx,yy=np.meshgrid(np.arange(x_min,x_max,plot_step),np.arange(y_min,y_max,plot_step))
    '''np.arange(min,max,step)takes a minimum value a maximum value and a step_size
    what this function does is create a 1d array and populates it with equally spaced
    values between from min to before max
    np.meshgrid(array1,array2) takes 2 1d arrays and combines the to create a grid the first
    is taken as x-axis i.e for rows and the second for y-axist i.e fro clolumns rows repeat
    array 1 values and column repeates array 2 values '''
    plt.tight_layout(h_pad=0.5,w_pad=0.5,pad=2.5)
    '''unction in Matplotlib is used to automatically adjust the layout of a figure to ensure 
    that subplots and other plot elements (like titles, axis labels, and legends) do not overlap.
    '''
    