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
    y_min,y_max=X[:, 1].min() - 1, X[:, 1].max() + 1
    # this is the same as above explanation
    xx,yy=np.meshgrid(np.arange(x_min,x_max,plot_step),np.arange(y_min,y_max,plot_step))
    '''np.arange(min,max,step)takes a minimum value a maximum value and a step_size
    what this function does is create a 1d array and populates it with equally spaced
    values between from min to before max
    np.meshgrid(array1,array2) takes 2 1d arrays and combines the to create a grid the first
    is taken as x-axis i.e for rows and the second for y-axist i.e from columns rows repeat
    array 1 values and column repeates array 2 values '''
    plt.clf()
    plt.tight_layout(h_pad=0.5,w_pad=0.5,pad=2.5)
    '''function in Matplotlib is used to automatically adjust the layout of a figure to ensure 
    that subplots and other plot elements (like titles, axis labels, and legends) do not overlap.
    '''
    z = Model.predict(np.c_[xx.ravel(),yy.ravel()])
    '''Used to generate predictions from the machine learning model for every point on the grid
    here xx and yy are the 2d grid arrays the .ravel() functions flatens a 2d array into a 1darray
    so we end up with a row matrix where when the first row ended in the original matrix ? the 2nd
    row added at the end of the first row and the third row is added at the end of the second row
    which was added at the end of the first row so we have one large row matrix
    the np.c_ function concatinates arrays along the y axis basically the first row matrix is 
    converted into column matrix and used as first matrix, the second row matrix is also converted
    into a column matrix and added as the second column of the concatenated matrix '''
    z = z.reshape(xx.shape)
    '''Reshape aligns the predictions with the grid so that they can be visualized'''
    cs = plt.contourf(xx,yy,z,cmap=plt.cm.RdYlBu)
    '''This shows the decision boundary where the model transitions between different predicted
    classes '''
    if two:
        cs =plt.contourf(xx,yy,z,cmap=plt.cm.RdYlBu)
        for i,color in zip(np.unique(Y),plotting_colors):
            '''Loop to iterate through unique class labels in Y and assign a color to each class
            np.unique(Y) finds all the unique values in Y and returns them in a sorted manner
            .zip() combines two sequences in our case np.unique(Y) and plotting_colors element
            by element
            if np.unique(y)=[1,2,3] and plottint_colors = 'ryb' then zip() function will create
            [1,r],[2,y],[3,b]'''
            idx=np.where(Y==i)
            '''y==1 creates a boolean array that is the same shape as the original array but has 
            value true for which ever index y==1
            The np.where function returns returns the indices of the elements where the condition
            was true ? '''
            plt.scatter(X[idx, 0], X[idx, 1], label=y, cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

        plt.savefig('./pngFiles/contour.png')
        
    else:
        set_={0,1,2}
        print(set_)
        for i,color in zip(range(3),plotting_colors):
            idx=np.where(Y==1)
            if np.any(idx):
                set_.remove(i)
                plt.scatter(X[idx,0],X[idx,1],label=Y,cmap=plt.cm.RdYlBu,edgecolor='black',s=15)
        for i in set_:
            idx=np.where(iris.target==i)
            plt.scatter(X[idx,0],X[idx,1],marker='x',color='black')
        plt.savefig('./pngFiles/contour_plot.png')
def plot_probability_array(X,probability_array):
    plot_array=np.zeros((X.shape[0],30))
    '''plot_array is going to have the same rows as X becuase X.shape[0] returns the number of rows
    of x the second number 30 means 30 columns it is the input to the number of rows of the array
    so in the end plot_array is going to have rows equal to number of rows of array x 30 columns 
    all filled with 0 '''
    col_start = 0
    ones =np.ones((X.shape[0],30))
    '''same as plot_array but ones is going to have 1'''
    for class_,col_end in enumerate([10,20,30]):
        plot_array[:,col_start:col_end]=np.repeat(probability_array[:,class_].reshape(-1,1),10,axis=1)
        col_start=col_end
    '''plot_array[:,co
    l_start:col_end] is using slices to select only subset of rows and columns
    from plot_array[]'''
    plt.clf()
    plt.imshow(plot_array)
    '''plt.imshow() takes a 2d array as input and visualizes it as an image'''
    plt.xticks([])
    '''ticks are markers along an axis xticks mean ticks along x axis here we are passing an empty
    list meaning there will be no markers along any axis'''
    plt.ylabel("samples")
    '''Is the ylabel of y axis'''
    plt.xlabel("Probability of 3 classes")
    '''label of x axis'''
    plt.colorbar()
    '''plt.colorbar() adds colorbar to the plot A color bar is a visual representation of the 
    mapping between numeric values and colors assigned to them'''
    plt.savefig("./pngFiles/probability_plot")
pair =[1,3]
'''1darray'''
iris = datasets.load_iris()
'''loads the iris data set from sklearn.dataset or whatever'''
x = iris.data[:,pair]
'''rox,columns : for rows selets all the rows, pair for columns selects column 1 and 3.
as our feature dataset'''
y=iris.target
array=np.unique(y)
'''returns unique values from within y to array'''
plt.clf()
plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.RdYlBu)
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Petal Width")
plt.savefig('./pngFiles/iris.png')
logregression = LogisticRegression(random_state=2).fit(x,y)
'''Fiting datasets to the model'''
probability =logregression.predict_proba(x)
'''saving predictions'''
plot_probability_array(x,probability)
'''plotting the predictions using the function'''
print(probability[0,:])
print(probability[0,:].sum())
'''sums the contants of all the columns in row 0 ?'''
print(np.argmax(probability[0,:]))
'''aplies the argmax function for softmax_regression'''
softmax_regression = np.argmax(probability,axis=1)
'''applying the argmax function to the predictions made by the model'''
print(softmax_regression)
predictions = logregression.predict(x)
print("Accuracy Score",accuracy_score(predictions,softmax_regression))
from sklearn import svm
model = svm.SVC(kernel='linear',gamma=0.5,probability=True).fit(x,y)
svm_predictions = model.predict(x)
print("svm Accuracy score ",accuracy_score(y,svm_predictions))
decision_boundary(x,y,model,iris,1)