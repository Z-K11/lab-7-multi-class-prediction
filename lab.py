# Start
plotting_colors = 'ryb'
plot_step =0.2

def decision_boundary(X,Y,Model,iris,two=None):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
'''x is a 2d array contains rows and columns x[rows,columns]x[:,:]selects all rows and columns
   x[:,0] selects all rows from the 0th column so x[:,0].min() finds the minimum value from all
   the rows in column 0 and x[:,0].max() finds the maximum value from all the rows in 0th Column'''
   