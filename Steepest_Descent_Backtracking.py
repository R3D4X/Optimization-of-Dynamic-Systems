# import packages for scientific python and plotting
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import Bananafunction as fkt
#import Himmelblaufunction as fkt

a = 1;
b = 100;
x_size = 10;
y_size = 20; 

#fkt.plot_funct(a, b, x_size, y_size);
#fkt.plot_funct(1, 100, 20, 20);
#fkt.plot_funct(1, 100, 30, 20);
# def plot_funct(a, b, x_size, y_size):

#Starting point
theta = [2,1];
theta[0] = -999;
theta[1] =  999;

while(theta[0] < -10 or theta[0] > 10):
    theta[0] = int(input("Startpoint X [-10,10]: "))
    if(theta[0] < -10 or theta[0] > 10):
        print("Wrong value. x is in [-10,10]");
while(theta[1] < -20 or theta[1] > 20):
    theta[1] = int(input("Startpoint Y [-20,20]: "))
    if(theta[1] < -20 or theta[1] > 20):
        print("Wrong value. y is in [-20,20]");

theta_history = np.zeros((2,50));
theta_history[0][0] = theta[0]; #Add starting point to theta_history so we can plot our guess
theta_history[1][0] = theta[1];

#def plot_funct_with_Theta(a, b, x_size, y_size, x_theta, y_theta):
fkt.plot_funct_with_Theta(a, b, x_size, y_size,theta_history);    #Plot the function and our starting point

print("Theta = [ " + str(theta[0]) + " , " + str(theta[1]) + "]");

grad = fkt.grad(a, b, theta[0], theta[1]);
print("Gradient bananafkt = [ " + str(grad[0]) + " , " + str(grad[1]) + " ]");

j_firstvalue = fkt.funct(a, b, theta[0], theta[1]);
print("Value of objective fkt at the beginning = " + str(j_firstvalue));

old_grad = [0,0];

current_iteration = 0;

while( ((np.abs(grad[0]) > 10e-3) or (np.abs(grad[1]) > 10e-3)) and 
        current_iteration <= 1000):

############################### Calculate Direction ####################################################################

    #Steepest descend direction
    p_sd = -1 * fkt.grad(a, b, theta[0], theta[1]);
    #def grad(a, b, x, y)

    #print("p_sd = [ " + str(p_sd[0]) + " , " + str(p_sd[1]) + " ]");

############################### 1 Dimensional Search  ###################################################################

    a = 1;          #Steplength at the beginning
    a_min = 10e-9; #Minimum steplength
    c = 0.75;        #Decrease factor
    stop = -1;      #Boolean for poor people

    new_theta = [0,0];          #Thetabuffer
    new_theta[0] = theta[0];
    new_theta[1] = theta[1];

    j_old = fkt.funct(a, b, new_theta[0], new_theta[1]); #Old (current) value of the objective function

    i_cnt = 0;
    while (stop != 1):
        i_cnt += 1;
        #def funct(a, b, x, y):
        new_theta[0] = theta[0] + a * p_sd[0];
        new_theta[1] = theta[1] + a * p_sd[1];
        j = fkt.funct(a, b, new_theta[0], new_theta[1]);
        #print("Theta = [ " + str(theta[0]) + " , " + str(theta[1]) + "] -> J(Theta) = " + str(j_old));
        #print("New Theta = [ " + str(new_theta[0]) + " , " + str(new_theta[1]) + "] -> J(New_Theta) = " + str(j));
        #print(j);
        if( j < 0.3*j_old ):
            stop = 1;
            #print("Break j < 0.1 * j_old Iteration: " + str(i_cnt));
        elif( a < a_min ):
            stop = 1;
            #print("Break a < a_min Iteration: " + str(i_cnt));
            #print("Break " + str(a) + " < " + str(a_min) + " Iteration: " + str(i_cnt));
        else:
            a = c * a;

    print("New Theta = [ " + str(new_theta[0]) + " , " + str(new_theta[1]) + "] -> J(New_Theta) = " + str(j));
############################### Calculate New Iterate ####################################################################
#
#        new_theta[0] = theta[0] + a * p_sd[0];
#        new_theta[1] = theta[1] + a * p_sd[1];
#
#       Those values will be updated througout the whole loop. If the program exits the loop, the value is set.
#       This value will be stored to be plotted later.
#
##########################################################################################################################


    #print("Size of theta_history[0].size = " + str(theta_history[0].size));
    if current_iteration >= theta_history[0].size:  #If theta-buffer (for plotting) is full
        interm_buffer = np.zeros((2,(theta_history[0].size + 100))); #Allocate memory with space for 100 more values
        for a in range(theta_history[0].size - 1):  #Copy old values to new buffer
            interm_buffer[0][a] = theta_history[0][a];
            interm_buffer[1][a] = theta_history[1][a];
        theta_history = interm_buffer;  #Thetahistory points to new buffer
        interm_buffer = None; #Buffer points to NULL

    theta_history[0][current_iteration] = theta[0]; #Store old theta in history so we can plot all thetas later
    theta_history[1][current_iteration] = theta[1];

    theta[0] = new_theta[0];    #Store the new theta as current theta
    theta[1] = new_theta[1];

    old_grad[0] = grad[0];      #Store the current gradient as old gradient
    old_grad[1] = grad[1];
    grad = fkt.grad(a, b, theta[0], theta[1]);  #Calculate the current gradient

################################### Debug ##########################################################
#    print("Gradient bananafkt = [ " + str(grad[0]) + " , " + str(grad[1]) + " ]");
#    print("Old gradient bananafkt = [ " + str(old_grad[0]) + " , " + str(old_grad[1]) + " ]");
#    print("New Theta = [ " + str(theta[0]) + " , " + str(theta[1]) + "]");
#    print("Old Objective Function = " + str(j_old));
#    print("New Objective Function = " + str(fkt.funct(a, b, new_theta[0], new_theta[1])));
####################################################################################################

    if((np.abs(grad[0]) < 10e-3) or (np.abs(grad[1]) < 10e-3)): #Check wether the gradient is sufficiently small (Minima)
        print(str(np.abs(grad[0])) + " > " + str(10e-3) + " or " + str(np.abs(grad[1])) + " > 10e-3)");
        print("Gradient is sufficiently small... Exiting");

    if(current_iteration >= 1000):
        print("Abort minimization maximum iteration is reached!");

    current_iteration += 1;

######################################## Plot results ####################################################################

#def plot_funct_with_Theta(a, b, x_size, y_size, x_theta, y_theta):
fkt.plot_funct_with_Theta(a, b, x_size, y_size,theta_history);    #Plot the result of the minimization with all intermidiate results

print("Old Objective Function = " + str(j_firstvalue));
print("New Objective Function = " + str(fkt.funct(a, b, new_theta[0], new_theta[1])));
print("Minimum Objective Function = " + str(fkt.funct(a, b, 1, 1)));
print("End");
