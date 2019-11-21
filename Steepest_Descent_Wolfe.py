# import packages for scientific python and plotting
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#import Bananafunction as fkt
#import Quadraticfunction as fkt
import Himmelblaufunction as fkt

import time

def test_wolfe(c_1, c_2, banana_a, banana_b, theta, a):
    result = -1; # -1 and 0 -> Not fullfilled 
                 #  1 -> Fullfilled
    #Steepest descend direction
    p_sd = -1 * fkt.grad(banana_a, banana_b, theta[0], theta[1]);
    new_theta = [0,0];          #Thetabuffer
    new_theta[0] = theta[0] + a * p_sd[0];
    new_theta[1] = theta[1] + a * p_sd[1];
    print("Test Wolfe: Theta = [ " + str(theta[0]) + " , " + str(theta[1]) + "]");
    print("Test Wolfe: New Theta = [ " + str(new_theta[0]) + " , " + str(new_theta[1]) + "]");
    print("Test Wolfe: p_sd = [ " + str(p_sd[0]) + " , " + str(p_sd[1]) + "]");
    print("Test Wolfe: a = " + str(a));

    phi_ak = fkt.funct(banana_a, banana_b, new_theta[0], new_theta[1]);  #Objective Function of new Theta

    grad = fkt.grad(banana_a, banana_b, theta[0], theta[1]);  #Current gradient
    print("Test Wolfe: grad = [ " + str(grad[0]) + " , " + str(grad[1]) + "]");

    second_tailor_approx = fkt.funct(banana_a, banana_b, theta[0], theta[1]);  # Current Objective Funct + c_1 * current gradiend(transposed) * steepest_descent_direction * a -> 2 Tailor approx
    second_tailor_approx += c_1 * (grad[0]*p_sd[0] + grad[1]*p_sd[1]) * a;
    print("Test Wolfe 1: " + str(phi_ak) + " <= " + str(second_tailor_approx));
    if(phi_ak <= second_tailor_approx):    #Check for Wolfe criterion No.1
        result = 0; #Step 1 archived
        
        new_grad = fkt.grad(banana_a, banana_b, new_theta[0], new_theta[1]); #Calculate new gradient
        new_grad_direct = new_grad[0]*p_sd[0] + new_grad[1]*p_sd[1]; #First term of wolfe 2
        print("Test Wolfe: new_grad = [ " + str(new_grad[0]) + " , " + str(new_grad[1]) + "]");
        approx_grad_direct = c_2 * (grad[0]*p_sd[0] + grad[1]*p_sd[1]); #Second term of wolfe 2
        print("Test Wolfe 2: " + str(new_grad_direct) + " >= " + str(approx_grad_direct));
        if(new_grad_direct >= approx_grad_direct): #Check for Wolfe criterion No.2
            result = 1; #Set result to success
    #time.sleep(1);
    return result; #Return result


banana_a = 1;
banana_b = 100;
x_size = 10;
y_size = 20; 

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
fkt.plot_funct_with_Theta(banana_a, banana_b, x_size, y_size,theta_history);    #Plot the function and our starting point

print("Theta = [ " + str(theta[0]) + " , " + str(theta[1]) + "]");

grad = fkt.grad(banana_a, banana_b, theta[0], theta[1]);
print("Gradient bananafkt = [ " + str(grad[0]) + " , " + str(grad[1]) + " ]");

j_firstvalue = fkt.funct(banana_a, banana_b, theta[0], theta[1]);
print("Value of objective fkt at the beginning = " + str(j_firstvalue));

old_grad = [0,0];

current_iteration = 0;

while( ((np.abs(grad[0]) > 10e-3) or (np.abs(grad[1]) > 10e-3)) and 
        current_iteration <= 1000):

############################### Calculate Direction ####################################################################

    #Steepest descend direction
    p_sd = -1 * fkt.grad(banana_a, banana_b, theta[0], theta[1]);
    #def grad(a, b, x, y)

    #print("p_sd = [ " + str(p_sd[0]) + " , " + str(p_sd[1]) + " ]");

############################### 1 Dimensional Search  ###################################################################

    a = 1;          #Steplength at the beginning
    c_1 = 0.5;        #Wolfe factor No.1 c_1 is [0,1]
    c_2 = 0.7;        #Wolfe factor No.2 c_2 is [c_1,1]
    stop = -1;      #Boolean for poor people
    wolfe_watchdog = 0; #Counts iteration and breaks
    new_theta = [0,0];          #Thetabuffer
    new_theta[0] = theta[0];
    new_theta[1] = theta[1];

    while (stop != 1):
        #def test_wolfe(c_1, c_2, banana_a, banana_b, theta, a):
        result =  test_wolfe(c_1, c_2, banana_a, banana_b, theta, a);   #Check Wolfe criterion
        wolfe_watchdog += 1;
        if( (result == 1) and (wolfe_watchdog < 550) ):    #If Wolfe is valid
            stop = 1;
        elif( (result == 0) and (wolfe_watchdog >= 550) ): #If we can't find an a with wolfe 1 and 2 we accept wolfe 1 points
            stop = 1;
        else: #We havent found a valid point so change a
            a = a*0.9;    #first value that will be processed will be a_min * 2
            if(a < 10e-12):
                a = 10;
        if(wolfe_watchdog > 1100):
            print("Error couldn't find suitable a... Quitting...");
            quit();

    print("Result: a = " + str(a));
############################### Calculate New Iterate ####################################################################
#
#        new_theta[0] = theta[0] + a * p_sd[0];
#        new_theta[0] = theta[0] + a * p_sd[0];
#
    new_theta[1] = theta[1] + a * p_sd[1];
    new_theta[1] = theta[1] + a * p_sd[1];
#
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

    j_old = fkt.funct(banana_a, banana_b, theta[0], theta[1]);

    theta[0] = new_theta[0];    #Store the new theta as current theta
    theta[1] = new_theta[1];

    old_grad[0] = grad[0];      #Store the current gradient as old gradient
    old_grad[1] = grad[1];
    grad = fkt.grad(banana_a, banana_b, theta[0], theta[1]);  #Calculate the current gradient

################################### Debug ##########################################################
#    print("Gradient bananafkt = [ " + str(grad[0]) + " , " + str(grad[1]) + " ]");
#    print("Old gradient bananafkt = [ " + str(old_grad[0]) + " , " + str(old_grad[1]) + " ]");
    print("New Theta = [ " + str(theta[0]) + " , " + str(theta[1]) + "]");
    print("Old Objective Function = " + str(j_old));
    print("New Objective Function = " + str(fkt.funct(banana_a, banana_b, new_theta[0], new_theta[1])));
####################################################################################################

    if((np.abs(grad[0]) < 10e-3) or (np.abs(grad[1]) < 10e-3)): #Check wether the gradient is sufficiently small (Minima)
        print(str(np.abs(grad[0])) + " > " + str(10e-3) + " and " + str(np.abs(grad[1])) + " > 10e-3)");
        print("Gradient is sufficiently small... Exiting");

    if(current_iteration >= 1000):
        print("Abort minimization maximum iteration is reached!");

    current_iteration += 1;

######################################## Plot results ####################################################################

#def plot_funct_with_Theta(a, b, x_size, y_size, x_theta, y_theta):
fkt.plot_funct_with_Theta(banana_a, banana_b, x_size, y_size,theta_history);    #Plot the result of the minimization with all intermidiate results

print("Old Objective Function = " + str(j_firstvalue));
print("New Objective Function = " + str(fkt.funct(banana_a, banana_b, new_theta[0], new_theta[1])));
print("Minimum Objective Function = " + str(fkt.funct(banana_a, banana_b, 1, 1)));
print("End");
