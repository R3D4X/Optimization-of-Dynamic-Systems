# import packages for scientific python and plotting
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#J=a*x^2-b*x*y+a*y^2;
#grad=[a*x-b*y  a*y-b*x];
#H=[a -b; -b a];

def plot_funct_with_Theta(a, b, x_size, y_size, theta):
    x_ax = np.linspace(-x_size,x_size);
    y_ax = np.linspace(-y_size,y_size);
    X,Y = np.meshgrid(x_ax,y_ax);
    quadratic_fkt = a*X**2 - b*X*Y + a*Y**2;

    fig, ax = plt.subplots();
    cs = ax.contour(X,Y,quadratic_fkt, locator=plt.LogLocator());
    fmt = ticker.LogFormatterMathtext();
    fmt.create_dummy_axis();
    ax.clabel(cs, cs.levels, fmt=fmt);
    ax.set_title("Bananafunction Min at [1,1]");
    for a in range(theta[0].size - 1): #Plot all thetas inclusive the intermidiate results
        if(theta[0][a] != 0 or theta[1][a] != 0): #Only plot calculated values (instance created via np.zeros and probability for (0,0) is near zero...)
            plt.plot(theta[0][a],theta[1][a],'ro');
    #minima=[1 1];
    plt.plot(1,1,'rx');

    plt.show();

def funct(a, b, x, y):
    quadratic_fkt = a*x**2;
    quadratic_fkt -= b*x*y;
    quadratic_fkt += a*y**2;
    return quadratic_fkt;


def grad(a, b, x, y):
    quadratic_fkt = np.zeros((2,1));
    quadratic_fkt[0] = a*x - b*y; # x**2 = x^2
    quadratic_fkt[1] = a*y - b*x;
    return quadratic_fkt;

def hessian(a, b, x, y):
    quadratic_fkt = np.zeros((2,2));
    quadratic_fkt[0][0] = a;
    quadratic_fkt[1][0] = -b;
    quadratic_fkt[0][1] = -b;
    quadratic_fkt[1][1] = a;
    return quadratic_fkt;

def neg_hessian(a, b, x, y):
    quadratic_fkt = np.zeros((2,2));
    determinant = a**2 - b**2;
    quadratic_fkt[0][0] = a;
    quadratic_fkt[1][0] = b;
    quadratic_fkt[0][1] = b;
    quadratic_fkt[1][1] = a;
    quadratic_fkt[0][0] = quadratic_fkt[0][0] / determinant;
    quadratic_fkt[1][0] = quadratic_fkt[0][0] / determinant;
    quadratic_fkt[0][1] = quadratic_fkt[0][0] / determinant;
    quadratic_fkt[1][1] = quadratic_fkt[0][0] / determinant;
    return quadratic_fkt;
