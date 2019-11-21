# import packages for scientific python and plotting
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_funct_with_Theta(a, b, x_size, y_size, theta):
    x_ax = np.linspace(-x_size,x_size);
    y_ax = np.linspace(-y_size,y_size);
    X,Y = np.meshgrid(x_ax,y_ax);
    banana_fkt = a**2 - 2*a*X + X**2 + b*Y**2 - 2*b*Y*X**2 + b*X**4;

    fig, ax = plt.subplots();
    cs = ax.contour(X,Y,banana_fkt, locator=plt.LogLocator());
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
    banana_fkt = a**2;
    banana_fkt -= 2*a*x;
    banana_fkt += x**2;
    banana_fkt += b*y**2;
    banana_fkt -= 2*b*y*x**2;
    banana_fkt += b*x**4;
    return banana_fkt;


def grad(a, b, x, y):
    banana_fkt = np.zeros((2,1));
    banana_fkt[0] = -2*a + 2*x - 4*b*y*x + 4*b*x**3; # x**2 = x^2
    banana_fkt[1] = 2*b*y - 2*b*x**2;
    return banana_fkt;

def hessian(a, b, x, y):
    banana_fkt = np.zeros((2,2));
    banana_fkt[0][0] = 12*b*x**2 - 4*b*y + 2;
    banana_fkt[1][0] = -4*b*x;
    banana_fkt[0][1] = -4*b*x;
    banana_fkt[1][1] = 2*b;
    return banana_fkt;

def neg_hessian(a, b, x, y):
    banana_fkt = np.zeros((2,2));
    determinant = -8*b**2*y+24*b**2*x**2+4*b-16*b**2*x**2;
    banana_fkt[0][0] = 2*b;
    banana_fkt[1][0] = 4*b*x;
    banana_fkt[0][1] = 4*b*x;
    banana_fkt[1][1] = 12*b*x**2 - 4*b*y + 2;
    banana_fkt[0][0] = banana_fkt[0][0] / determinant;
    banana_fkt[1][0] = banana_fkt[0][0] / determinant;
    banana_fkt[0][1] = banana_fkt[0][0] / determinant;
    banana_fkt[1][1] = banana_fkt[0][0] / determinant;
    return banana_fkt;
