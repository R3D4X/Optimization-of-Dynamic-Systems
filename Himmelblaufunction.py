# import packages for scientific python and plotting
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#J= (x**2 +y -11)**2 + (x+ y**2 - 7)**2;
#grad=[a*x-b*y  a*y-b*x];
#H=[a -b; -b a];

def plot_funct_with_Theta(a, b, x_size, y_size, theta):
    x_ax = np.linspace(-x_size,x_size);
    y_ax = np.linspace(-y_size,y_size);
    X,Y = np.meshgrid(x_ax,y_ax);

    himmelblau_fkt = (X**2 + Y -11)**2 + (X + Y**2 - 7)**2;

    fig, ax = plt.subplots();
    cs = ax.contour(X,Y,himmelblau_fkt, locator=plt.LogLocator());
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
    himmelblau_fkt = (x**2 + y -11)**2 
    himmelblau_fkt += (x + y**2 - 7)**2;
    return himmelblau_fkt;


def grad(a, b, x, y):
    himmelblau_fkt = np.zeros((2,1));
    himmelblau_fkt[0] = 4*x**3 + 4*x*y - 42*x + 2*y**2 - 14; # x**2 = x^2
    himmelblau_fkt[1] = 2*x**2 - 26*y - 22 + 4*x*y + 4*y**3;
    return himmelblau_fkt;

def hessian(a, b, x, y):
    himmelblau_fkt = np.zeros((2,2));
    himmelblau_fkt[0][0] = 12*x**2 + 4*y -42;
    himmelblau_fkt[1][0] = 4*x + 4*y;
    himmelblau_fkt[0][1] = 4*x + 4*y;
    himmelblau_fkt[1][1] = 4*x + 12*y**2 - 26;
    return himmelblau_fkt;

def neg_hessian(a, b, x, y):
    himmelblau_fkt = np.zeros((2,2));
    determinant = (12*x**2 + 4*y -42);
    determinant *= (4*x + 12*y**2 - 26);
    determinant -= (4*x + 4*y)**2;
    himmelblau_fkt[0][0] = 4*x + 12*y**2 - 26;
    himmelblau_fkt[1][0] = -4*x - 4*y;
    himmelblau_fkt[0][1] = -4*x - 4*y;
    himmelblau_fkt[1][1] = 12*x**2 + 4*y -42;
    himmelblau_fkt[0][0] = himmelblau_fkt[0][0] / determinant;
    himmelblau_fkt[1][0] = himmelblau_fkt[0][0] / determinant;
    himmelblau_fkt[0][1] = himmelblau_fkt[0][0] / determinant;
    himmelblau_fkt[1][1] = himmelblau_fkt[0][0] / determinant;
    return himmelblau_fkt;
