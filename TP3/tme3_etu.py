from arftools import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def mse(datax,datay,w):
    """ retourne la moyenne de l'erreur aux moindres carres """

    datax, datay = datax.reshape(len(datay),-1), datay.reshape(-1,1)

    return np.square(np.subtract(datay,datax * w)).mean()

def mse_g(datax,datay,w):

    """ retourne le gradient moyen de l'erreur au moindres carres """

    datax, datay = datax.reshape(len(datay),-1),datay.reshape(-1,1)
    minustwos = -2 * np.ones(len(datay))

    M = datay - np.dot(datax, w.T)

    #return (-2/np.shape(datax)[0]) * np.sum(np.dot(datax.T, M))
    return np.dot(minustwos,np.subtract(datay,datax * w.T)).mean()

def hinge(datax,datay,w):
    """ retourne la moyenne de l'erreur hinge """
    if len(datax.shape)==1: datax = datax.reshape(1,-1)
    #datax, datay = datax.reshape(len(datay), -1), datay.reshape(-1, 1)
    return np.mean(np.maximum(0, -datay*np.dot(datax, w)))
    #return np.maximum(0, -datay * datax.dot(w)).mean()

def hinge_g(datax,datay,w):
    """ retourne le gradient moyen de l'erreur hinge """
    datax, datay, w = datax.reshape(len(datay),-1), datay.reshape(-1,1), w.reshape(-1,1)

    return np.maximum(np.zeros(len(datay)),np.ones(len(datay))-(datay * (datax.dot(w)))).mean()


class Lineaire(object):
    def __init__(self,loss=hinge,loss_g=hinge_g,max_iter=1000,eps=0.01):
        """ :loss: fonction de cout
            :loss_g: gradient de la fonction de cout
            :max_iter: nombre d'iterations
            :eps: pas de gradient
        """
        self.max_iter, self.eps = max_iter,eps
        self.loss, self.loss_g = loss, loss_g

    def fit(self,datax,datay,testx=None,testy=None):
        """ :datax: donnees de train
            :datay: label de train
            :testx: donnees de test
            :testy: label de test
        """
        # on transforme datay en vecteur colonne
        datay = datay.reshape(-1,1)
        N = len(datay)
        datax = datax.reshape(N,-1)
        D = datax.shape[1]
        self.w = np.random.random((1,D))
        pass

    def predict(self,datax):
        if len(datax.shape)==1:
            datax = datax.reshape(1,-1)
        return np.sign(np.dot(datax, self.w.T)).reshape(-1)

    def score(self,datax,datay):
        """retourne le score moyen par prédiction"""
        return np.mean(self.predict(datax) != datay)



def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")



def plot_error(datax,datay,f,step=10):
    grid,x1list,x2list=make_grid(xmin=-4,xmax=4,ymin=-4,ymax=4)
    plt.contourf(x1list,x2list,np.array([f(datax,datay,w) for w in grid]).reshape(x1list.shape),25)
    plt.colorbar()
    plt.show()



if __name__=="__main__":
    """ Tracer des isocourbes de l'erreur """
    plt.ion()
    trainx,trainy =  gen_arti(nbex=1000,data_type=0,epsilon=1)
    testx,testy =  gen_arti(nbex=1000,data_type=0,epsilon=1)
    plt.figure()
    plot_error(trainx,trainy,mse)
    plt.figure()
    plot_error(trainx,trainy,hinge)
    perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.1)
    perceptron.fit(trainx,trainy)
    print("Erreur : train %f, test %f"% (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
    plt.figure()
    plot_frontiere(trainx,perceptron.predict,200)
    plot_data(trainx,trainy)
