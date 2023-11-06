import numpy as np
from scipy.spatial.distance import jensenshannon, cosine, euclidean

def euclDis(u:np.array, v:np.array):

    return euclidean(u, v)

def l1norm(u:np.array, v:np.array):
    return np.linalg.norm(u,v, ord=1)

def chi2_distance(u:np.array, v:np.array):
    chi = 0.5 * np.sum(((u - v) ** 2) / (u + v+1e-8))
    return chi
def hellkdis(u:np.array, v:np.array):
    distance = np.sum(np.sqrt(u * v))
    return distance

def histogram_intersection(u:np.array, v:np.array):
    return np.sum(np.minimum(u, v))

def jensensim(v:np.array, u:np.array):
    return 1-jensenshannon(v, u)

def cos_sim(v:np.array, u:np.array):
    return 1- abs(cosine(v,u))