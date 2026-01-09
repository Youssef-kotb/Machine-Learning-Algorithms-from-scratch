import numpy as np



class HingeLoss:
    def __init__(self):
        pass

    def forward(self, y ,scores):
        return np.maximum(0,1-(y*scores))
    
    def backward(self, y, scores):
        
        grad = np.zeros_like(scores)
        mask = y * scores < 1
        grad[mask] = -y[mask]
        return grad
