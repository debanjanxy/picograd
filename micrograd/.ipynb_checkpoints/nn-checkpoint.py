import random
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
    

class Neuron(Module):
    def __init__(self, nin):
        self.weights = [Value(random.uniform(-1, 1)) for i in range(nin)]
        self.bias = Value(random.uniform(-1, 1))
        
    def __call__(self, x):
        act = sum([wi * xi for wi, xi in zip(self.weights, x)], self.bias)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.weights + [self.bias]
    

class Layer(Module):
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
        
    def __call__(self, inputs):
        outs = [neuron(inputs) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):
    def __init__(self, nin, nouts):
        size = [nin] + nouts
        self.layers = [Layer(size[i], size[i + 1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]