import random
from typing import Any

from mini_torch.engine import Value

class Neuron:

    def __init__(self,nin) -> None:

        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, -1))

    def parameters(self):
        return self.w + [self.b]
    
    def __call__(self, *args: Any, **kwds: Any) -> Value:
        # w * x + b
        try:
            x = args[0]
        except Exception as e:
            print(f"Nothing passed in the neuron. {e}")

        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        
        return out

class Layer:

    def __init__(self, nin, nout) -> None:
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

        # params = []

        # for neuron in self.neurons:
        #     params.extend(neuron.parameters())

        # return params

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        
        try:
            x = args[0]
        except Exception as e:
            print(f"Nothing passed in the neuron. {e}")

        outs = [n(x) for n in self.neurons]

        if len(outs) == 1:
            return outs[0]
        else:
            return outs
    

class MLP:

    def __init__(self, nin, nouts) -> None:
        
        # nin -> Number of inputs in the MLP
        # nouts -> List of Neurons in each layer

        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        
        try:
            x = args[0]
        except Exception as e:
            print(f"Nothing passed in the neuron. {e}")

        for layer in self.layers:
            x = layer(x)
        
        return x