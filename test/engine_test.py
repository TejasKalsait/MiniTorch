import sys
sys.path.append('../')

from mini_torch.engine import Value
from mini_torch.network import MLP

def test_value():

    print("Testing Value Class...")

    # Forward Pass
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')

    w1 = Value(-3.0, label='w1')
    w2 = Value(-3.0, label='w2')
    b = Value(4.7, label='b')

    x1w1 = x1 * w1 ; x1w1.label = 'x1w1'
    x2w2 = x2 * w2 ; x2w2.label = 'x2w2'

    x1w1x2w2 = x1w1 + x2w2 ; x1w1x2w2.label = 'x1w1x2w2'

    n = x1w1x2w2 + b ; n.label = 'n'
    o = n.tanh() ; o.label = 'o'

    # Backward Pass
    o.backward()

    print(f"Printing the first grad value -> {w1.grad}")
    print("-----------------------------------------------------------------------------------------")

def test_mlp():

    print("Testing MLP Class...")

    # DATA
    xs = [[2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]]

    ys = [1.0, -1.0, -1.0, 1.0]

    network = MLP(3, [4, 4, 1])

    # FORWARD
    ypred = []
    for x in xs:
        ypred.append(network(x))
    loss = sum([(ygt-ypre)**2  for ygt, ypre in zip(ys, ypred)]) / len(ys)

    # ZERO GRADIENT
    for p in network.parameters():
        p.grad = 0

    # BACKPROPOGATION
    loss.backward()

    # UPDATE WEIGHTS
    for p in network.parameters():
        p.data -= 0.1 * p.grad

    print(f"Printing the first grad value -> {network.layers[0].neurons[0].w[0].grad}")
    print("-----------------------------------------------------------------------------------------")

def main():

    test_value()
    test_mlp()

    return

if __name__ == '__main__':
    main()