- MicroGrad is an autograd (automatic gradient) engine

## Derivaties

- variable with positive derivative means slightly increasing this will increase the final loss
- variable with negative derivative with loss means increasing this slightly will decrease the loss.

- Therefore, to decrease the loss,

- 1) Learning rate times negative derivate step basically means decreasing by a nudge. Negative derivative and further drease it will increase the loss. So we need to bump this value up

- 2) Learning rate times positive derivate step basically means increasing by a nudge. Positive derivative and further inreasing will increase the loss, so actually decrease this value

SEE HOW YOU NEED A MINUS IN THE UPDATE STEP

new_weight = old_weight - (learning rate * derivation)


## Class Special functions

1) __repr__ -> returns a string representation of the object.

2) __add__ -> returns a new object which defines addition operation.

c = a + b
then object a's __add__ is only called with `self` as a and `other` as b


## Value object parameters

1) data -> value it represents
2) -prev = tuple of children
3) grad = variable to store the gradient (no required while init)
4) _op = string operation which resulted in this value ('+', '-', '*', '**') (not required while init)
5) _label = string unique identifier


## Local Derivatives

1) Base Case
Derivative of loss with respect to loss will always be 1.0 (Linear change)

2) Multiplication
If c = a*b
local derivative of c wrt a is always value of b
and local derivative of c wrt b is always the value of a

3) Addition (Just passes on gradients)
if c = a + b
local derivative of c wrt a is 1.0 &
local derivative of c wrt b is 1.0
Therefore a plus just allows the gradients to pass locally

4) Tanh
if o = tanh(n), then derivative of o wrt n is 1 - o**2

## Neuron

