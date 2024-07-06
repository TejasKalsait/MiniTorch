import math

# Data Structure to store Nodes

class Value:

     def __init__(self, data, _children = (), _op = '', label = '') -> None:
            self.data = data
            self._prev = set(_children)
            self.grad = 0
            self._backward = lambda : None
            self._op = _op
            self.label = label
    
     def __add__(self, other):
          
          other = other if isinstance(other, Value) else Value(other)
         
          out = Value(self.data + other.data, (self, other), '+')
         
          def _backward():
              self.grad += 1.0 * out.grad
              other.grad += 1.0 * out.grad

          out._backward = _backward
          return out
     
     def __sub__(self, other):
          return self + (-other)
     
     def __neg__(self):
          return self * -1

     def __mul__(self, other):
         
         other = other if isinstance(other, Value) else Value(other)

         out = Value(self.data * other.data, (self, other), '*')

         def _backward():
              self.grad += other.data * out.grad
              other.grad += self.data * out.grad
          
         out._backward = _backward
         return out
     
     def __truediv__(self, other):
          return self * other**-1
     
     def exp(self):

          x = self.data
          out = Value(math.exp(x), (self,), 'exp')

          def _backward():
               self.grad += out.data * out.grad
          out._backward = _backward

          return out
     
     def __pow__(self, other):

          assert isinstance(other, (int, float)), "Only int anf float supported"
          out = Value(self.data ** other, (self,), 'pow')

          def _backward():
               self.grad += (other * self.data ** (other - 1)) * out.grad

          out._backward = _backward
    
          return out
     
     def tanh(self):
         x = self.data
         t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
         out = Value(t, (self,), _op = 'tanh')

         def _backward():
              self.grad += (1 - t**2)  * out.grad
              #assert isinstance(self.grad, float)
         out._backward = _backward
         return out
     
     def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
     
     def backward(self):

          # Base case
          self.grad = 1.0

          # Creating a topological graph
          topo = []
          visited = set()

          def dfs(node):
               if node not in visited:
                    visited.add(node)
                    for child in node._prev:
                         dfs(child)
                    topo.append(node)

          try:
               dfs(self)
          except Exception as e:
               print(f"Error while finding Topological ordering {e}")

          # Finding gradients for each node
          try:
               for node in topo[::-1]:
                    node._backward()
          except Exception as e:
               print(f"Error while calculating gradients {e}")
     
     def __repr__(self) -> str:
        return f"Value(data={self.data})"
     
     def __rtruediv__(self, other):
          return other * self**-1
     
     def __rmul__(self, other):
          return self * other
     
     def __radd__(self, other):
          return self + other
     
     def __rsub__(self, other):
          return other + (-self)