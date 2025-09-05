

#always this imports
import math
import numpy as np
import matplotlib.pyplot as plt


from rich import print


#from micrograd.engine import Value # type: ignore



def f(x):
    return 3*x**2 - 4*x + 5


from graphviz import Digraph # type: ignore
import tempfile
import os
import subprocess




class Value:
    def __init__ (self, data, _children=(), _op = '', label = '' ):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
        self.label = label

    # print nicer representation of data
    def __repr__(self):
        return f"Value(data={self.data})"   
    
    
    def __add__(self, other):
        #other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other),  '+')
        return out
        
        
    def __mul__(self, other):
        #other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data , (self, other), '*')
        return out

# visualization of neural net graphs
def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root, view=True):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    # Create a temporary file
  with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as f:
    filename = f.name

  dot.render(filename=filename, cleanup=True, view=view)
    
  return dot

if __name__ == "__main__":
    
        
    # a = Value(-4.0)
    # b = Value(2.0)

    # c = a + b
    # d = a * b + b**3

    # c += c + 1
    # c += 1 + c + (-a)

    # d += d * 2 + (b + a).relu()
    # d += 3 * d + (b - a).relu()

    # e = c - d
    # f = e**2
    # g = f / 2.0
    # g += 10.0 / f

    # print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
    # g.backward()
    # print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
    # print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db



    # xs = np.arange(-5.0, 5.0, 0.25)
    # ys = f(xs)
    # plt.plot(xs, ys)

    #derivative
    print(r"The derivative as a limit is: $f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$")
    
    h = 0.0000001
    x = 3.0
    
    # calculation of slope
    print( (f(x + h) - f(x))/h )
    
    
    #lets get more complex
    # inputs
    a= 2.0
    b = -3.0
    c = 10.0
    d = a * b + c
    print(d)
    #Devrivatives of d with respect to a,b,c
    
    # multiplies to the negative modifier of the expresion
    h = 0.0001
    d1 = a * b + c
    a += h
    d2 = a *b + c
    print('d1', d1)
    print('d2', d2)
    
    print('slope ' ,(d2 - d1 )/h )


    # reduces the negative modifier of the expresion
    a= 2.0
    b = -3.0
    c = 10.0
    d = a * b + c

    b += h
    d2 = a *b + c
    print('d1', d1)
    print('d2', d2)
    
    print('slope ' ,(d2 - d1 )/h )

    # add to the positive modifier of the expresion
    a= 2.0
    b = -3.0
    c = 10.0
    d = a * b + c

    c += h
    d2 = a *b + c
    print('d1', d1)
    print('d2', d2)
    
    print('slope ' ,(d2 - d1 )/h )



      
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b; e.label = 'c'
    d = e + c; d.label = 'd' 
    f = Value(-2.0, label='f')
    L = d *f; L.label='L'
    
    print(L._prev)
    print(L._op) 
    
    for child in L._prev:
      print("child: ", child)
      print("previos scalars: ", child._prev)
      print("operation: ", child._op)
    #draw_dot(L)
