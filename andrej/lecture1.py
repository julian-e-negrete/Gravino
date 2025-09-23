

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


def show_childs(node, visited=None, level=0):
    
    if visited is None:
        visited = set()
    if node in visited:
        return
    visited.add(node)

    indent = "  " * level
    print(f"{indent}Node: {node.label}, data: {node.data}, grad: {node.grad}, op: {node._op}")
    print(f"{indent}Child nodes (_prev): {[child.label for child in node._prev]}")
    
    for child in node._prev:
        show_childs(child, visited, level + 1)


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

    def tanh(self):
      x = self.data
      t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
      out = Value(t, (self, ), 'tanh')
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
    e = a * b
    e.label = 'e'   # <- unique
    d = e + c
    d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f
    L.label = 'L'

   
      
      
    L.grad = 1.0
    
    print("\n\n")
    show_childs(L)
    print("\n\n")
    print("start derivating to obtain the gradients")
        
    # f(x+h) - f(x)/h


    #L = d * f
    ### (d1 -d2) /h calculation of slope
    #dL/dd = ?
    #dL resulting of derivation of d 
    #dd original function
    #so the afecting of d to achieve L is f
    # then

    # (f*(x +h) - f(x)) /h
    # (f*d + f*h - f*d )/h
    # f*h / h
    # f
    d.grad = f.data
    print(f"dL/dd = {f}")



    #now lest use this but for f
    #L = d * f

    #fL /ff
    #what is afecting f to become L?
    # d is the afector so
    # (f(x+h) - f(x))/h 
    # (d(f+h) - d(f))/h
    # (d*f + d*h - d*f )/h
    # d*h/h
    # d

    f.grad = d.data
    print(f"fL/ff = {d}")
    
    #draw_dot(L)
    print("\n\n")
    show_childs(L)
    print("\n\n")
    
    
    print("CRUX PROPAGATION")
    
    
    # dL/dc

    # dL = f(x+h)
    # f = what modifies d to become L
    # dL = f(d + h)
    d.grad

    # how d is affected by e
    h = 0.00001
    h = Value(h);h.label = 'h'
    d1 = c + e
    d2 = c + (e + h)

    derivative =(d2.data - d1.data) / h.data
    e.grad = round(derivative, 10)


    # how d is affected by c
    h = 0.00001
    h = Value(h);h.label = 'h'
    d1 = c + e
    d2 = e + (c + h)



    derivative =(d2.data - d1.data) / h.data 

    c.grad = round(derivative, 10)

    #draw_dot(L)
    print("\n\n")
    show_childs(L)
    print("\n\n")

    
    
    #dL / dc = (dL / dd) * (dd / dc)
    # basically mutipling derivatives
    h = 0.00001
    h = Value(h);h.label = 'h'
    d1 = c + e
    d2 = e + (c + h)



    derivative =(d2.data - d1.data) / h.data 
    print(d.grad *  round(derivative, 10))
    c.grad = d.grad *  round(derivative, 10)


    d1 = c + e
    d2 = c + (e + h)



    derivative =(d2.data - d1.data) / h.data 

    e.grad = d.grad *  round(derivative, 10)

    print(d.grad) 
    
    
    print("\n\n")
    show_childs(L)
    print("\n\n")  
    
    
    #ed / eb = (ed / ee) * (ee / eb)

    h = 0.00001
    h = Value(h);h.label = 'h'
    d1 = a * b
    d2 = a * (b + h)
    derivative =(d2.data - d1.data) / h.data 

    b.grad = e.grad *  round(derivative, 10)

    print(derivative)


    #ed / ea = (ed / ee) * (ee / ea)

    d1 = b * a
    d2 = b * (a + h)

    derivative =(d2.data - d1.data) / h.data 

    a.grad = e.grad *  round(derivative, 10)

    a.grad = round(a.grad, 8)
    print(derivative)

    print("\n\n")
    show_childs(L)
    print("\n\n")  

    




    
    # inputs x1,x2
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')
    # weights w1,w2
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')
    # bias of the neuron
    b = Value(6.8813735870195432, label='b')
    # x1*w1 + x2*w2 + b
    x1w1 = x1*w1; x1w1.label = 'x1*w1'
    x2w2 = x2*w2; x2w2.label = 'x2*w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
    n = x1w1x2w2 + b; n.label = 'n'
    o = n.tanh(); o.label = 'o'
    
    
    print("\n\n")
    show_childs(o)
    print("\n\n")  
    
    
    print("developing form omarchy")