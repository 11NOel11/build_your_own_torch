
class Value():
    def __init__(self,data:float,_prev=(),_op:str=""):
        self.data=float(data)
        self.grad=0.0
        self._prev=set(_prev)
        self._op=_op
        self._backward=lambda:None
        

    def __repr__(self):
        return f"Value(data:{self.data}, grad:{self.grad})"
    
    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data+other.data,(self,other),"+")
        def _backward():
            # dL/dself += dL/dout * dout/dself  = out.grad * 1

            self.grad+=out.grad
            other.grad+=out.grad

        out._backward=_backward
        return out 
    
    def __radd__(self,other):
        return self+other

     
    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out=Value(self.data*other.data,(self,other),"*")
        def _backward():
            # dL/dself += dL/dout * dout/dself  = out.grad * other

            self.grad+=out.grad*other.data
            other.grad+=out.grad*self.data

        out._backward=_backward

        return out
        

    def __rmul__(self,other):
        return self*other


    def backward(self):
        topo=[]
        visited=set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)
        build_topo(self)

        self.grad=1.0#cuz output or wherever backwards starts form d self/deslf will be 1.0

        for node in reversed(topo):
            node._backward()

    def __neg__(self):
        return (-1)*self
    def __sub__(self,other):
        return self +(-other)
    def __rsub__(self,other):
        return other + (-self)
    def __truediv__(self,other):
        return self * (other** -1) #our mult doesnt do raised to negative powers and we havent defined exponentials too so is this implementaion it or more fintune needd
    

    def __pow__(self,c):
        assert isinstance(c, (int, float)), "__pow__ only supports int/float powers"

        out = Value(self.data**c,(self,),"**")
        def _backward():
            #dL/dself=dL/dout*dout/dself=out.grad*(c*self.data**(c-1))
            self.grad+=out.grad*(c*(self.data**(c-1)))

            

        out._backward=_backward
        return out 
    def tanh(self):
        from math import tanh 
        
        out=Value(tanh(self.data),(self,),"tanh")
        def _backward():
            #dL/dself=same stuff as above
            self.grad+=out.grad*(1-(out.data)**2)
        out._backward=_backward
        return out

    def exp(self):
        from math import exp
        out = Value(exp(self.data),(self,),"exp")
        def _backward():
            self.grad+= out.grad*out.data
        out._backward=_backward
        return out
        

