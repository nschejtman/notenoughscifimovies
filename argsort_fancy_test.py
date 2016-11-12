Python 2.7.12 (v2.7.12:d33e0cf91556, Jun 27 2016, 15:24:40) [MSC v.1500 64 bit (AMD64)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> import numpy as np
>>> a = np.array([1, 2, 3])
>>> a
array([1, 2, 3])
>>> a + 3
array([4, 5, 6])
>>> a / a
array([1, 1, 1])
>>> a / (a + 2)
array([0, 0, 0])
>>> a
array([1, 2, 3])
>>> a*a
array([1, 4, 9])
>>>  a*=a
 
  File "<pyshell#8>", line 2
    a*=a
    ^
IndentationError: unexpected indent
>>> a*=1
>>> a*=a
>>> a
array([1, 4, 9])
>>> a = [a,a]
>>> a
[array([1, 4, 9]), array([1, 4, 9])]
>>> a = [[1,2,3],[4,5,6]]
>>> a = np.array(a)
>>> a
array([[1, 2, 3],
       [4, 5, 6]])
>>> a[[0],[[0,2]]]
array([[1, 3]])
>>> a = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12])
	     
SyntaxError: invalid syntax
>>> a = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
>>> a
array([[ 1,  2,  3],
       [ 4,  5,  6],
       [ 7,  8,  9],
       [10, 11, 12]])
>>> a[[0,2],[[0,1],[1,2]]]
array([[1, 8],
       [2, 9]])
>>> a[[0,2],[[0],[1]]]
array([[1, 7],
       [2, 8]])
>>> a[[[0,2,1,3]],[0,1,2,3]]

Traceback (most recent call last):
  File "<pyshell#23>", line 1, in <module>
    a[[[0,2,1,3]],[0,1,2,3]]
IndexError: index 3 is out of bounds for axis 1 with size 3
>>> a[[[0,2,1]],[0,1,2]]
array([[1, 8, 6]])
>>> a[[[0,2,1],[2,3,3]],[0,1,2]]
array([[ 1,  8,  6],
       [ 7, 11, 12]])
>>> np.argsort(axis=0)

Traceback (most recent call last):
  File "<pyshell#26>", line 1, in <module>
    np.argsort(axis=0)
TypeError: argsort() takes at least 1 argument (1 given)
>>> np.argsort(a, axis=0)
array([[0, 0, 0],
       [1, 1, 1],
       [2, 2, 2],
       [3, 3, 3]], dtype=int64)
>>> b = np.rand((4,3))

Traceback (most recent call last):
  File "<pyshell#28>", line 1, in <module>
    b = np.rand((4,3))
AttributeError: 'module' object has no attribute 'rand'
>>> b = np.random(4,3)

Traceback (most recent call last):
  File "<pyshell#29>", line 1, in <module>
    b = np.random(4,3)
TypeError: 'module' object is not callable
>>> b = np.random.rand(4,3)
>>> b
array([[ 0.81400232,  0.04425077,  0.28838263],
       [ 0.63234452,  0.20659601,  0.6795355 ],
       [ 0.97647556,  0.52617665,  0.91530969],
       [ 0.82512578,  0.42831585,  0.58863367]])
>>> b*10
array([[ 8.14002321,  0.44250769,  2.88382631],
       [ 6.32344517,  2.06596007,  6.79535498],
       [ 9.76475563,  5.26176648,  9.15309687],
       [ 8.25125777,  4.28315852,  5.88633675]])
>>> b = int(b*10)

Traceback (most recent call last):
  File "<pyshell#33>", line 1, in <module>
    b = int(b*10)
TypeError: only length-1 arrays can be converted to Python scalars
>>> b = (b*10).map(int)

Traceback (most recent call last):
  File "<pyshell#34>", line 1, in <module>
    b = (b*10).map(int)
AttributeError: 'numpy.ndarray' object has no attribute 'map'
>>> map(b*10,int)

Traceback (most recent call last):
  File "<pyshell#35>", line 1, in <module>
    map(b*10,int)
TypeError: argument 2 to map() must support iteration
>>> map(int,b*10)

Traceback (most recent call last):
  File "<pyshell#36>", line 1, in <module>
    map(int,b*10)
TypeError: only length-1 arrays can be converted to Python scalars
>>> b = np.array([[4,5,2],[11,10,8],[3,7,6],[0,1,9]])
>>> b
array([[ 4,  5,  2],
       [11, 10,  8],
       [ 3,  7,  6],
       [ 0,  1,  9]])
>>> np.argsort(b,axis=0)
array([[3, 3, 0],
       [2, 0, 2],
       [0, 2, 1],
       [1, 1, 3]], dtype=int64)
>>> np.argsort(b,axis=1)
array([[2, 0, 1],
       [2, 1, 0],
       [0, 2, 1],
       [0, 1, 2]], dtype=int64)
>>> 
