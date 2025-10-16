---
jupyter:
  jupytext:
    default_lexer: ipython3
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region -->
# A Quick Introduction to Python

**Prepared for the Bank of Portugal Computational Economics Course (Oct 2025)**

**Author:** [John Stachurski](https://johnstachurski.net)



This notebook provides a super quick introduction to Python.

Participants who don't need it can either ask more advanced questions, work, or sleep.

Participants who want a slower treatment can either

1. slow the class down by asking lots of questions, or
1. review the first few [QuantEcon Python programming lectures](https://python-programming.quantecon.org/intro.html) after the class
<!-- #endregion -->

<!-- #region -->
## Example Task: Plotting a White Noise Process

Task: simulate and plot the white noise process $ \epsilon_0, \epsilon_1,
\ldots, \epsilon_T $, where each draw $ \epsilon_t $ is independent standard
normal.


### Version 1

Here are a few lines of code that perform the task we set
<!-- #endregion -->

```python
import numpy as np
import matplotlib.pyplot as plt   

fig, ax = plt.subplots()
ϵ_values = np.random.randn(100)   # 100 draws from N(0, 1)
ax.plot(ϵ_values)                # Plot draws
```

Let’s discuss some aspects of this program.


#### Imports

The first two lines import functionality from external code libraries:

```python
import numpy as np
import matplotlib.pyplot as plt 
```

<!-- #region -->
The first line imports [NumPy](https://python-programming.quantecon.org/numpy.html), a Python package for tasks like

- working with arrays (vectors and matrices)  
- common mathematical functions like `cos` and `sqrt`  
- generating random numbers  
- linear algebra, etc.  


After `import numpy as np` we have access to these attributes via the syntax `np.attribute`.

Here’s two more examples
<!-- #endregion -->

```python
np.sqrt(4)
```

```python
np.log(4)
```

#### Why So Many Imports?

When you want to do something interesting with Python, you almost always need
to import additional functionality.

Why is that?


#### Importing Names Directly

Recall this code that we saw above

```python
import numpy as np
np.sqrt(4)
```

Here’s another way to access NumPy’s square root function

```python
from numpy import sqrt
sqrt(4)
```

### A Version with a For Loop

Here’s a (less efficient) version that illustrates `for` loops and Python lists.

```python
ts_length = 100
ϵ_values = []       # Empty list

for i in range(ts_length):
    e = np.random.randn()
    ϵ_values.append(e)
    
fig, ax = plt.subplots()
ax.plot(ϵ_values)
```

How does it work?

How do you like significant whitespace??

<!-- #region -->
#### Lists


Consider the statement `ϵ_values = []`, which creates an empty list.

Lists are a native Python data structure used to group a collection of objects.

Here's another:
<!-- #endregion -->

```python
x = [10, 'foo', False]
type(x)
```

When adding a value to a list, we can use the syntax `list_name.append(some_value)`

```python
x.append(2.5)
x
```

Here `append()` is what’s called a **method**, which is a function "attached to" an object -- in this case, the list `x`.

- Python objects such as lists and strings all have methods that can
  manipulate data contained in the object.  
- String objects have string methods, list objects have list methods, etc.

Another useful list method is `pop()`

```python
x
```

```python
x.pop()
```

```python
x
```

<!-- #region -->
How can I get a list of all methods?


### Indices

Lists in Python are zero-based (as in C, Java or Go), so the first element is referenced by `x[0]`
<!-- #endregion -->

```python
x[0]   # First element of x
```

```python
x[1]   # Second element of x
```

<!-- #region -->
Who likes zero based lists/arrays?


To get the last element use?
<!-- #endregion -->

<!-- #region -->
### While Loops


For the purpose of illustration, let’s modify our program to use a `while` loop instead of a `for` loop.
<!-- #endregion -->

```python
ts_length = 100
ϵ_values = []
i = 0
while i < ts_length:
    e = np.random.randn()
    ϵ_values.append(e)
    i = i + 1             # Equivalent: i += 1

fig, ax = plt.subplots()
ax.plot(ϵ_values)
```

How does it work?


**Exercise**

Plot the balance of a bank account over $0, \ldots, T$ when $T=50$.

* There are no withdraws 
* The initial balance is $ b_0 = 10 $ and the interest rate is $ r = 0.025$.

The balance updates from period $ t $ to $ t+1 $ according to $ b_{t+1} = (1 + r) b_t $.

Your task is to generate and plot the sequence $b_0, b_1, \ldots, b_T $.

You can use a Python list to store this sequence, or a NumPy array.

In the first case, start with

```python
T = 50
b = []
```

In the second case, you can use a statement such as

```python
T = 50
b = np.empty(T+1)   # Allocate memory to store all b_t
```

and then populate `b` in a for loop.

```python
# Put your solution here
```

```python
for _ in range(20):
    print("Solution below!")
```

Here's the list-based solution

```python
b = []
x = 10         # initial balance
r = 0.025         # interest rate
T = 50            # end date
for t in range(T):
    b.append(x)
    x = (1 + r) *x
b.append(x)

fig, ax = plt.subplots()
ax.plot(b, label='bank balance')
ax.legend()
plt.show()
```

And here's the NumPy array-based solution.

```python
b = np.empty(T+1) # an empty NumPy array, to store all b_t
b[0] = 10         # initial balance
for t in range(T):
    b[t+1] = (1 + r) * b[t]

fig, ax = plt.subplots()
ax.plot(b, label='bank balance')
ax.legend()
```

**Exercise**

Simulate and plot the correlated time series

$$
    x_{t+1} = \alpha \, x_t + \epsilon_{t+1}
    \quad \text{where} \quad
    x_0 = 0
    \quad \text{and} \quad t = 0,\ldots,T
$$

were $ \{\epsilon_t\} $ is IID and standard normal.

In your solution, restrict your import statements to

```python
import numpy as np
import matplotlib.pyplot as plt
```

Set $ T=200 $ and $ \alpha = 0.9 $.

```python
# Put your code here
```

```python
for i in range(20):
    print("Solution below.")
```

Here’s one solution.

```python
α = 0.9
T = 200
x = np.empty(T+1)
x[0] = 0

for t in range(T):
    x[t+1] = α * x[t] + np.random.randn()

fig, ax = plt.subplots()
ax.plot(x)
```

**Exercise** 

Plot three simulated time series,
one for each of the cases $ \alpha=0 $, $ \alpha=0.8 $ and $ \alpha=0.98 $.

Use a `for` loop to step through the $ \alpha $ values.

If you can, add a legend, to help distinguish between the three time series.

- If you call the `plot()` function multiple times before calling `show()`, all of the lines you produce will end up on the same figure.  
- For the legend, if `var = 42`, then `f'foo{var}'` evaluates to `'foo42'`.

```python
# Put your code here
```

```python
for i in range(20):
    print("Solution below.")
```

```python
α_values = [0.0, 0.8, 0.98]
T = 200
x = np.empty(T+1)

fig, ax = plt.subplots()

for α in α_values:
    x[0] = 0
    for t in range(T):
        x[t+1] = α * x[t] + np.random.randn()
    ax.plot(x, label=f'$\\alpha = {α}$')

ax.legend()
```

## Branches and conditions

In Python, conditions are usually implemented with if-else syntax.

Here’s an example, that prints -1 for each negative number in an array and 1
for each nonnegative number

```python
numbers = [-9, 2.3, -11, 0]
```

```python
for x in numbers:
    if x < 0:
        print(-1)
    else:
        print(1)
```

**Exercise**

Simulate and plot the correlated time series

$$
    x_{t+1} = \alpha \, |x_t| + \epsilon_{t+1}
    \quad \text{where} \quad
    x_0 = 0
    \quad \text{and} \quad t = 0,\ldots,T
$$

were $ \{\epsilon_t\} $ is IID and standard normal.  Use

```python
α = 0.9
T = 200
```

Do not use an existing function such as `abs()` or `np.abs()`
to compute the absolute value.

Replace this existing function with an if-else condition.

```python
# Put your code here
```

```python
for i in range(20):
    print("Solution below.")
```

Here’s one way:

```python
x = np.empty(T+1)
x[0] = 0

for t in range(T):
    if x[t] < 0:
        abs_x = - x[t]
    else:
        abs_x = x[t]
    x[t+1] = α * abs_x + np.random.randn()

fig, ax = plt.subplots()
ax.plot(x)
```

Here’s another way to write the same thing:

```python
α = 0.9
T = 200
x = np.empty(T+1)
x[0] = 0

for t in range(T):
    abs_x = - x[t] if x[t] < 0 else x[t]
    x[t+1] = α * abs_x + np.random.randn()

fig, ax = plt.subplots()
ax.plot(x)
```

## Data Types

Computer programs typically keep track of a range of data types.

For example, `1.5` is a floating point number, while `1` is an integer.

Another data type is Boolean values, which can be either `True` or `False`

```python
x = True
x
```

We can check the type of any object in memory using the `type()` function.

```python
type(x)
```

What happens in the next line of code?

```python
y = 100 < 10
y
```

```python
type(y)
```

In arithmetic expressions, `True` is converted to `1` and `False` is converted `0`.

This is called **Boolean arithmetic** and is often useful in programming.

Here are some examples

```python
x + y
```

```python
x * y
```

```python
bools = [True, True, False, True]  # List of Boolean values
sum(bools)
```

### Containers

Python has several basic types for storing collections of (possibly heterogeneous) data.

We have already discussed lists.

A related data type is **tuples**, which are "immutable" lists

```python
x = ('a', 'b')  # Parentheses instead of the square brackets
x =  'a', 'b'    # Or no brackets --- the meaning is identical
x
```

```python
type(x)
```

In Python, an object is called **immutable** if, once created, the object cannot be changed.

Conversely, an object is **mutable** if it can still be altered after creation.

Python lists are mutable

```python
x = [1, 2]
x[0] = 10
x
```

But tuples are not

```python
x = (1, 2)
#x[0] = 10  # Uncomment and run
```

Tuples (and lists) can be “unpacked” as follows

```python
integers = (10, 20, 30)
x, y, z = integers
x
```

```python
y
```

#### Slice Notation

To access multiple elements of a sequence (a list, a tuple or a string), you can use Python’s slice
notation.

For example,

```python
a = ["a", "b", "c", "d", "e"]
a[1:]
```

```python
a[1:3]
```

The rule is `a[m:n]` returns `n - m` elements, starting at `a[m]`.

Also:

```python
a[-2:]  # Last two elements of the list
```

```python
s = 'foobar'
s[-3:]  # Last three elements
```

## Iterating

One of the most important tasks in computing is stepping through a
sequence of data and performing a given action.

One of Python’s strengths is its simple, flexible interface to iteration.

### Looping over Different Objects

Many Python objects are "iterable", in the sense that they can be looped over.

To give an example, let’s write the file us_cities.txt, which lists US cities and their population, to the present working directory.

```python
%%writefile us_cities.txt
new york: 8244910
los angeles: 3819702
chicago: 2707120
houston: 2145146
philadelphia: 1536471
phoenix: 1469471
san antonio: 1359758
san diego: 1326179
dallas: 1223229
```

Suppose that we want to make the information more readable, by capitalizing names and adding commas to mark thousands.

The program below reads the data in and makes the conversion:

```python
with open('us_cities.txt', 'r') as data_file:
    for line in data_file:
        city, population = line.split(':')         # Tuple unpacking
        city = city.title()                        # Capitalize city names
        population = f'{int(population):,}'        # Add commas to numbers
        print(city.ljust(15) + population)
```

### Looping without Indices

Python tends to favor looping without explicit indexing.

For example,

```python
x_values = [1, 2, 3]  # Some iterable x
for x in x_values:
    print(x * x)
```

is preferred to

```python
for i in range(len(x_values)):
    print(x_values[i] * x_values[i])
```

Python provides some facilities to simplify looping without indices.

One is `zip()`, which is used for stepping through pairs from two sequences.

For example, try running the following code

```python
countries = ('Japan', 'Korea', 'China')
cities = ('Tokyo', 'Seoul', 'Beijing')

for country, city in zip(countries, cities):
    print(f'The capital of {country} is {city}')
```

If we actually need the index from a list, one option is to use `enumerate()`.

To understand what `enumerate()` does, consider the following example

```python
letter_list = ['a', 'b', 'c']
for index, letter in enumerate(letter_list):
    print(f"letter_list[{index}] = '{letter}'")
```

### List Comprehensions

[List comprehensions](https://en.wikipedia.org/wiki/List_comprehension) are an elegant Python tool for creating lists.

Consider the following example, where the list comprehension is on the
right-hand side of the second line

```python
animals = ['dog', 'cat', 'bird']
plurals = [animal + 's' for animal in animals]
plurals
```

Here’s another example

```python
range(8)
```

```python
list(range(8))
```

```python
doubles = [2 * x for x in range(8)]
doubles
```

<!-- #region -->
## Comparisons and Logical Operators


### Comparisons

Comparisons are statements such as
<!-- #endregion -->

```python
1 < 10
```

In Python we can chain inequalities

```python
1 < 2 < 3
```

```python
1 <= 2 <= 3
```

When testing for equality we use `==`

```python
x = 1    # Assignment
x == 2   # Comparison
```

For non numerical comparisons we can use `is`

```python
None is None
```

For “not equal” use `!=`

```python
1 != 2
```

### Combining Expressions

We can combine expressions using `and`, `or` and `not`.

These are the standard logical connectives (conjunction, disjunction and denial)

```python
1 < 2 and 'f' in 'foo'
```

```python
1 < 2 and 'g' in 'foo'
```

```python
1 < 2 or 'g' in 'foo'
```

```python
not not True
```

### Coding Style and Documentation

A consistent coding style make code easier to understand and maintain.

You can find Python programming philosophy by typing `import this` at the prompt.

See also the Python style guide [PEP8](https://www.python.org/dev/peps/pep-0008/).


**Exercise**

1. Given two numeric lists or tuples `x_vals` and `y_vals` of equal length, compute
their inner product using `zip()`.
2. In one line, count the number of even numbers in 0,…,99.  (Hint: `x % 2` returns 0 if `x` is even, 1 otherwise.)
3. Given `pairs = ((2, 5), (4, 2), (9, 8), (12, 10))`, count the number of pairs `(a, b)`
such that both `a` and `b` are even.

```python
# Put your solutions here
```

```python
for i in range(20):
    print("Solutions below.")
```

**Part 1 Solution:**

Here’s one possible solution

```python
x_vals = [1, 2, 3]
y_vals = [1, 1, 1]
sum([x * y for x, y in zip(x_vals, y_vals)])
```

This also works

```python
sum(x * y for x, y in zip(x_vals, y_vals))
```

**Part 2 Solution:**

One solution is

```python
sum([x % 2 == 0 for x in range(100)])
```

This also works:

```python
sum(x % 2 == 0 for x in range(100))
```

So does this:

```python
len([x for x in range(100) if x % 2 == 0])
```

**Part 3 Solution:**

Here’s one possibility

```python
pairs = ((2, 5), (4, 2), (9, 8), (12, 10))
sum([x % 2 == 0 and y % 2 == 0 for x, y in pairs])
```

## Defining Functions

### Basic Syntax

Here’s a very simple Python function

```python
def f(x):
    return 2 * x + 1
```

Now that we’ve defined this function, let’s *call* it and check whether it does what we expect:

```python
f(1)   
```

```python
f(10)
```

Here’s a longer function, that computes the absolute value of a given number.

(Such a function already exists as a built-in, but let’s write our own for the
exercise.)

```python
def new_abs_function(x):
    if x < 0:
        abs_value = -x
    else:
        abs_value = x
    return abs_value
```

Let’s call it to check that it works:

```python
new_abs_function(3)
```

```python
new_abs_function(-3)
```

Note that a function can have arbitrarily many `return` statements (including zero).

```python
def another_abs_function(x):
    if x < 0:
        return -x
    return x
```

Functions without a return statement automatically return the special Python object `None`.

```python
def foo():
    pass
```

```python
x = foo()
print(x)
```

**Exercise**

Write a function that takes a string as an argument and returns the number of capital letters in the string.

(Hint:`'foo'.upper()` returns `'FOO'`.)

```python
# Put your solution here
```

```python
for i in range(20):
    print("Solution below.")
```

Here’s one solution:

```python
def count_upper_case(string):
    count = 0
    for letter in string:
        if letter == letter.upper() and letter.isalpha():
            count += 1
    return count

count_upper_case('The Rain in Spain')
```

Alternatively,

```python
def count_upper_case(string):
    return sum([c.isupper() and c.isalpha() for c in string])

count_upper_case('The Rain in Spain')
```

<!-- #region -->
### Keyword Arguments


The following example illustrates the syntax
<!-- #endregion -->

```python
def f(x, a=1, b=1):
    return a + b * x
```

The keyword argument values we supplied in the definition of `f` become the default values

```python
f(2)
```

They can be modified as follows

```python
f(2, a=4, b=5)
```

### The Flexibility of Python Functions

- Functions can be (and often are) defined inside other functions.  
- Any object can be passed to a function as an argument, including other functions.  
- A function can return any kind of object, including functions.

<!-- #region -->
### One-Line Functions: `lambda`


The `lambda` keyword is used to create simple functions on one line.

For example,
<!-- #endregion -->

```python
def f(x):
    return x**3
```

is equivalent to.

```python
f = lambda x: x**3
```

One use case is "anonymous" functions

```python
from scipy.integrate import quad
quad(lambda x: x**3, 0, 2)
```

### More examples of functions

Consider again the code

```python
ts_length = 100
ϵ_values = []   # empty list

for i in range(ts_length):
    e = np.random.randn()
    ϵ_values.append(e)

fig, ax = plt.subplots()
ax.plot(ϵ_values)
```

Let's create a function that does most of the work.

```python
def generate_data(n):
    ϵ_values = []
    for i in range(n):
        e = np.random.randn()
        ϵ_values.append(e)
    return ϵ_values

fig, ax = plt.subplots()
data = generate_data(100)
ax.plot(data)
```

Here's an alternative where we pass a function to a function:

```python
def generate_data(n, generator_type):
    ϵ_values = []
    for i in range(n):
        e = generator_type()
        ϵ_values.append(e)
    return ϵ_values

data = generate_data(100, np.random.uniform)
fig, ax = plt.subplots()
ax.plot(data)
```

**Exercise**

The binomial random variable $Y$ gives the number of successes in $ n $ binary
independent trials, where each trial succeeds with probability $ p $.

Without any import besides `from numpy.random import uniform`, write a function
`binomial_rv` such that `binomial_rv(n, p)` generates one draw of $ Y $.

Hint: If $ U $ is uniform on $ (0, 1) $ and $ p \in (0,1) $, then the expression `U < p` evaluates to `True` with probability $ p $.

```python
# Put your solution here
```

```python
for i in range(20):
    print("Solution below.")
```

Here's one solution:

```python
from numpy.random import uniform

def binomial_rv(n, p):
    count = 0
    for i in range(n):
        U = uniform()
        if U < p:
            count += 1 
    return count

binomial_rv(10, 0.5)
```

<!-- #region -->
## OOP: Objects and Methods


The traditional programming paradigm (Fortran, C, MATLAB, etc.) is called **procedural**.

Another important paradigm is **object-oriented programming** (OOP) 

In the OOP paradigm, data and functions are bundled together into “objects” —
and functions in this context are referred to as **methods**.

Methods are called on to transform the data contained in the object.

- Think of a Python list that contains data and has methods such as `append()`
  and `pop()` that transform the data.  

A third paradigm is **functional programming** 

* Built on the idea of composing functions.
* We'll discuss this more when we get to JAX

Python is a pragmatic language that blends object-oriented, functional and procedural styles.

But at a foundational level, Python *is* object-oriented.

By this we mean that, in Python, *everything is an object*.


### Objects


In Python, an *object* is a collection of data and instructions held in computer memory that consists of

1. a type  
1. a unique identity  
1. data (i.e., content, reference count)  
1. methods
<!-- #endregion -->

<!-- #region -->
#### Type


Python provides for different types of objects, to accommodate different categories of data.

For example
<!-- #endregion -->

```python
s = 'This is a string'
type(s)
```

```python
x = 42   # Now let's create an integer
type(x)
```

The type of an object matters for many expressions.

For example, the addition operator between two strings means concatenation

```python
'300' + 'cc'
```

On the other hand, between two numbers it means ordinary addition

```python
300 + 400
```

Consider the following expression

```python
#'300' + 400   # uncomment and run
```

Here we are mixing types, and it’s unclear to Python whether the user wants to

Python is *strongly typed* -- throws an error rather than trying to perform
hidden type conversion.

<!-- #region -->
#### Identity


In Python, each object has a unique identifier, which helps Python (and us) keep track of the object.

The identity of an object can be obtained via the `id()` function
<!-- #endregion -->

```python
y = 2.5
z = 2.5
id(y)
```

```python
id(z)
```

Here `y` and `z` have the same value but they are not the same object.


**Question** Why is the following case different??!

```python
a = 10
b = 10
id(a)
```

```python
id(b)
```

#### Object Content: Data and Attributes

If we set `x = 42` then we create an object of type `int` that contains
the data `42`.

In fact, it contains more, as the following example shows

```python
x = 42
x
```

```python
x.imag
```

```python
x.__class__
```

When Python creates this integer object, it stores with it various auxiliary
information, such as the imaginary part, and the type.

Any name following a dot is called an *attribute* of the object to the left of the dot.

- e.g.,`imag` and `__class__` are attributes of `x`.  

We see from this example that objects have attributes that contain auxiliary information.

<!-- #region -->
### Methods


Attributes that are *callable* (act like functions) are called *methods*.

Informally, methods are functions that are bundled with objects.
<!-- #endregion -->

```python
x = ['foo', 'bar']
callable(x.append)
```

```python
callable(x.__doc__)
```

Methods typically act on the data contained in the object they belong to, or combine that data with other data

```python
x = ['a', 'b']
x.append('c')
x
```

```python
s = 'This is a string'
s.upper()
```

```python
s.lower()
```

```python
s.replace('This', 'That')
```

A great deal of Python functionality is organized around method calls.

For example, consider the following piece of code

```python
x = ['a', 'b']
x[0] = 'aa'  # Item assignment using square bracket notation
x
```

It doesn’t look like there are any methods used here, but in fact the square
bracket assignment notation is just a convenient interface to a method call.

What actually happens is that Python calls the `__setitem__` method, as follows

```python
x = ['a', 'b']
x.__setitem__(0, 'aa')  # Equivalent to x[0] = 'aa'
x
```

## Inspection Using Rich

There’s a nice package called [rich](https://github.com/Textualize/rich) that helps us view the contents of an object.

```python
#!pip install rich   # Uncomment if necessary
```

For example,

```python
from rich import inspect
x = 10
inspect(x)
```

If we want to see the methods as well, we can use

```python
inspect(x, methods=True)
```

In fact there are still more methods, as you can see if you execute `inspect(10, all=True)`.

<!-- #region -->
## Names and Namespaces

### Variable Names in Python


Consider the Python statement
<!-- #endregion -->

```python
x = 42
```

In Python, `x` is called a **name**, and the statement `x = 42` **binds** the
name `x` to the integer object `42`.

There is no problem binding two or more names to the one object, regardless of what that object is

```python
def f(string):      # Create a function called f
    print(string)   # that prints any string it's passed

g = f
id(g) == id(f)
```

```python
g('test')
```

What happens when the number of names bound to an object goes to zero?

Here’s an example of this situation:

```python
x = 'foo'    # the string object 'foo' is created in memory and x is bound to it 
x = 'bar'    # x is rebound to the string object 'bar'
```

Now there are no names bound to `'foo'`.

Python releases `'foo'` to be garbage collected.

(The memory slot that stores that object is returned to the operating system.)

<!-- #region -->
### Namespaces


Recall from the preceding discussion that the statement
<!-- #endregion -->

```python
x = 42
```

binds the name `x` to the integer object on the right-hand side.

This process of binding `x` to the correct object is implemented as a dictionary.

This dictionary is called a namespace.


Python uses multiple namespaces, creating them on the fly as necessary.

For example, every time we import a module, Python creates a namespace for that module.

To see this in action, suppose we write a script `mathfoo.py` as follows

```python
%%writefile mathfoo.py
pi = 'foobar'
x = 42
```

Let's import this "module"

```python
import mathfoo
```

Next let’s import the `math` module from the standard library

```python
import math
```

Both of these modules have an attribute called `pi`

```python
math.pi
```

```python
mathfoo.pi
```

These two different bindings of `pi` exist in different namespaces, each one implemented as a dictionary.

If you wish, you can look at the dictionary directly, using `module_name.__dict__`.

```python
mathfoo.__dict__.keys()  # Show all keys in the dictionary
```

```python
mathfoo.__dict__['pi']
```

<!-- #region -->
### Interactive Sessions


In Python, **all** code executed by the interpreter runs in some module.

What about commands typed at the prompt?

These are also regarded as being executed within a module — in this case, a module called `__main__`.

To check this, we can look at the current module name via the value of `__name__`
<!-- #endregion -->

```python
print(__name__)
```

When we run a script using IPython’s `run` command, the contents of the file are
executed as part of `__main__` too.

To see this, let’s create a file `mod.py` that prints its own `__name__` attribute

```python
%%file mod.py
print(__name__)
```

Now let’s look at two different ways of running it in IPython

```python
import mod  # Standard import
```

```python
%run mod.py  # Run interactively
```

In the second case, the code is executed as part of `__main__`, so `__name__` is equal to `__main__`.

To see the contents of the namespace of `__main__` we use `vars()` rather than `vars(__main__)`.

If you do this in IPython, you will see a whole lot of variables that IPython
needs, and has initialized when you started up your session.

If you prefer to see only the variables you have initialized, use the IPython
"magic" command `%whos`

```python
x = 2
y = 3
import numpy as np

%whos
```

### Global and local namespaces

The **global namespace** is *the namespace of the module currently being executed*.

For example, suppose that we start the interpreter and begin making assignments.

We are now working in the module `__main__`, and hence the namespace for `__main__` is the global namespace.


When we call a function, the interpreter creates a **local namespace** for that
function, and registers the variables in that namespace.

Variables in the local namespace are called *local variables*.

After the function returns, the function's namespace is deallocated and lost.

While the function is executing, we can view the contents of the local namespace with `locals()`.

For example, consider

```python
def f(x):
    a = 2
    print(locals())
    return a * x
```

Now let’s call the function

```python
f(1)
```

<!-- #region -->
### The `__builtins__` Namespace


We have been using various built-in functions, such as `max(), dir(), str(), list(), len(), range(), type()`, etc.

How does access to these names work?

- These definitions are stored in a module called `__builtin__`.  
- They have their own namespace called `__builtins__`.
<!-- #endregion -->

```python
# Show the first 10 names in `__builtins__`
dir(__builtins__)[:10]
```

We can access elements of the namespace as follows

```python
__builtins__.max
```

But `__builtins__` is special, because we can always access them directly as well

```python
max
```

```python
__builtins__.max == max
```

<!-- #region -->
## Name Resolution

Namespaces help us organize variable names.

(Type `import this` at the prompt and look at the last item that’s printed)

At any point of execution, there are at least two namespaces that can be accessed directly.

(“Accessed directly” means without using a dot, as in  `pi` rather than `math.pi`)

These namespaces are

- The global namespace (of the module being executed)  
- The builtin namespace  


If the interpreter is executing a function, then the directly accessible namespaces are

- The local namespace of the function  
- The global namespace (of the module being executed)  
- The builtin namespace  


Sometimes functions are defined within other functions, like so
<!-- #endregion -->

```python
def f():
    a = 2
    def g():
        b = 4
        print(a * b)
    g()
```

<!-- #region -->
Here `f` is the *enclosing function* for `g`, and each function gets its
own namespaces.

Now we can give the rule for how namespace resolution works:

The order in which the interpreter searches for names is

1. the local namespace (if it exists)  
1. the hierarchy of enclosing namespaces (if they exist)  
1. the global namespace  
1. the builtin namespace  


If the name is not in any of these namespaces, the interpreter raises a `NameError`.

This is called the **LEGB rule** (local, enclosing, global, builtin).
<!-- #endregion -->

### Mutable Versus Immutable Parameters

This is a good time to say a little more about mutable vs immutable objects.

Consider the code segment

```python
def f(x):
    x = x + 1
    return x

x = 100
print(f"f(x) = {f(x)} and x = {x}")
```

<!-- #region -->
Here's what happens

* First `f` and `x` are registered in the global namespace.
* The call `f(x)` creates a local namespace and adds `x` to it, bound to `100`.
* Next, this local `x` is rebound to the new integer object `101`, and this value is returned.
 

None of this affects the global `x`.

However, it’s a different story when we use a **mutable** data type such as a list
<!-- #endregion -->

```python
def f(x):
    x[0] = x[0] + 1
    return x

x = [100]
print(f"f(x) = {f(x)} and x = {x}")
```

Here’s what happens

- `f` is registered as a function in the global namespace  
- `x` is bound to `[100]` in the global namespace  
- The call `f(x)`  
  - Creates a local namespace  
  - Adds `x` to the local namespace, bound to `[100]`  
  - Mutates the data in the list to `[101]`
  - Returns the mutated list
 
Global `x` is still bound to the (now mutated) list
