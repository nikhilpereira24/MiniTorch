"""
Collection of the core mathematical operators used throughout the code base.
"""


import math

# ## Task 0.1
# Implementation of a prelude of elementary functions.


def mul(x, y):
    ":math:`f(x, y) = x * y`"
    return x * y


def id(x):
    ":math:`f(x) = x`"
    return x


def add(x, y):
    ":math:`f(x, y) = x + y`"
    return x + y


def neg(x):
    ":math:`f(x) = -x`"
    return float(-x)


def lt(x, y):
    ":math:`f(x) =` 1.0 if x is less than y else 0.0"
    if x < y:
        return 1.0
    else:
        return 0.0


def eq(x, y):
    ":math:`f(x) =` 1.0 if x is equal to y else 0.0"
    return 1.0 if x == y else 0.0


def max(x, y):
    ":math:`f(x) =` x if x is greater than y else y"
    if x > y:
        return x
    else:
        return y


def is_close(x, y):
    ":math:`f(x) = |x - y| < 1e-2` "
    if x is None and y is None:
        return 1
    else:
        if math.fabs(x - y) < 1e-2:
            return 1
        else:
            return 0


def sigmoid(x):
    r"""
    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}`

    (See `<https://en.wikipedia.org/wiki/Sigmoid_function>`_ .)

    Calculate as

    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}` if x >=0 else :math:`\frac{e^x}{(1.0 + e^{x})}`

    for stability.

    Args:
        x (float): input

    Returns:
        float : sigmoid value
    """
    if x >= 0.0:
        return (1.0) / (1.0 + math.exp(-x))
    else:
        return (math.exp(x)) / (1.0 + math.exp(x))


def relu(x):
    """
    :math:`f(x) =` x if x is greater than 0, else 0

    (See `<https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`_ .)

    Args:
        x (float): input

    Returns:
        float : relu value
    """
    if x > 0:
        return x
    else:
        return 0.0


EPS = 1e-6


def log(x):
    ":math:`f(x) = log(x)`"
    return math.log(x + EPS)


def exp(x):
    ":math:`f(x) = e^{x}`"
    return math.exp(x)


def log_back(x, d):
    r"If :math:`f = log` as above, compute d :math:`d \times f'(x)`"
    return d / (x + EPS)


def inv(x):
    ":math:`f(x) = 1/x`"
    return 1.0 / x


def inv_back(x, d):
    r"If :math:`f(x) = 1/x` compute d :math:`d \times f'(x)`"
    return -(1.0 / x ** 2.0) * d


def relu_back(x, d):
    r"If :math:`f = relu` compute d :math:`d \times f'(x)`"
    if x > 0:
        return d
    elif x == 0:
        return EPS
    else:
        return 0.0


def map(fn):
    """
    Higher-order map.

    .. image:: figs/Ops/maplist.png


    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (one-arg function): Function from one value to one value.

    Returns:
        function : A function that takes a list, applies `fn` to each element, and returns a
        new list
    """

    def apply(ls):  # define inner function
        ret = []  # create empty list
        for x in ls:  # for the elements in the list
            ret.append(fn(x))  # add the elements after applying the function passed it
        return ret  # return the list

    return apply  # return the inner function


def negList(ls):
    "Use :func:`map` and :func:`neg` to negate each element in `ls`"
    negater = map(neg)  # create an object that stores the function
    return negater(ls)  # call the stored function on the list


def zipWith(fn):
    """
    Higher-order zipwith (or map2).

    .. image:: figs/Ops/ziplist.png

    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (two-arg function): combine two values

    Returns:
        function : takes two equally sized lists `ls1` and `ls2`, produce a new list by
        applying fn(x, y) on each pair of elements.

    """

    def apply(ls1, ls2):  # inner function taking two lists
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret
        # ret = []  # empty return list
        # counter = 0  # estabilish a counter
        # if len(ls1) == len(ls2):  # if they equal each other
        #     for y in ls2:  # iterate through either list 1 or 2
        #         ret.append(
        #             fn(ls1[counter], y)
        #         )  # append the same index for both list 1 and 2 after applying the function
        #         counter += 1  # add to the counter
        #     return ret  # return the final list

    return apply  # return the function


def addLists(ls1, ls2):
    "Add the elements of `ls1` and `ls2` using :func:`zipWith` and :func:`add`"
    zipAdder = zipWith(add)  # store the object function with add
    return zipAdder(ls1, ls2)  # apply the object function on the two passed lists


def reduce(fn, start):
    """
    Higher-order reduce.
    Args:
        fn (two-arg function): combine two values
        start (float): start value
    Returns:
        function : function that takes a list `ls` of elements
        computes the reduction
    """

    def apply(ls):  # define an inner function
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return apply
    # counter = 0  # create a counter variable
    # for y in ls:  # for the elements in the list
    #     if counter == 0:  # if the index is 0
    #         temp = fn(
    #             start, ls[counter]
    #         )  # apply fn with the list and the start variable
    #         counter += 1  # iterate the counter
    #     else:
    #         temp = fn(
    #             temp, ls[counter]
    #         )  # apply the function on the running temporary value
    #         counter += 1  # iteratre the counter
    # return temp  # return the final value

    # return apply  # return the inner function


def sum(ls):
    "Sum up a list using :func:`reduce` and :func:`add`."
    return reduce(add, 0.0)(ls)
    # check that the list is not empty
    # if len(ls) > 0:
    #     reduceSummer = reduce(
    #         add, 0
    #     )  # store the object function with add and 0 being the starting value
    #     return reduceSummer(ls)  # apply the function reduceSummer on the list
    # else:
    #     return None  # return None for empty lists can also throw an error here


def prod(ls):
    "Product of a list using :func:`reduce` and :func:`mul`."
    return reduce(mul, 1.0)(ls)
    # if len(ls) > 0:  # check the list is not empty
    #     reduceMultiplier = reduce(
    #         mul, 1
    #     )  # store the object function with multipy and 1 being the starting value
    #     return reduceMultiplier(
    #         ls
    #     )  # apply and return the reduceMultiplier function on the list
    # else:
    #     return 1  # return None
