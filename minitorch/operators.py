"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    r"""Multiplies two numbers : $f(x, y) = x \times y$

    Args:
        x (float): First number
        y (float): Second number

    Returns:
        float: Result of x * y
    """
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged : $f(x) = x$

    Args:
        x (float): A number

    Returns:
        float: Input unchanged
    """
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers : $f(x, y) = x + y$

    Args:
        x (float): First number
        y (float): Second number

    Returns:
        float: Result of the addition x + y
    """
    return x + y


def neg(x: float) -> float:
    """Negates a number : $f(x) = -x$

    Args:
        x (float): A number

    Returns:
        float: The negated number
    """
    return -x


def lt(x: float, y: float) -> float:
    """Checks if one number is less than another : $f(x) =$ 1.0 if x is less than y else 0.0

    Args:
        x (float): Number to be checked
        y (float): Number checked against

    Returns:
        float: Result of the comparison in float
    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Checks if two numbers are equal : $f(x) =$ 1.0 if x is equal to y else 0.0

    Args:
        x (float): Number to be checked
        y (float): Number to be checked against

    Returns:
        float: Result of the comparison in float
    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers : $f(x) =$ x if x is greater than y else y

    Args:
        x (float): First number
        y (float): Second number

    Returns:
        float: Largest number between the two
    """
    return x if bool(lt(y, x)) else y


def is_close(x: float, y: float, eps: float = 1e-2) -> float:
    """Checks if two numbers are close in value : $f(x) = |x - y| < 1e-2$"

    Args:
        x (float): First number
        y (float): Second number
        eps (float) : Maximum distance to be considered close

    Returns:
        float: Whether or not the numbers are closed
    """
    return lt(abs(x - y), eps)


def sigmoid(x: float) -> float:
    r"""Calculates the sigmoid function : $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$

        (See https://en.wikipedia.org/wiki/Sigmoid_function )

        Calculate as

        $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

        for stability.

    Args:
        x (float): Input number

    Returns:
        float: Result of the sigmoid function applied to the input
    """
    if x >= 0:
        return 1.0 / (1.0 + exp(-x))
    else:
        return exp(x) / (1.0 + exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function : $f(x) =$ x if x is greater than 0, else 0

        (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks) .)

    Args:
        x (float): Input number

    Returns:
        float: Result of the ReLU function applied to the input
    """
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Calculates the natural logarithm : $f(x) = log(x)$

    Args:
        x (float): Input number

    Returns:
        float: Result of the log function applied to the input
    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Calculates the exponential function : $f(x) = e^{x}$

    Args:
        x (float): Input number

    Returns:
        float: Result of the exp function applied to the input
    """
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    r"""Computes the derivative of log times a second arg : If $f = log$ as above, compute $d \times f'(x)$


    Args:
        x (float): Input number
        d (float): Number to multiply the derivative

    Returns:
        float: Result of d • log'(x)
    """
    if x > 0.0:
        return d / (x + EPS)
    else:
        raise ValueError(
            "The log function is undefined for x <= 0 and thus cannot be derived"
        )


def inv(x: float) -> float:
    """Calculates the reciprocal : $f(x) = 1/x$

    Args:
        x (float): Input number

    Raises:
        ValueError: The reciprocal function is undefined for x = 0.0

    Returns:
        float: Result of the inv function applied to the input
    """
    if x == 0.0:
        raise ValueError("Reciprocal function is undefined for x = 0.0")
    else:
        return 1.0 / x


def inv_back(x: float, d: float) -> float:
    r"""Computes the derivative of reciprocal times a second arg : If $f(x) = 1/x$ compute $d \times f'(x)$

    Args:
        x (float): Input number
        d (float): Number to multiply the derivative

    Raises:
        ValueError: The reciprocal function is undefined for x = 0.0 and thus cannot be derived

    Returns:
        float: Result of d • inv'(x)
    """
    if x == 0.0:
        raise ValueError(
            "The reciprocal function is undefined for x = 0.0 and thus cannot be derived"
        )
    else:
        return neg(d / x**2)


def relu_back(x: float, d: float) -> float:
    r"""Computes the derivative of ReLU times a second arg : If $f = relu$ compute $d \times f'(x)$
        The ReLU function cannot be derived for x = 0 as ReLU+(0) ≠ ReLU-(0) but Pytorch's Autograd
        set its value to 0 according to https://hal.science/hal-03265059/file/Impact_of_ReLU_prime.pdf

    Args:
        x (float): Input number
        d (float): Number to multiply the derivative

    Returns:
        float: Result of d • ReLU'(x)
    """
    return 0.0 if x < 0 else d


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Nota Bene : We could use map from python but the idea is to do it ourself

    Args:
        fn: Function from one value to one value.

    Returns:
         A function that takes a list, applies `fn` to each element, and returns a
         new list
    """

    def f(iterable_x):
        return [fn(x) for x in iterable_x]

    return f


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Use `map` and `neg` to negate each element in `ls`

    Args:
        ls (Iterable[float]): Iterable to negate

    Returns:
        Iterable[float]: Result of the negation
    """
    return map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
         Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """

    def f(iterable_x, iterable_y):
        return [fn(x, y) for x, y in zip(iterable_x, iterable_y)]

    return f


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add the elements of `ls1` and `ls2` using `zipWith` and `add`

    Args:
        ls1 (Iterable[float]): First Iterable
        ls2 (Iterable[float]): Second Iterable

    Returns:
        Iterable[float]: Result of the element-wise sum of the two Iterables
    """
    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
         Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """

    def f(iterable_x):
        val = start
        for x in iterable_x:
            val = fn(val, x)
        return val

    return f


def sum(ls: Iterable[float]) -> float:
    """Sum up a list using `reduce` and `add`.

    Args:
        ls (Iterable[float]): List of floats to be summed

    Returns:
        float: Sum of the list's elements
    """
    return reduce(add, 0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Product of a list using `reduce` and `mul`.

    Args:
        ls (Iterable[float]): List of floats to be multiplied

    Returns:
        float: Product of the list's elements
    """
    return reduce(mul, 1.0)(ls)
