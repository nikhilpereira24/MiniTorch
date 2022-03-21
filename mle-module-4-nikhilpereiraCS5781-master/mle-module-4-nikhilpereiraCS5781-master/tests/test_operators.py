from minitorch.operators import (
    mul,
    add,
    neg,
    relu,
    addLists,
    prod,
    negList,
    id,
    inv,
    lt,
    eq,
    max,
    sigmoid,
    relu_back,
    log_back,
    inv_back,
    sum,
)
from hypothesis import given
from hypothesis.strategies import lists
from tests.strategies import small_floats, assert_close
import pytest
from minitorch import MathTest


# ## Task 0.1 Basic hypothesis tests.


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_same_as_python(x, y):
    "Check that the main operators all return the same value of the python version"
    assert_close(mul(x, y), x * y)
    assert_close(add(x, y), x + y)
    assert_close(neg(x), -x)
    assert_close(max(x, y), x if x > y else y)
    if x != 0.0:
        assert_close(inv(x), 1.0 / x)


@pytest.mark.task0_1
@given(small_floats)
def test_relu(a):
    if a > 0:
        assert relu(a) == a
    if a < 0:
        assert relu(a) == 0.0


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_relu_back(a, b):
    if a > 0:
        assert relu_back(a, b) == b
    if a < 0:
        assert relu_back(a, b) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_id(a):
    assert id(a) == a


@pytest.mark.task0_1
@given(small_floats)
def test_lt(a):
    "Check that a - 1.0 is always less than a"
    assert lt(a - 1.0, a) == 1.0
    assert lt(a, a - 1.0) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_max(a):
    assert max(a - 1.0, a) == a
    assert max(a, a - 1.0) == a
    assert max(a + 1.0, a) == a + 1.0
    assert max(a, a + 1.0) == a + 1.0


@pytest.mark.task0_1
@given(small_floats)
def test_eq(a):
    assert eq(a, a) == 1.0
    assert eq(a, a - 1.0) == 0.0
    assert eq(a, a + 1.0) == 0.0


# ## Task 0.2 - Property Testing

# Implement the following property checks
# that ensure that your operators obey basic
# mathematical rules.


@pytest.mark.task0_2
@given(small_floats)
def test_sigmoid(a):
    """Check properties of the sigmoid function, specifically
    * It is always between 0.0 and 1.0.
    * one minus sigmoid is the same as negative sigmoid
    * It crosses 0 at 0.5
    * it is  strictly increasing.
    """
    assert 0.0 <= sigmoid(a) <= 1.0  # It is always between 0.0 and 1.0.
    print(
        1 - sigmoid(a), sigmoid(-a)
    )  # one minus sigmoid is the same as negative sigmoid
    assert_close((1 - sigmoid(a)), sigmoid(-a))
    if a == 0:  # It crosses 0 at 0.5
        assert sigmoid(a) == 0.5
    assert sigmoid(a) * (1 - sigmoid(a)) >= 0  # it is  strictly increasing.


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_transitive(a, b, c):
    "Test the transitive property of less-than (a < b and b < c implies a < c)"
    # TODO: Implement for Task 0.2.
    if a < b < c:  # testing the transitive property
        assert lt(a, b) == 1.0
        assert lt(b, c) == 1.0
        assert lt(a, c) == 1.0
    elif a > b > c:  # testing that when a>b>c less than doesnt hold
        assert lt(a, b) == 0
        assert lt(b, c) == 0
        assert lt(a, c) == 0
    elif a == b == c:  # testing that when a=b=c than the transitive property holds
        assert lt(a, b) == 0
        assert lt(b, c) == 0
        assert lt(a, c) == 0


@pytest.mark.task0_2
@given(small_floats, small_floats)
def test_symmetric(a, b):
    """
    Write a test that ensures that :func:`minitorch.operators.mul` is symmetric, i.e.
    gives the same value regardless of the order of its input.
    """
    assert mul(a, b) == mul(b, a)  # tests symetric


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_distribute(z, x, y):
    r"""
    Write a test that ensures that your operators distribute, i.e.
    :math:`z \times (x + y) = z \times x + z \times y`
    """
    assert_close(z * (add(x, y)), ((mul(z, x)) + (mul(z, y))))  # tests distribution


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_other(a, b, c):
    """
    Making sure the max function works and is consistent with the lt function
    """
    if a > b > c:  # test the max function and lt function our consistent
        assert max(a, b) == a
        assert max(b, c) == b
        assert lt(a, b) == 0.0
        assert lt(b, c) == 0.0


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats, small_floats)
def test_zip_with(a, b, c, d):
    x1, x2 = addLists([a, b], [c, d])
    y1, y2 = a + c, b + d
    assert_close(x1, y1)
    assert_close(x2, y2)


@pytest.mark.task0_3
@given(
    lists(small_floats, min_size=5, max_size=5),
    lists(small_floats, min_size=5, max_size=5),
)
def test_sum_distribute(ls1, ls2):
    """
    Write a test that ensures that the sum of `ls1` plus the sum of `ls2`
    is the same as the sum of each element of `ls1` plus each element of `ls2`.
    """
    temp = 0
    for x in ls1:
        temp = temp + x  # manual running sums
    temp2 = 0
    for x in ls2:
        temp2 = temp2 + x  # manual running sums
    assert_close(
        add(sum(ls1), sum(ls2)), add(temp, temp2)
    )  # asserts the sum function works with adding running sums


@pytest.mark.task0_3
@given(lists(small_floats))
def test_sum(ls):
    assert_close(sum(ls), sum(ls))  # tests the sum function


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats)
def test_prod(x, y, z):
    assert_close(prod([x, y, z]), x * y * z)  # tests the product function


@pytest.mark.task0_3
@given(lists(small_floats))
def test_negList(ls):
    check = negList(ls)
    for i in range(len(ls)):
        assert_close(check[i], -ls[i])


# ## Generic mathematical tests

# For each unit this generic set of mathematical tests will run.


one_arg, two_arg, _ = MathTest._tests()


@given(small_floats)
@pytest.mark.parametrize("fn", one_arg)
def test_one_args(fn, t1):
    name, base_fn, _ = fn
    base_fn(t1)


@given(small_floats, small_floats)
@pytest.mark.parametrize("fn", two_arg)
def test_two_args(fn, t1, t2):
    name, base_fn, _ = fn
    base_fn(t1, t2)


@given(small_floats, small_floats)
def test_backs(a, b):
    relu_back(a, b)
    inv_back(a + 2.4, b)
    log_back(abs(a) + 4, b)
