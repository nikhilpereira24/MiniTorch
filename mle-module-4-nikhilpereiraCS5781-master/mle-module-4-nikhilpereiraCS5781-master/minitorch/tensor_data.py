import random
from .operators import prod
from numpy import array, float64, ndarray
import numba

MAX_DIMS = 32


class IndexingError(RuntimeError):
    "Exception raised for indexing errors."
    pass


def index_to_position(index, strides):
    """
    Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
        index (array-like): index tuple of ints
        strides (array-like): tensor strides

    Returns:
        int : position in storage
    """
    # index = [1,0]
    # stride = [5,1]
    # sum [5*1 + 0*1] = 5
    # index = [1,2]
    # stride = [5,1]
    # sum = [5*1 + 2*1] = 7
    # return the sum of the multiplication of index*strides

    # returns sum(index*strides)
    mul = 0  # variable to accumulate the multiplication result which will be returned as the index in storage
    for i, s in zip(
        index, strides
    ):  # iterate through both index of the tensor and strides
        mul += i * s  # multiply each pair of index, stride passed in
    return mul  # return the accumulated result


def to_index(ordinal, shape, out_index):
    """
    Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
        ordinal (int): ordinal position to convert.
        shape (tuple): tensor shape.
        out_index (array): the index corresponding to position.

    Returns:
    None : Fills in `out_index`.
    """
    cur_ord = ordinal + 0
    for i in range(len(shape) - 1, -1, -1):
        sh = shape[i]
        out_index[i] = int(cur_ord % sh)
        cur_ord = cur_ord // sh
    # for dim in range(len(shape)):  # iterate through the shape of the tensor passed in
    #     slicer = prod(
    #         shape[dim + 1 :]
    #     )  # value that divides the tensor by dimension dim, Example Shape (2,3,4), prod = 3x4 = 12 cubes in dim = 0
    #     out_index[dim] = int(
    #         ordinal / slicer
    #     )  # creates slice of tensor to find index in tensor, Example ordinal = 24, out_index[0] = 24/12 = 2, out_index = (2,0,0) filled on next iterator of for loop
    #     ordinal %= slicer  # update ordinal value by using the modular to get the next dimension to be set, Since 24 % 12 = 0, the next two index in shape will be set to 0


def broadcast_index(big_index, big_shape, shape, out_index):
    """
    Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
        big_index (array-like): multidimensional index of bigger tensor
        big_shape (array-like): tensor shape of bigger tensor
        shape (array-like): tensor shape of smaller tensor
        out_index (array-like): multidimensional index of smaller tensor

    Returns:
        None : Fills in `out_index`.
    """
    # Convert a `big_index` in `big_shape` to a smaller `out_index`
    # in `shape` following broadcasting rules
    # Convert a big_index of big_shape to a smaller out_index of shape following broadcasting rules by mutating the out_index argument.
    # big_shape may be larger or have more dimensions than the shape given.
    # If so, additional dimensions of big_shape may need to be mapped to 0 or removed when mutating out_index

    # shape = list(shape)  # store shape
    # i = 0  # store a counter variable that tells us how many dimension we need to add
    # while len(shape) != len(big_index):
    #     shape.insert(0, 1)  # adding 1s until the small shape == big shape
    #     i += 1  # keeping track how many times this happen

    # nlist = list()  # new list to add the resulting out indices
    # for dim in range(len(shape)):
    #     # calculating big index % shape, #Example let big_index shape = [5,2], shape = [1,2], 5%1 = 0, 2%2 = 0, out_index[0,0]
    #     # Example 2, big_index = [1,1], shape = [1,2], 1%1 = 0, 1%2 = 1 so out_index[0,1]
    #     nlist.append(big_index[dim] % shape[dim])  # append this value to an empty list

    # xlist = nlist[
    #     i:
    # ]  # removing any extra dimensions that were previously set because they dont matter in smaller out tensor
    # for x in range(len(xlist)):  # loop through this list
    #     out_index[x] = xlist[x]  # set out index
    for i, s in enumerate(shape):
        if s > 1:
            out_index[i] = big_index[i + (len(big_shape) - len(shape))]
        else:
            out_index[i] = 0
    return None


def shape_broadcast(shape1, shape2):
    """
    Broadcast two shapes to create a new union shape.

    Args:
        shape1 (tuple) : first shape
        shape2 (tuple) : second shape

    Returns:
        tuple : broadcasted shape

    Raises:
        IndexingError : IndexingError
    """
    a, b = shape1, shape2
    m = max(len(a), len(b))
    c_rev = [0] * m
    a_rev = list(reversed(a))
    b_rev = list(reversed(b))
    for i in range(m):
        if i >= len(a):
            c_rev[i] = b_rev[i]
        elif i >= len(b):
            c_rev[i] = a_rev[i]
        else:
            c_rev[i] = max(a_rev[i], b_rev[i])
            if a_rev[i] != c_rev[i] and a_rev[i] != 1:
                raise IndexingError("Broadcast failure {a} {b}")
            if b_rev[i] != c_rev[i] and b_rev[i] != 1:
                raise IndexingError("Broadcast failure {a} {b}")
    return tuple(reversed(c_rev))

    # def broadcastHelper(shape1, shape2):
    #     shapeZipped = tuple(zip(shape1, shape2))  # zip together the two shapes
    #     broadCast = []
    #     for tup in shapeZipped:
    #         if (
    #             tup[0] == tup[1] or 1 in tup
    #         ):  # Rule 1: Any dimension of size 1 can be zipped with dimensions of size n > 1 by assuming the dimension is copied n times.
    #             # if the two shapes are the same size, and any of the dimensions are 1, return the broadcasted shapes
    #             broadCast.append(max(tup))
    #         elif tup[0] != tup[1] and 1 not in tup:  # if violates broadcast rules
    #             raise IndexingError("Cannot Broadcast")
    #     return tuple(broadCast)

    # # Rule 2: Extra dimensions of shape 1 can be added to a tensor to ensure the same number of dimensions with another tensor.
    # # if the two shapes are not the same size, add a view 1 to the left side of tuple and then broadcast
    # def resizeShape(shape1, shape2):  # function that resizes the smaller shape
    #     diffSize = len(shape1) != len(shape2)
    #     minShape = (list(shape1), list(shape2))[len(shape1) > len(shape2)]
    #     maxShape = (list(shape1), list(shape2))[len(shape1) < len(shape2)]
    #     while diffSize:
    #         minShape.insert(
    #             0, 1
    #         )  # Rule 3: Any extra dimension of size 1 can only be implicitly added on the left side of the shape.
    #         if len(minShape) == len(maxShape):
    #             diffSize = False
    #     return tuple(minShape), tuple(maxShape)

    # sameSize = len(list(shape1)) == len(list(shape2))  # boolean sameSize variable
    # if sameSize:
    #     broadCastShape = broadcastHelper(shape1, shape2)  # try broadcasting
    # else:
    #     shape1, shape2 = resizeShape(shape1, shape2)  # resize the shapes
    #     broadCastShape = broadcastHelper(shape1, shape2)  # broadcast shapes
    # return broadCastShape


def strides_from_shape(shape):
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    def __init__(self, storage, shape, strides=None):
        if isinstance(storage, ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self):  # pragma: no cover
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self):
        """
        Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous
        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a, shape_b):
        return shape_broadcast(shape_a, shape_b)

    def index(self, index):
        if isinstance(index, int):
            index = array([index])
        if isinstance(index, tuple):
            index = array(index)

        # Check for errors
        if index.shape[0] != len(self.shape):
            raise IndexingError(f"Index {index} must be size of {self.shape}.")
        for i, ind in enumerate(index):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {index} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {index} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self):
        lshape = array(self.shape)
        out_index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self):
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key):
        return self._storage[self.index(key)]

    def set(self, key, val):
        self._storage[self.index(key)] = val

    def tuple(self):
        return (self._storage, self._shape, self._strides)

    def permute(self, *order):
        """
        Permute the dimensions of the tensor.

        Args:
            order (list): a permutation of the dimensions

        Returns:
            :class:`TensorData`: a new TensorData with the same storage and a new dimension order.
        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"
        return TensorData(
            self._storage,
            tuple([self.shape[o] for o in order]),
            tuple([self._strides[o] for o in order]),
        )

        # newStrideFinal = []  # create an empty list to hold the final stride
        # newShapeFinal = []  # create an empty list to hold the final shape
        # for x in range(len(self.shape)):  # iterate through the shape
        #     newStrideFinal.append(
        #         self.strides[order[x]]
        #     )  # set the final stride to the orders index
        #     newShapeFinal.append(
        #         self.shape[order[x]]
        #     )  # set the final shape to the the orders index
        # return TensorData(  # creates the Tensor
        #     storage=self._storage,
        #     shape=tuple(newShapeFinal),
        #     strides=tuple(newStrideFinal),
        # )

    def to_string(self):
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
