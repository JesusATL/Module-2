import numpy as np
from .tensor_data import (
    count,
    index_to_position,
    broadcast_index,
    shape_broadcast,
    MAX_DIMS,
)


def tensor_map(fn):
    """
    Higher-order tensor map function ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
        fn: function from float-to-float to apply
        out (array): storage for out tensor
        out_shape (array): shape for out tensor
        out_strides (array): strides for out tensor
        in_storage (array): storage for in tensor
        in_shape (array): shape for in tensor
        in_strides (array): strides for in tensor

    Returns:
        None : Fills in `out`
    """

    def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):
        # TODO: Implement for Task 2.2.
        #raise NotImplementedError('Need to implement for Task 2.2')

        idx_out = [ 0.0 for e in out_strides]
        idx_in = [ 0.0 for e in in_strides]

        for pos in range(len(in_storage)):
            count(pos, out_shape,  idx_out)
            broadcast_index(idx_out, out_shape, in_shape, idx_in)
            out[index_to_position(idx_out, out_strides)] = fn(in_storage[index_to_position(idx_in, in_strides)])
    return _map

def map(fn):
    """
    Higher-order tensor map function ::

      fn_map = map(fn)
      b = fn_map(a)


    Args:
        fn: function from float-to-float to apply.
        a (:class:`TensorData`): tensor to map over
        out (:class:`TensorData`): optional, tensor data to fill in,
               should broadcast with `a`

    Returns:
        :class:`TensorData` : new tensor data
    """

    f = tensor_map(fn)

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)
        f(*out.tuple(), *a.tuple())
        return out

    return ret


def tensor_zip(fn):
    """
    Higher-order tensor zipWith (or map2) function. ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)


    Args:
        fn: function mapping two floats to float to apply
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """

    def _zip(
        out,
        out_shape,
        out_strides,
        a_storage,
        a_shape,
        a_strides,
        b_storage,
        b_shape,
        b_strides,
    ):
        # TODO: Implement for Task 2.2.

#        idx_out = [ 0.0 for e in out_strides]
#        idx_b  =[ 0.0 for e in b_strides]
#
#        for pos in range(len(a_storage)):
#            count(pos, out_shape,  idx_out)
#            count(pos, b_shape,  idx_b)
#
#            out[index_to_position(idx_out, out_strides)] = fn(a_storage[[pos]] , b_storage[index_to_position(idx_b, b_strides)])

        idx_out = [ 0.0 for e in out_strides]
        idx_a  =[ 0.0 for e in a_strides]
        idx_b  =[ 0.0 for e in b_strides]

        for pos in range(len(out)):
            count(pos, out_shape,  idx_out)
            broadcast_index(idx_out, out_shape, b_shape, idx_b)

            broadcast_index(idx_out, out_shape, a_shape, idx_a)
#            count(pos, b_shape,  idx_b)

            out[index_to_position(idx_out, out_strides)] = fn(a_storage[index_to_position(idx_a, a_strides)] , b_storage[index_to_position(idx_b, b_strides)])



        #raise NotImplementedError('Need to implement for Task 2.2')

    return _zip



def zip(fn):
    """
    Higher-order tensor zip function ::

      fn_zip = zip(fn)
      c = fn_zip(a, b)

    Args:
        fn: function from two floats-to-float to apply
        a (:class:`TensorData`): tensor to zip over
        b (:class:`TensorData`): tensor to zip over

    Returns:
        :class:`TensorData` : new tensor data
    """

    f = tensor_zip(fn)

    def ret(a, b):
        if a.shape != b.shape:
            c_shape = shape_broadcast(a.shape, b.shape)
        else:
            c_shape = a.shape
        out = a.zeros(c_shape)
        f(*out.tuple(), *a.tuple(), *b.tuple())
        return out

    return ret


def tensor_reduce(fn):
    """
    Higher-order tensor reduce function. ::

      fn_reduce = tensor_reduce(fn)
      c = fn_reduce(out, ...)

    Args:
        fn: reduction function mapping two floats to float
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        reduce_shape (array): shape of reduction (1 for dimension kept, shape value for dimensions summed out)
        reduce_size (int): size of reduce shape

    Returns:
        None : Fills in `out`
    """
    
    def _reduce(
        out,
        out_shape,
        out_strides,
        a_storage,
        a_shape,
        a_strides,
        reduce_shape,
        reduce_size,
    ):
        

        #Out len :  3
        #Out shape :  [1 1 3]
        #Out strd :  [3 3 1]
        #A shape :  [2 1 3]
        #A strd :  [3 3 1]
        #Reduce shape :  [2, 1, 1]
        #Reduce sze :  2


        # TODO: Implement for Task 2.2.
        
        idx_out = [ 0.0 for e in out_strides]
        idx_a  =[ 0.0 for e in a_strides]

        for pos_out in range(len(out)):
            count(pos_out, out_shape, idx_out)
            for reduce_pos in range(reduce_size):
                count(reduce_pos, reduce_shape, idx_a)

                for dim in range(len(reduce_shape)):
                    if reduce_shape[dim] == 1:
                        idx_a[dim] = idx_out[dim] 
                
                pos_a = index_to_position(idx_a, a_strides)
                out[pos_out] = fn(out[pos_out], a_storage[pos_a])
    return _reduce
    #raise NotImplementedError('Need to implement for Task 2.2')

def reduce(fn, start=0.0):
    """
    Higher-order tensor reduce function. ::

      fn_reduce = reduce(fn)
      reduced = fn_reduce(a, dims)


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`TensorData`): tensor to reduce over
        dims (list, optional): list of dims to reduce
        out (:class:`TensorData`, optional): tensor to reduce into

    Returns:
        :class:`TensorData` : new tensor data
    """

    f = tensor_reduce(fn)

    # START Code Update
    def ret(a, dims=None, out=None):
        old_shape = None
        if out is None:
            out_shape = list(a.shape)
            for d in dims:
                out_shape[d] = 1
            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start
        else:
            old_shape = out.shape
            diff = len(a.shape) - len(out.shape)
            out = out.view(*([1] * diff + list(old_shape)))

        # Assume they are the same dim
        assert len(out.shape) == len(a.shape)

        # Create a reduce shape / reduce size
        reduce_shape = []
        reduce_size = 1
        for i, s in enumerate(a.shape):
            if out.shape[i] == 1:
                reduce_shape.append(s)
                reduce_size *= s
            else:
                reduce_shape.append(1)

        # Apply
        f(*out.tuple(), *a.tuple(), reduce_shape, reduce_size)

        if old_shape is not None:
            out = out.view(*old_shape)
        return out

    return ret
    # END Code Update


class TensorOps:
    map = map
    zip = zip
    reduce = reduce
