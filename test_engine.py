import engine
import genetic_components.node as n
import tensorflow as tf
import pytest
import math

@pytest.fixture
def x_tensor():
    x_size = 10
    y_size = 20
    x_size_tensor = tf.range(x_size)
    x_size_tensor = tf.reshape(x_size_tensor, [-1,1])
    x_size_tensor = tf.tile(x_size_tensor, [1, y_size])
    x_tensor = tf.cast(x_size_tensor, tf.float32)
    return x_tensor

@pytest.fixture
def y_tensor():
    x_size = 10
    y_size = 20
    y_size_tensor = tf.range(y_size)
    y_size_tensor = tf.reshape(y_size_tensor, [1,-1])
    y_size_tensor = tf.tile(y_size_tensor, [x_size, 1])
    y_tensor = tf.cast(y_size_tensor, tf.float32)
    return y_tensor



def test_abs(x_tensor, y_tensor):
    foo = n.resolve_abs_node(tf.constant(-1, shape=[10,20]))
    bar = n.resolve_abs_node(tf.constant(1, shape=[10,20]))
    ree = n.resolve_abs_node(x_tensor)
    assert (run_tensor(foo) == run_tensor(bar)).all()
    assert ree.shape == (10,20)
    assert ree.dtype == tf.float32

def test_add(x_tensor, y_tensor):
    foo = n.resolve_add_node(tf.constant(-1, shape=[10,20]), tf.constant(1, shape=[10,20]))
    bar = n.resolve_add_node(tf.constant(1, shape=[10,20]), tf.constant(-1, shape=[10,20]))
    pan = n.resolve_add_node(tf.constant(0, shape=[10,20]), tf.constant(0, shape=[10,20]))
    assert (run_tensor(foo) == run_tensor(bar)).all()
    assert (run_tensor(foo) == run_tensor(pan)).all()
    ree = n.resolve_add_node(x_tensor, y_tensor)
    assert ree.shape == (10,20)
    assert ree.dtype == tf.float32

def test_and(x_tensor, y_tensor):
    foo = n.resolve_and_node(tf.constant(1, shape=[10,20]), tf.constant(1, shape=[10,20]))
    bar = n.resolve_and_node(tf.constant(0, shape=[10,20]), tf.constant(1, shape=[10,20]))
    pan = n.resolve_and_node(tf.constant(13, shape=[10,20]), tf.constant(13, shape=[10,20]))
    ree = n.resolve_and_node(x_tensor, y_tensor)
    assert (run_tensor(foo) == run_tensor(tf.constant(1, shape=[10,20]))).all()
    assert (run_tensor(bar) == run_tensor(tf.constant(0, shape=[10,20]))).all()
    assert (run_tensor(pan) == run_tensor(tf.constant(13, shape=[10,20]))).all()
    assert ree.shape == (10,20)
    assert ree.dtype == tf.float32

def test_cos(x_tensor, y_tensor):
    foo = n.resolve_cos_node(tf.constant(0, shape=[10,20], dtype=tf.float32))
    bar = n.resolve_cos_node(tf.constant(math.pi, shape=[10,20], dtype=tf.float32))
    pan = tf.cast(n.resolve_cos_node(tf.constant(math.pi / 2, shape=[10,20], dtype=tf.float32)), tf.int32)
    ree = n.resolve_cos_node(x_tensor)
    assert (run_tensor(foo) == run_tensor(tf.constant(1, shape=[10,20]))).all()
    assert (run_tensor(bar) == run_tensor(tf.constant(-1, shape=[10,20]))).all()
    assert (run_tensor(pan) == run_tensor(tf.constant(0, shape=[10,20]))).all()
    assert ree.shape == (10,20)
    assert ree.dtype == tf.float32

def test_div(x_tensor, y_tensor):
    foo = n.resolve_div_node(tf.constant(13, shape=[10,20]), tf.constant(13, shape=[10,20]))
    bar = n.resolve_div_node(tf.constant(1, shape=[10,20]), tf.constant(0, shape=[10,20]))
    pan = n.resolve_div_node(tf.constant(0, shape=[10,20]), tf.constant(1, shape=[10,20]))
    assert (run_tensor(foo) == run_tensor(tf.constant(1, shape=[10,20]))).all()
    assert (run_tensor(bar) == run_tensor(tf.constant(0, shape=[10,20]))).all()
    assert (run_tensor(pan) == run_tensor(tf.constant(0, shape=[10,20]))).all()
    ree = n.resolve_div_node(x_tensor, y_tensor)
    assert ree.shape == (10,20)
    assert ree.dtype == tf.float32

def test_exp(x_tensor, y_tensor):
    foo = n.resolve_exp_node(tf.constant(0, shape=[10,20], dtype=tf.float32))
    bar = n.resolve_exp_node(tf.constant(1, shape=[10,20], dtype=tf.float32))
    pan = n.resolve_exp_node(tf.constant(-1, shape=[10,20], dtype=tf.float32))
    ree = n.resolve_exp_node(x_tensor)
    assert (run_tensor(foo) == run_tensor(tf.constant(1, shape=[10,20]))).all()
    assert (run_tensor(bar) == run_tensor(tf.constant(math.e, shape=[10,20]))).all()
    assert (run_tensor(pan) == run_tensor(tf.constant(1 / math.e, shape=[10,20]))).all()
    assert ree.shape == (10,20)
    assert ree.dtype == tf.float32

def test_if(x_tensor, y_tensor):
    aux1 = tf.convert_to_tensor([[1.0, 2.0], [4.0, 5.0]])
    aux2 = tf.constant(3, shape=[2,2], dtype=tf.float32)
    foo = n.resolve_if_node(tf.constant(1,shape=[2,2], dtype=tf.float32), tf.constant(0,shape=[2,2], dtype=tf.float32), tf.cast(tf.math.greater(aux1, aux2), tf.float32), 2, 2)
    assert (run_tensor(foo) == run_tensor(tf.convert_to_tensor([[0,0], [1,1]]))).all()
    ree = n.resolve_if_node(x_tensor, y_tensor, x_tensor, 10, 20)
    assert ree.shape == (10,20)
    assert ree.dtype == tf.float32

def test_log(x_tensor, y_tensor):
    foo = n.resolve_log_node(tf.constant(1, shape=[10,20], dtype=tf.float32))
    bar = n.resolve_log_node(tf.constant(math.e, shape=[10,20], dtype=tf.float32))
    ree = n.resolve_log_node(x_tensor)
    assert (run_tensor(foo) == run_tensor(tf.constant(0, shape=[10,20]))).all()
    assert (run_tensor(bar) == run_tensor(tf.constant(1, shape=[10,20]))).all()
    assert ree.shape == (10,20)
    assert ree.dtype == tf.float32

def test_max(x_tensor, y_tensor):
    foo = n.resolve_max_node(tf.constant(13, shape=[10,20]), tf.constant(13, shape=[10,20]))
    bar = n.resolve_max_node(tf.constant(1, shape=[10,20]), tf.constant(0, shape=[10,20]))
    aux1 = tf.convert_to_tensor([[1.0, 2.0], [4.0, 5.0]])
    aux2 = tf.constant(3, shape=[2,2], dtype=tf.float32)
    pan = n.resolve_max_node(aux1, aux2)
    assert (run_tensor(foo) == run_tensor(tf.constant(13, shape=[10,20]))).all()
    assert (run_tensor(bar) == run_tensor(tf.constant(1, shape=[10,20]))).all()
    assert (run_tensor(pan) == run_tensor(tf.convert_to_tensor([[3,3],[4,5]]))).all()
    ree = n.resolve_max_node(x_tensor, y_tensor)
    assert ree.shape == (10,20)
    assert ree.dtype == tf.float32

def test_mdist(x_tensor, y_tensor):
    foo = n.resolve_mdist_node(tf.constant(0, shape=[10,20]), tf.constant(2, shape=[10,20]), 10, 20)
    bar = n.resolve_mdist_node(tf.constant(1, shape=[10,20]), tf.constant(1, shape=[10,20]), 10, 20)
    assert (run_tensor(foo) == run_tensor(tf.constant(1, shape=[10,20]))).all()
    assert (run_tensor(bar) == run_tensor(tf.constant(1, shape=[10,20]))).all()
    ree = n.resolve_mdist_node(x_tensor, y_tensor, 10, 20)
    assert ree.shape == (10,20)
    assert ree.dtype == tf.float32

def test_min(x_tensor, y_tensor):
    foo = n.resolve_min_node(tf.constant(13, shape=[10,20]), tf.constant(13, shape=[10,20]))
    bar = n.resolve_min_node(tf.constant(1, shape=[10,20]), tf.constant(0, shape=[10,20]))
    aux1 = tf.convert_to_tensor([[1.0, 2.0], [4.0, 5.0]])
    aux2 = tf.constant(3, shape=[2,2], dtype=tf.float32)
    pan = n.resolve_min_node(aux1, aux2)
    assert (run_tensor(foo) == run_tensor(tf.constant(13, shape=[10,20]))).all()
    assert (run_tensor(bar) == run_tensor(tf.constant(0, shape=[10,20]))).all()
    assert (run_tensor(pan) == run_tensor(tf.convert_to_tensor([[1,2],[3,3]]))).all()
    ree = n.resolve_min_node(x_tensor, y_tensor)
    assert ree.shape == (10,20)
    assert ree.dtype == tf.float32


def test_mod(x_tensor, y_tensor):
    foo = n.resolve_mod_node(tf.constant(4, shape=[10,20]), tf.constant(2, shape=[10,20]))
    bar = n.resolve_mod_node(tf.constant(4, shape=[10,20]), tf.constant(3, shape=[10,20]))
    assert (run_tensor(foo) == run_tensor(tf.constant(0, shape=[10,20]))).all()
    assert (run_tensor(bar) == run_tensor(tf.constant(1, shape=[10,20]))).all()
    ree = n.resolve_mod_node(x_tensor, y_tensor)
    assert ree.shape == (10,20)
    assert ree.dtype == tf.float32

def test_mult(x_tensor, y_tensor):
    foo = n.resolve_mult_node(tf.constant(13, shape=[10,20], dtype=tf.float32), tf.constant(1.2, shape=[10,20]))
    bar = n.resolve_mult_node(tf.constant(1, shape=[10,20]), tf.constant(0, shape=[10,20]))
    pan = n.resolve_mult_node(tf.constant(3, shape=[10,20]), tf.constant(1, shape=[10,20]))
    assert (run_tensor(foo) == run_tensor(tf.constant(15.6, shape=[10,20]))).all()
    assert (run_tensor(bar) == run_tensor(tf.constant(0, shape=[10,20]))).all()
    assert (run_tensor(pan) == run_tensor(tf.constant(3, shape=[10,20]))).all()
    ree = n.resolve_mult_node(x_tensor, y_tensor)
    assert ree.shape == (10,20)
    assert ree.dtype == tf.float32

def test_neg(x_tensor, y_tensor):
    foo = n.resolve_neg_node(tf.constant(1, shape=[10,20], dtype=tf.float32))
    bar = n.resolve_neg_node(tf.constant(-1, shape=[10,20], dtype=tf.float32))
    ree = n.resolve_neg_node(x_tensor)
    assert (run_tensor(foo) == run_tensor(tf.constant(-1, shape=[10,20]))).all()
    assert (run_tensor(bar) == run_tensor(tf.constant(1, shape=[10,20]))).all()
    assert ree.shape == (10,20)
    assert ree.dtype == tf.float32

def test_or(x_tensor, y_tensor):
    foo = n.resolve_or_node(tf.constant(1, shape=[10,20]), tf.constant(1, shape=[10,20]))
    bar = n.resolve_or_node(tf.constant(0, shape=[10,20]), tf.constant(1, shape=[10,20]))
    tur = n.resolve_or_node(tf.constant(0, shape=[10,20]), tf.constant(0, shape=[10,20]))
    pan = n.resolve_or_node(tf.constant(13, shape=[10,20]), tf.constant(13, shape=[10,20]))
    ree = n.resolve_or_node(x_tensor, y_tensor)
    assert (run_tensor(foo) == run_tensor(tf.constant(1, shape=[10,20]))).all()
    assert (run_tensor(bar) == run_tensor(tf.constant(1, shape=[10,20]))).all()
    assert (run_tensor(tur) == run_tensor(tf.constant(0, shape=[10,20]))).all()
    assert (run_tensor(pan) == run_tensor(tf.constant(13, shape=[10,20]))).all()
    assert ree.shape == (10,20)
    assert ree.dtype == tf.float32

def test_pow(x_tensor, y_tensor):
    foo = n.resolve_pow_node(tf.constant(0, tf.float32, shape=[10,20]), tf.constant(1, tf.float32, shape=[10,20]))
    bar = n.resolve_pow_node(tf.constant(5, tf.float32, shape=[10,20]), tf.constant(0, tf.float32, shape=[10,20]))
    pan = n.resolve_pow_node(tf.constant(3, tf.float32, shape=[10,20]), tf.constant(-1, tf.float32, shape=[10,20]))
    tun = n.resolve_pow_node(tf.constant(3, tf.float32, shape=[10,20]), tf.constant(2, tf.float32, shape=[10,20]))
    assert (run_tensor(foo) == run_tensor(tf.constant(0, shape=[10,20]))).all()
    assert (run_tensor(bar) == run_tensor(tf.constant(1, shape=[10,20]))).all()
    assert (run_tensor(pan) == run_tensor(tf.constant(3, shape=[10,20]))).all()
    assert (run_tensor(tun) == run_tensor(tf.constant(9, shape=[10,20]))).all()
    ree = n.resolve_pow_node(x_tensor, y_tensor)
    assert ree.shape == (10,20)
    assert ree.dtype == tf.float32

def test_sign(x_tensor, y_tensor):
    foo = n.resolve_sign_node(tf.constant(3, shape=[10,20], dtype=tf.float32))
    bar = n.resolve_sign_node(tf.constant(-4.3, shape=[10,20], dtype=tf.float32))
    pan = n.resolve_sign_node(tf.constant(-0, shape=[10,20], dtype=tf.float32))
    ree = n.resolve_sign_node(x_tensor)
    assert (run_tensor(foo) == run_tensor(tf.constant(1, shape=[10,20]))).all()
    assert (run_tensor(bar) == run_tensor(tf.constant(-1, shape=[10,20]))).all()
    assert (run_tensor(pan) == run_tensor(tf.constant(0, shape=[10,20]))).all()
    assert ree.shape == (10,20)
    assert ree.dtype == tf.float32

def test_sin(x_tensor, y_tensor):
    foo = n.resolve_sin_node(tf.constant(0, shape=[10,20], dtype=tf.float32))
    bar = tf.cast(n.resolve_sin_node(tf.constant(math.pi, shape=[10,20], dtype=tf.float32)), tf.int32)
    pan = tf.cast(n.resolve_sin_node(tf.constant(math.pi / 2, shape=[10,20], dtype=tf.float32)), tf.int32)
    ree = n.resolve_sin_node(x_tensor)
    assert (run_tensor(foo) == run_tensor(tf.constant(0, shape=[10,20]))).all()
    assert (run_tensor(bar) == run_tensor(tf.constant(0, shape=[10,20]))).all()
    assert (run_tensor(pan) == run_tensor(tf.constant(1, shape=[10,20]))).all()
    assert ree.shape == (10,20)
    assert ree.dtype == tf.float32

def test_sqrt(x_tensor, y_tensor):
    foo = n.resolve_sqrt_node(tf.constant(0, tf.float32, shape=[10,20]), 10, 20)
    pan = n.resolve_sqrt_node(tf.constant(-1, tf.float32, shape=[10,20]), 10, 20)
    tun = n.resolve_sqrt_node(tf.constant(6.25, tf.float32, shape=[10,20]), 10, 20)
    assert (run_tensor(foo) == run_tensor(tf.constant(0, shape=[10,20]))).all()
    assert (run_tensor(pan) == run_tensor(tf.constant(0, shape=[10,20]))).all()
    assert (run_tensor(tun) == run_tensor(tf.constant(2.5, shape=[10,20]))).all()
    ree = n.resolve_sqrt_node(x_tensor, 10, 20)
    assert ree.shape == (10,20)
    assert ree.dtype == tf.float32

def test_sub(x_tensor, y_tensor):
    foo = n.resolve_sub_node(tf.constant(13, shape=[10,20]), tf.constant(13, shape=[10,20]))
    bar = n.resolve_sub_node(tf.constant(1, shape=[10,20]), tf.constant(0, shape=[10,20]))
    pan = n.resolve_sub_node(tf.constant(-1, shape=[10,20]), tf.constant(-2, shape=[10,20]))
    assert (run_tensor(foo) == run_tensor(tf.constant(0, shape=[10,20]))).all()
    assert (run_tensor(bar) == run_tensor(tf.constant(1, shape=[10,20]))).all()
    assert (run_tensor(pan) == run_tensor(tf.constant(1, shape=[10,20]))).all()
    ree = n.resolve_sub_node(x_tensor, y_tensor)
    assert ree.shape == (10,20)
    assert ree.dtype == tf.float32
 
def test_tan(x_tensor, y_tensor):
    bar = tf.cast(n.resolve_tan_node(tf.constant(0, shape=[10,20], dtype=tf.float32)), tf.int32)
    pan = tf.cast(n.resolve_tan_node(tf.constant(math.pi / 4, shape=[10,20], dtype=tf.float32)), tf.int32)
    ree = n.resolve_sin_node(x_tensor)
    assert (run_tensor(bar) == run_tensor(tf.constant(0, shape=[10,20]))).all()
    assert (run_tensor(pan) == run_tensor(tf.constant(1, shape=[10,20]))).all()
    assert ree.shape == (10,20)
    assert ree.dtype == tf.float32

def test_xor(x_tensor, y_tensor):
    foo = n.resolve_xor_node(tf.constant(1, shape=[10,20]), tf.constant(1, shape=[10,20]))
    bar = n.resolve_xor_node(tf.constant(0, shape=[10,20]), tf.constant(1, shape=[10,20]))
    tur = n.resolve_xor_node(tf.constant(0, shape=[10,20]), tf.constant(0, shape=[10,20]))
    pan = n.resolve_xor_node(tf.constant(1, shape=[10,20]), tf.constant(0, shape=[10,20]))
    ree = n.resolve_xor_node(x_tensor, y_tensor)
    assert (run_tensor(foo) == run_tensor(tf.constant(0, shape=[10,20]))).all()
    assert (run_tensor(bar) == run_tensor(tf.constant(1, shape=[10,20]))).all()
    assert (run_tensor(tur) == run_tensor(tf.constant(0, shape=[10,20]))).all()
    assert (run_tensor(pan) == run_tensor(tf.constant(1, shape=[10,20]))).all()
    assert ree.shape == (10,20)
    assert ree.dtype == tf.float32 

def run_tensor(tensor):
    sess = tf.compat.v1.Session()
    result = sess.run(tensor)
    sess.close()
    return result

def test_engine():
    assert engine.engine(5, 5, 3, 0.2, 0.9, [10,10], 0)

if __name__ == '__main__':
    x_size = 10
    y_size = 20
    x_size_tensor = tf.range(x_size)
    x_size_tensor = tf.reshape(x_size_tensor, [-1,1])
    x_size_tensor = tf.tile(x_size_tensor, [1, y_size])
    x_tensor = tf.cast(x_size_tensor, tf.float32)

    y_size_tensor = tf.range(y_size)
    y_size_tensor = tf.reshape(y_size_tensor, [1,-1])
    y_size_tensor = tf.tile(y_size_tensor, [x_size, 1])
    y_tensor = tf.cast(y_size_tensor, tf.float32)
    test_engine()