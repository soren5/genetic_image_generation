import tensorflow as tf
special_case_3_child = {'if'}
special_case_1_child = {'abs', 'cos', 'sin', 'tan', 'neg', 'exp', 'log', 'sign'}

def resolve_x_node(x_size, y_size):
    x_size_tensor = tf.range(x_size)
    x_size_tensor = tf.reshape(x_size_tensor, [-1,1])
    x_size_tensor = tf.tile(x_size_tensor, [1, y_size])
    x_size_tensor = tf.cast(x_size_tensor, tf.float32)
    return x_size_tensor

def resolve_y_node(x_size, y_size):
    y_size_tensor = tf.range(y_size)
    y_size_tensor = tf.reshape(y_size_tensor, [1,-1])
    y_size_tensor = tf.tile(y_size_tensor, [x_size, 1])
    y_size_tensor = tf.cast(y_size_tensor, tf.float32)
    return y_size_tensor 

def resolve_constant_node(x_size, y_size):
    terminal_tensor = tf.fill([x_size,y_size], self.value)
    terminal_tensor = tf.cast(terminal_tensor, tf.float32)
    return terminal_tensor

def resolve_abs_node(child1):
    return tf.math.abs(child1)

def resolve_add_node(child1, child2):
    return tf.math.add(child1, child2)

def resolve_and_node(child1, child2) :
    left_child_tensor = tf.cast(child1, tf.uint8)
    right_child_tensor = tf.cast(child2, tf.uint8)
    return tf.cast(tf.bitwise.bitwise_and(left_child_tensor, right_child_tensor), tf.float32)

def resolve_cos_node(child1) :
    return tf.math.cos(child1)

def resolve_div_node(child1, child2) :
    left_child_tensor = tf.cast(child1, tf.float32)
    right_child_tensor = tf.cast(child2, tf.float32)
    return tf.math.divide_no_nan(left_child_tensor, right_child_tensor)
    
def resolve_exp_node(child1) :
    return tf.math.exp(child1)

def resolve_if_node(child1, child2, child3, x_size, y_size):
    return tf.where(
        tf.cast(child3, tf.bool),
        child1,
        child2)
        
def resolve_log_node(child1):
    return tf.math.log(child1)

def resolve_max_node(child1, child2):
    return tf.math.maximum(child1, child2)

def resolve_mdist_node(child1, child2, x_size, y_size):
    return tf.math.divide(tf.cast(tf.math.add(child1, child2), tf.float32), tf.constant(2, tf.float32, shape=[x_size, y_size]))

def resolve_min_node(child1, child2):
    return tf.math.minimum(child1, child2)
    
def resolve_mod_node(child1, child2):
    return tf.math.mod(child1, child2)

def resolve_mult_node(child1, child2):
    return tf.math.multiply(child1, child2)

def resolve_neg_node(child1):
    return tf.math.negative(child1)

def resolve_or_node(child1, child2):
    left_child_tensor = tf.cast(child1, tf.uint8)
    right_child_tensor = tf.cast(child2, tf.uint8)
    return tf.cast(tf.bitwise.bitwise_or(left_child_tensor, right_child_tensor), tf.float32)

def resolve_pow_node(child1, child2):
    return tf.math.pow(tf.math.abs(child1), tf.math.abs(child2))

def resolve_scalarT_node(child1):
    #TODO
    return child1

def resolve_scalarV_node(child1):
    #TODO
    return child1

def resolve_sign_node(child1):
    return tf.math.divide_no_nan(
        tf.math.abs(child1),
        child1)

def resolve_sin_node(child1):
    return tf.math.sin(child1)

def resolve_sqrt_node(child1, x_size, y_size):
    return tf.where(child1 > 0, tf.math.sqrt(child1), tf.constant(0, tf.float32, [x_size, y_size]))

def resolve_sub_node(child1, child2):
    return tf.math.subtract(child1, child2)

def resolve_tan_node(child1):
    return tf.math.tan(child1)

def resolve_warp_node(child1, child2):
    #TODO
    return child1

def resolve_xor_node(child1, child2):
    left_child_tensor = tf.cast(child1, tf.uint8)
    right_child_tensor = tf.cast(child2, tf.uint8)
    return tf.cast(tf.bitwise.bitwise_xor(left_child_tensor, right_child_tensor), tf.float32)

class Node:
    def __init__(self, value, children, terminal):
        self.value = value
        self.children = children
        self.terminal = terminal

    def get_tensor(self, x_size, y_size):
        if self.terminal:
            if self.value == 'x':
                return resolve_x_node(x_size, y_size)
            elif self.value == 'y':
                return resolve_y_node(x_size, y_size)
            else:
                return resolve_constant_node(x_size, y_size)
        else:
            child1 = None
            child2 = None
            child3 = None

            try: 
                child1 = self.children[0].get_tensor(x_size, y_size)
            except:
                pass
            try: 
                child2 = self.children[1].get_tensor(x_size, y_size)
            except:
                pass
            try: 
                child3 = self.children[2].get_tensor(x_size, y_size)
            except:
                pass

            if self.value == 'abs':
                return resolve_abs_node(child1)
            if self.value == 'add':
                return resolve_add_node(child1, child2)
            if self.value == 'and' :
                return resolve_and_node(child1, child2)
            if self.value == 'cos' :
                return resolve_cos_node(child1)
            if self.value == 'div' :
                return resolve_div_node(child1, child2)
            if self.value == 'exp' :
                return resolve_exp_node(child1)
            if self.value == 'if':
                return resolve_if_node(child1, child2, child3, x_size, y_size)
            if self.value == 'log':
                return resolve_log_node(child1)
            if self.value == 'max':
                return resolve_max_node(child1, child2)
            if self.value == 'mdist':
                return resolve_mdist_node(child1, child2, x_size, y_size)
            if self.value == 'min':
                return resolve_min_node(child1, child2)
            if self.value == 'mod':
                return resolve_mod_node(child1, child2)
            if self.value == 'mult':
                return resolve_mult_node(child1, child2)
            if self.value == 'neg':
                return resolve_neg_node(child1)
            if self.value == 'or':
                return resolve_or_node(child1, child2)
            if self.value == 'pow':
                return resolve_pow_node(child1, child2)
            if self.value == 'scalarT':
                #TODO
                return resolve_scalarT_node(child1)
            if self.value == 'scalarV':
                #TODO
                return resolve_scalarV_node(child1)
            if self.value == 'sign':
                return resolve_sign_node(child1)
            if self.value == 'sin':
                return resolve_sin_node(child1)
            if self.value == 'sqrt':
                return resolve_sqrt_node(child1, x_size, y_size)
            if self.value == 'sub':
                return resolve_sub_node(child1, child2)
            if self.value == 'tan':
                return resolve_tan_node(child1)
            if self.value == 'warp':
                #TODO
                return resolve_warp_node(child1, child2)
            if self.value == 'xor':
                return resolve_xor_node(child1, child2)


    def get_string(self):
        if self.terminal:
            return str(self.value)
        elif self.value == '/':
            return 'prot_div(' + self.children[0].get_string() + ', ' + self.children[1].get_string() + ')'
        elif self.value in special_case_3_child:
            return 'if_func(' + self.children[2].get_string() + ', ' + self.children[0].get_string() + ', ' + self.children[1].get_string() + ')'
        elif self.value in special_case_1_child:
            return '(' + self.children[0].get_string() + ')'
        else:
            return str(self.value) + '(' + self.children[0].get_string() + ', ' + self.children[1].get_string() + ')'

    def get_depth(self, depth=1):
        if self.terminal:
            return depth
        else:
            max_depth = 0
            for i in self.children:
                child_depth = i.get_depth(depth + 1)
                if max_depth < child_depth:
                    max_depth = child_depth
            return max_depth
