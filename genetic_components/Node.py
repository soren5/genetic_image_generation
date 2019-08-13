import tensorflow as tf
special_case_3_child = {'if'}
special_case_1_child = {'cos', 'sin', 'tan', 'invert'}
class Node:
    def __init__(self, value, children, terminal):
        self.value = value
        self.children = children
        self.terminal = terminal

    def get_tensor(self, x_size, y_size):
        if self.terminal:
            if self.value == 'x':
                x_size_tensor = tf.range(x_size)
                x_size_tensor = tf.reshape(x_size_tensor, [-1,1])
                x_size_tensor = tf.tile(x_size_tensor, [1,x_size])
                x_size_tensor = tf.cast(x_size_tensor, tf.float32)
                return x_size_tensor
            elif self.value == 'y':
                y_size_tensor = tf.range(y_size)
                y_size_tensor = tf.reshape(y_size_tensor, [1,-1])
                y_size_tensor = tf.tile(y_size_tensor, [y_size, 1])
                y_size_tensor = tf.cast(y_size_tensor, tf.float32)
                return y_size_tensor 
            else:
                terminal_tensor = tf.fill([x_size,y_size], self.value)
                terminal_tensor = tf.cast(terminal_tensor, tf.float32)
                return terminal_tensor
        else:
            if self.value == '+':
                return tf.math.add(self.children[0].get_tensor(x_size, y_size), self.children[1].get_tensor(x_size, y_size))
            if self.value == '-':
                return tf.math.subtract(self.children[0].get_tensor(x_size, y_size), self.children[1].get_tensor(x_size, y_size))
            if self.value == '/':
                left_child_tensor = tf.cast(self.children[0].get_tensor(x_size, y_size), tf.float32)
                right_child_tensor = tf.cast(self.children[1].get_tensor(x_size, y_size), tf.float32)
                return tf.math.divide_no_nan(left_child_tensor, right_child_tensor)
            if self.value == '*':
                return tf.math.multiply(self.children[0].get_tensor(x_size, y_size), self.children[1].get_tensor(x_size, y_size))
            if self.value == 'cos':
                return tf.math.cos(self.children[0].get_tensor(x_size, y_size))
            if self.value == 'sin':
                return tf.math.sin(self.children[0].get_tensor(x_size, y_size))
            if self.value == 'tan':
                return tf.math.tan(self.children[0].get_tensor(x_size, y_size))
            left_child_tensor = tf.cast(self.children[0].get_tensor(x_size, y_size), tf.uint8)
            if self.value == 'invert':
                return tf.cast(tf.bitwise.invert(left_child_tensor), tf.float32)
            right_child_tensor = tf.cast(self.children[1].get_tensor(x_size, y_size), tf.uint8)
            if self.value == '^':
                return tf.cast(tf.bitwise.bitwise_xor(left_child_tensor, right_child_tensor), tf.float32)
            if self.value == 'and':
                return tf.cast(tf.bitwise.bitwise_and(left_child_tensor, right_child_tensor), tf.float32)
            if self.value == 'or':
                return tf.cast(tf.bitwise.bitwise_or(left_child_tensor, right_child_tensor), tf.float32)
            


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
            return '(' + self.children[0].get_string() + str(self.value) + self.children[1].get_string() + ')'

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
