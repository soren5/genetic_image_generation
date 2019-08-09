import tensorflow as tf
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
                return x_size_tensor
            elif self.value == 'y':
                y_size_tensor = tf.range(y_size)
                y_size_tensor = tf.reshape(y_size_tensor, [1,-1])
                y_size_tensor = tf.tile(y_size_tensor, [y_size, 1])
                return y_size_tensor 
            else:
                terminal_tensor = tf.fill([x_size,y_size], self.value)
                return terminal_tensor
        else:
            if self.value == '+':
                return tf.math.add(self.children[0].get_tensor(x_size, y_size), self.children[1].get_tensor(x_size, y_size))
            if self.value == '-':
                return tf.math.subtract(self.children[0].get_tensor(x_size, y_size), self.children[1].get_tensor(x_size, y_size))
            if self.value == '/':
                left_child_tensor = self.children[0].get_tensor(x_size, y_size)
                right_child_tensor = self.children[1].get_tensor(x_size, y_size)
                return tf.cast(tf.math.divide_no_nan(left_child_tensor, right_child_tensor), tf.int32)
            if self.value == '*':
                return tf.math.multiply(self.children[0].get_tensor(x_size, y_size), self.children[1].get_tensor(x_size, y_size))
            if self.value == '^':
                return tf.bitwise.bitwise_xor(self.children[0].get_tensor(x_size, y_size), self.children[1].get_tensor(x_size, y_size))

    def get_string(self):
        if self.terminal:
            return str(self.value)
        elif self.value == '/':
            return 'prot_div(' + self.children[0].get_string() + ', ' + self.children[1].get_string() + ')'
        elif self.value == 'if':
            return 'if_func(' + self.children[2].get_string() + ', ' + self.children[0].get_string() + ', ' + self.children[1].get_string() + ')'
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
