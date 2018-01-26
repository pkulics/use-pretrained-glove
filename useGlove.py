# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

#embedding_size = 300
def embLayer(input):
    with open('n_GloVe.txt', 'r') as file1:
        emb = []
        vocab = []
        for line in file1.readlines():
            row = line.strip().split(' ')
            vocab.append(row[0])
            emb.append(row[1:])

    emb = np.asarray(emb, dtype="float32")
    with tf.variable_scope('embedding'):
        embedding = tf.Variable(emb, name='emb')
        embeding_input = tf.nn.embedding_lookup(embedding, input)

        return embeding_input


input = [1,2]
emb= embLayer([1,2])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    out = sess.run(emb)
print out