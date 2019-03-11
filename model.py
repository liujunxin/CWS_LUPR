#coding=utf-8
import tensorflow as tf
import numpy as np
import collections
import pickle as pkl
from utils import *
import os
class CWScnncrf(object):
    def __init__(self, config):
        self.vocab_size = config.vocab_size#0:PAD，1:UNK
        self.emb_dim = config.emb_dim
        self.pro_dim = config.pro_dim
        self.output_dim = config.output_dim#label class num
        self.use_pretrain_emb = config.use_pretrain_emb
        self.pretrain_emb = config.pretrain_emb
        self.emb_trainable = config.emb_trainable
        self.batch_size = config.batch_size
        self.reg = config.reg
        self.maxlen = config.maxlen
        self.filter_sizes = config.filter_sizes
        self.num_filters = config.num_filters
        self.clip_norm = config.clip_norm
        self.dropSet = config.dropSet
        self.config = config

        self.add_placeholders()

        self.add_embedding()

        self.ch_embs = self.get_embedding(self.input_sens)
        self.ch_embs = self.ch_embs * self.mask[:,:,None]

        self.add_model_variables()

        self.calc_batch_loss()
    def add_placeholders(self):
        self.input_sens = tf.placeholder(tf.int32, [None, None], name='input_sens')
        self.seqlen = tf.placeholder(tf.int32, [None], name='seqlen')
        self.labels = tf.placeholder(tf.int32, [None, None], name='labels')
        self.mask = tf.placeholder(tf.float32, [None, None], name='mask')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.batch_s = tf.placeholder(tf.int32, shape=[], name='batch_s')
        self.loss_weight = tf.placeholder(tf.float32, shape=[None], name='loss_weight')
    def add_embedding(self):
        with tf.variable_scope('Embed', regularizer=None):
            if self.use_pretrain_emb:
                char_embedding = tf.get_variable('char_embedding', shape=[self.vocab_size, self.emb_dim],
                                            initializer=tf.constant_initializer(self.pretrain_emb),
                                            trainable=self.emb_trainable, regularizer=None)
            else:
                char_embedding = tf.get_variable('char_embedding', shape=[self.vocab_size, self.emb_dim],
                                            initializer=tf.random_uniform_initializer(-1,1),
                                            trainable=self.emb_trainable, regularizer=None)
    def get_embedding(self, input_sens):
        with tf.variable_scope('Embed', regularizer=None, reuse=True):
            char_embedding = tf.get_variable('char_embedding', shape=[self.vocab_size, self.emb_dim])
            ch_embs = tf.nn.embedding_lookup(char_embedding, input_sens)
            return ch_embs
    def calc_wt_init(self, fan_in=300):#for xavier_initializer
        eps=1.0/np.sqrt(fan_in)
        return eps
    def add_model_variables(self):
        self.densedim = self.num_filters * len(self.filter_sizes)
        with tf.variable_scope('Encode', regularizer=None):
            for ii in range(len(self.filter_sizes)):
                hsize = self.emb_dim
                conv_W = tf.get_variable(('conv_W%d' % ii), [self.filter_sizes[ii], hsize, 1, self.num_filters],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
                conv_b = tf.get_variable(('conv_b%d' % ii), [self.num_filters],initializer=
                                             tf.constant_initializer(0.), regularizer=tf.contrib.layers.l2_regularizer(0.0))
        with tf.variable_scope('Decode', regularizer=None):
            out_W = tf.get_variable('out_W', [self.densedim, self.output_dim],
                                initializer=tf.random_uniform_initializer(-self.calc_wt_init(self.densedim),self.calc_wt_init(self.densedim)))
            out_b = tf.get_variable('out_b', [self.output_dim],
                                   initializer=tf.constant_initializer(0.0),regularizer=tf.contrib.layers.l2_regularizer(0.0))
    def calc_batch_loss(self):
        if 'emb' in self.dropSet and self.config.keep_prob < 1:
            self.ch_embs = tf.nn.dropout(self.ch_embs, self.keep_prob)

        cnn_outputs = []
        with tf.variable_scope('Encode', reuse=True):
            cnn_input = self.ch_embs
            encode_h = tf.expand_dims(cnn_input, -1)
            for ii in range(len(self.filter_sizes)):
                filter_size = self.filter_sizes[ii]
                hsize = self.emb_dim
                filter_shape = [filter_size, hsize, 1, self.num_filters]
                conv_W = tf.get_variable(('conv_W%d' % ii), filter_shape)
                conv_b = tf.get_variable(('conv_b%d' % ii), [self.num_filters])
                conv_in = tf.concat([tf.zeros([self.batch_s, filter_size//2, hsize, 1]), encode_h, tf.zeros([self.batch_s, filter_size-filter_size//2-1, hsize, 1])], axis=1)
                conv_out = tf.nn.conv2d(conv_in, conv_W, strides=[1, 1, 1, 1], padding='VALID')
                conv_out = tf.nn.bias_add(conv_out, conv_b)
                conv_out = tf.nn.relu(conv_out)
                conv_out = tf.reshape(conv_out, [self.batch_s, tf.shape(self.ch_embs)[1], self.num_filters])
                cnn_outputs.append(conv_out)
        cnn_outputs = tf.concat(cnn_outputs, axis=2)
        print(tf.shape(cnn_outputs))

        if 'cnn' in self.dropSet and self.config.keep_prob < 1:
            cnn_outputs = tf.nn.dropout(cnn_outputs, self.keep_prob)

        h2dense = tf.reshape(cnn_outputs, [-1, self.densedim])
        with tf.variable_scope("Decode",reuse=True):
            out_W = tf.get_variable('out_W', [self.densedim, self.output_dim])
            out_b = tf.get_variable('out_b', [self.output_dim])
            self.unary_scores = tf.matmul(h2dense, out_W) + out_b
            self.unary_scores = tf.reshape(self.unary_scores, [self.batch_s, -1, self.output_dim])
        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.unary_scores, self.labels, self.seqlen)
        self.loss = tf.reduce_mean(-self.log_likelihood*self.loss_weight)
        self.train_step = tf.train.RMSPropOptimizer(self.config.lr).minimize(self.loss)

    def train(self,
                trainsens,
                trainlabel,
                trainweight,
                validsens,
                validlabel,
                testsens,
                testlabel,
                sess,
                saver,
                dispFreq=10,
                validFreq=500,
                savepath='./ckpt/cnncrf'):
        update = 0
        printloss = 0
        best_valid_loss = 1000
        best_valid_epoch = 0
        for i in range(self.config.max_epochs):
            r = np.random.permutation(len(trainsens))
            trainsens = [trainsens[ii] for ii in r]
            trainlabel = [trainlabel[ii] for ii in r]
            trainweight = [trainweight[ii] for ii in r]
            for ii in range(0, len(trainsens), self.batch_size):
                endidx = min(ii+self.batch_size, len(trainsens))
                if endidx <= ii:
                    break
                xx, ll, mm, seql = prepare_data(trainsens[ii:endidx], trainlabel[ii:endidx])
                lw = np.asarray(trainweight[ii:endidx])
                batch_ss = xx.shape[0]
                feed_dict = {self.input_sens: xx, self.mask: mm, self.seqlen: seql, self.labels: ll, self.keep_prob: self.config.keep_prob, self.batch_s: batch_ss, self.loss_weight: lw}
                result = sess.run([self.loss, self.train_step], feed_dict=feed_dict)
                update += 1
                printloss += result[0]
                if update % dispFreq == 0:
                    print("Epoch:\t%d\tUPdate:\t%d\tloss:\t%f" % (i, update, printloss/dispFreq))
                    printloss = 0
            #valid
            print('valid begin!')
            print('valid:')
            validloss = self.evaluate(validsens, validlabel, sess)
            print('test begin!')
            print('test:')
            self.evaluate(testsens, testlabel, sess)
            if validloss < best_valid_loss:
                print('save the best model!')
                best_valid_loss = validloss
                best_valid_epoch = i
                saver.save(sess, '%s_best.ckpt' % savepath)

            if i - best_valid_epoch >= self.config.early_stopping:
                print('early stop!')
                break
        saver.save(sess,'%s_final.ckpt' % savepath)

    def evaluate(self, validsens, validlabel, sess):
        predicts = []
        validloss = []
        for ii in range(0, len(validsens), self.batch_size):
            endidx = min(ii+self.batch_size, len(validsens))
            if endidx <= ii:
                break
            xx, ll, mm, seql = prepare_data(validsens[ii:endidx], validlabel[ii:endidx])
            lw = np.asarray([1. for jj in range(endidx-ii)])
            batch_ss = xx.shape[0]
            feed_dict = {self.input_sens: xx, self.mask: mm, self.seqlen: seql, self.labels: ll, self.keep_prob: 1, self.batch_s: batch_ss, self.loss_weight: lw}
            loss, unary_scores, transition_params = sess.run([self.loss, self.unary_scores, self.transition_params], feed_dict=feed_dict)
            pp = []
            for us_, sl_ in zip(unary_scores, seql):
                us_ = us_[:sl_]
                viterbi_seq, _ = tf.contrib.crf.viterbi_decode(us_, transition_params)
                pp.append(viterbi_seq)
            predicts.extend(pp)
            validloss.append(loss)
        validloss = np.asarray(validloss)
        validloss = np.mean(validloss)
        accuracy, recall, precision, fscore = cal4metrics(predicts, validlabel)
        print("accuracy: %s\trecall: %s\nprecision: %s\tfscore: %s\n" % (accuracy, recall, precision, fscore))
        print('loss:\t%f' % validloss)
        print(transition_params)
        return validloss

    def giveTestResult(self, testsens, testlabel, sess, ofilepath='./result/bicrfresult.txt'):
        predicts = []
        print('test begin!')
        for ii in range(0, len(testsens), self.batch_size):
            endidx = min(ii+self.batch_size, len(testsens))
            if endidx <= ii:
                break
            xx, ll, mm, seql = prepare_data(testsens[ii:endidx], testlabel[ii:endidx])
            lw = np.asarray([1. for jj in range(endidx-ii)])
            batch_ss = xx.shape[0]
            feed_dict = {self.input_sens: xx, self.mask: mm, self.seqlen: seql, self.labels: ll, self.keep_prob: 1, self.batch_s: batch_ss, self.loss_weight: lw}
            unary_scores, transition_params = sess.run([self.unary_scores, self.transition_params], feed_dict=feed_dict)
            pp = []
            for us_, sl_ in zip(unary_scores, seql):
                us_ = us_[:sl_]
                viterbi_seq, _ = tf.contrib.crf.viterbi_decode(us_, transition_params)
                pp.append(viterbi_seq)
            predicts.extend(pp)
        accuracy, recall, precision, fscore = cal4metrics(predicts, testlabel)
        print("test:\naccuracy: %s\trecall: %s\nprecision: %s\tfscore: %s\n" % (accuracy, recall, precision, fscore))
        with open(ofilepath, 'w') as f:
            for ii in range(len(testsens)):
                f.write('%d' % predicts[ii][0])
                for jj in range(1, len(testsens[ii])):
                    f.write(' %d' % predicts[ii][jj])
                f.write('\n')
        return predicts

    def gene_prob(self, testsens, testlabel, r_dictionary, sess,
                  probresultpath='./result/bilitestprob.pkl',
                  segresultpath='./result/bilitest_seg_temp.txt'):
        probs = []
        predicts = []
        print('generate segment prob')
        for ii in range(0, len(testsens), self.batch_size):
            endidx = min(ii+self.batch_size, len(testsens))
            if endidx <= ii:
                break
            xx, ll, mm, seql = prepare_data(testsens[ii:endidx], testlabel[ii:endidx])
            batch_ss = xx.shape[0]
            feed_dict = {self.input_sens: xx, self.mask: mm, self.seqlen: seql, self.labels: ll, self.keep_prob: 1, self.batch_s: batch_ss}
            unary_scores, transition_params = sess.run([self.unary_scores, self.transition_params], feed_dict=feed_dict)
            pp = []
            for us_, sl_ in zip(unary_scores, seql):
                us_ = us_[:sl_]
                viterbi_seq, _ = tf.contrib.crf.viterbi_decode(us_, transition_params)
                pp.append(viterbi_seq)
                probs.append(us_)
            predicts.extend(pp)

        with open(probresultpath, 'wb') as f:
            pkl.dump(probs, f)

        output_CWS_result(predicts, testsens, r_dictionary, segresultpath)
        return probs, transition_params
