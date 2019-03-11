#coding=utf-8
import tensorflow as tf
import numpy as np
import collections
import pickle as pkl
from utils import *
import os
from data_utils import readfilelc, readfilelcul
from model import CWScnncrf
class Config(object):
    dataset = 'upuc'
    vocab_size = None
    emb_dim = 200
    output_dim = 4


    use_pretrain_emb = True
    #use_pretrain_emb = False
    pretrain_emb = None
    emb_trainable = True
    #emb_trainable = False

    filter_sizes = [3]
    num_filters = 400


    max_epochs = 100
    early_stopping = 3
    keep_prob = 0.5
    lr = 0.001
    batch_size = 64
    maxlen = 70
    dropSet = set(['cnn', 'lstm'])
    nolbili = 0
    maxiternum = 10

def main(saveid=0, dporate=0.5, jcweight=1., candinum=1, candi_k=64):
    print("save id %d" % saveid)
    print("dropout rate: %f" % dporate)
    config = Config()

    filepath = './data/%s_training.utf8' % config.dataset
    orisens, cwslabels, word_sens = gettraindata(filepath=filepath)
    oritrainsens = orisens
    traincwslabels = cwslabels

    filepath = './data/re/ZX/ZXD.labeled'
    orivalidsens, validcwslabels = readfilelc(filepath)
    filepath = './data/re/ZX/ZXE.labeled'
    oritestsens, testcwslabels = readfilelc(filepath)
    filepath = './data/nore/ZXRaw/ZXR16K.Mega.v1.1.labeled.v1'
    orinoltrainsens, noltraincwslabels = readfilelcul(filepath)

    dictionary = getDict(oritrainsens+orinoltrainsens)
    r_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    saveprefix = config.dataset


    trainsens, maxlen = getsensid(oritrainsens, dictionary)
    noltrainsens, nlmaxlen = getsensid(orinoltrainsens, dictionary)
    validsens, validmaxlen = getsensid(orivalidsens, dictionary)
    testsens, testmaxlen = getsensid(oritestsens, dictionary)
    maxlen = max(maxlen, nlmaxlen, validmaxlen, testmaxlen)
    config.maxlen = maxlen
    config.vocab_size = len(dictionary)
    if config.use_pretrain_emb:
        pretrain_file='./data/charVecForCWS.txt'
        config.emb_dim = 200
        pre_emb = get_pretrain_emb(dictionary, pretrain_file=pretrain_file, emb_size=config.emb_dim)
        config.pretrain_emb = pre_emb
        assert pre_emb.shape[0] == len(dictionary)

    print("%d train samples" % len(trainsens))
    print("%d no label train samples" % len(noltrainsens))
    print("%d valid samples" % len(validsens))
    print("%d test samples" % len(testsens))

    word_dict = getworddict(word_dictpath='./data/zx_dict.txt')
    temptrainsens = trainsens
    temptraincwslabels = traincwslabels

    config.keep_prob = dporate
    savesuffix = '%d' % (int(dporate*10))

    resultfolder = 'cwscnncrf_crossdomain'
    nolbili = config.nolbili
    f = open('./%s/%siterlog_%s_nolbili%d_%d.txt' % (resultfolder, saveprefix, savesuffix, int(100*nolbili), saveid), 'w', encoding='utf-8')
    f.write("accuracy:\trecall:\tprecision:\tfscore:\n")

    maxiternum = config.maxiternum + 1
    print("need iternum: %d" % maxiternum)
    for idx in range(maxiternum):
        print("iter: %d" % idx)
        tf.reset_default_graph()
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        model = CWScnncrf(config)
        init = tf.initialize_all_variables()
        with tf.Session(config=tfconfig) as sess:
            sess.run(init)
            saver = tf.train.Saver(max_to_keep=30)
            if idx == 0:
                trainweight = [1. for ii in range(len(trainsens))]
                savepath = './%s/ckpt/%scnncrf_%s_%d_iter%d_%d' % (resultfolder, saveprefix, savesuffix, int(100*nolbili), idx, saveid)
                model.train(trainsens, traincwslabels, trainweight, validsens, validcwslabels, testsens, testcwslabels, sess, saver, savepath=savepath)
                ofilepath = './%s/result/%scnncrf_%s_%d_iter%d_%d.txt' % (resultfolder, saveprefix, savesuffix, int(100*nolbili), idx, saveid)
                predicts = model.giveTestResult(testsens, testcwslabels, sess, ofilepath=ofilepath)
                ofilepath = './%s/result/%sCWSresult_cnncrf_%s_%d_iter%d_%d.txt' % (resultfolder, saveprefix, savesuffix, int(100*nolbili), idx, saveid)
                output_CWS_result(predicts, testsens, r_dictionary, ofilepath=ofilepath)
            else:
                labelfile='./%s/data/%sfakelabel_%s_%d_iter%d_%d.txt' % (resultfolder, saveprefix, savesuffix, int(100*nolbili), idx - 1, saveid)
                fake_labels = getlabelfromfile(labelfile=labelfile)
                assert len(fake_labels) == len(tempnoltrainsens)
                print("add %d fakesens" % (len(fake_labels)))
                trainsens = temptrainsens + tempnoltrainsens
                traincwslabels = temptraincwslabels + fake_labels
                trainweight = [1. for ii in range(len(trainsens))] + [jcweight*fake_weight[ii] for ii in range(len(fake_weight))]
                savepath = './%s/ckpt/%scnncrf_%s_%d_iter%d_%d' % (resultfolder, saveprefix, savesuffix, int(100*nolbili), idx, saveid)
                model.train(trainsens, traincwslabels, trainweight, validsens, validcwslabels, testsens, testcwslabels, sess, saver, savepath=savepath)
                ofilepath = './%s/result/%scnncrf_%s_%d_iter%d_%d.txt' % (resultfolder, saveprefix, savesuffix, int(100*nolbili), idx, saveid)
                predicts = model.giveTestResult(testsens, testcwslabels, sess, ofilepath=ofilepath)
                ofilepath = './%s/result/%sCWSresult_cnncrf_%s_%d_iter%d_%d.txt' % (resultfolder, saveprefix, savesuffix, int(100*nolbili), idx, saveid)
                output_CWS_result(predicts, testsens, r_dictionary, ofilepath=ofilepath)
            accuracy, recall, precision, fscore = cal4metrics(predicts, testcwslabels)
            loginfo = "%s\t%s\t%s\t%s\n" % (accuracy, recall, precision, fscore)
            f.write(loginfo)
            f.flush()
            if idx == maxiternum - 1:
                break
            tempnoltrainsens = noltrainsens[:]
            tempnoltraincwslabels = noltraincwslabels[:]
            probresultpath = './%s/result/%sprobs_%s_%d_iter%d_%d.pkl' % (resultfolder, saveprefix, savesuffix, int(100*nolbili), idx, saveid)
            segresultpath = './%s/result/%s_segtemp_%s_%d_iter%d_%d.txt' % (resultfolder, saveprefix, savesuffix, int(100*nolbili), idx, saveid)
            _, transition_params = model.gene_prob(tempnoltrainsens, tempnoltraincwslabels, r_dictionary, sess, probresultpath=probresultpath, segresultpath=segresultpath)
            probpath = './%s/result/%sprobs_%s_%d_iter%d_%d.pkl' % (resultfolder, saveprefix, savesuffix, int(100*nolbili), idx, saveid)
            olabelpath = './%s/data/%sfakelabel_%s_%d_iter%d_%d.txt' % (resultfolder, saveprefix, savesuffix, int(100*nolbili), idx, saveid)
            osenspath = './%s/data/%sfakeseg_%s_%d_iter%d_%d.txt' % (resultfolder, saveprefix, savesuffix, int(100*nolbili), idx, saveid)
            oscorepath = './%s/data/%sfakescores_%s_%d_iter%d_%d.txt' % (resultfolder, saveprefix, savesuffix, int(100*nolbili), idx, saveid)
            tempnoltrainsens, _, fake_weight = generate_fake_label_prmsnk(tempnoltrainsens, r_dictionary, transition_params, probpath=probpath, olabelpath=olabelpath, osenspath=osenspath, word_dict=word_dict, candi_num=candinum, candi_k=candi_k, oscorepath=oscorepath)

    f.close()



if __name__=='__main__':
    saveids = [ii for ii in range(0, 5)]#average for five experiments
    dporate = [0.7 for ii in range(5)]
    jcweight = [0.5 for ii in range(5)]
    candinum = [1 for ii in range(10)]
    candi_k = [64 for ii in range(10)]

    for ii in range(len(saveids)):
        main(saveids[ii], dporate=dporate[ii], jcweight=jcweight[ii], candinum=candinum[ii], candi_k=candi_k[ii])
