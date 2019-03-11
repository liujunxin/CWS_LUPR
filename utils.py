#coding=utf-8
import tensorflow as tf
import numpy as np
import collections
import pickle as pkl

def cal_metrics(predicts, labels):
    accuracy = 0.
    num = 0
    for ii in range(len(predicts)):
        for jj in range(len(predicts[ii])):
            if predicts[ii][jj] == labels[ii][jj]:
                accuracy += 1
            num += 1
    accuracy = accuracy / num
    return accuracy

def cal4metrics(predicts, labels):
    accuracy = cal_metrics(predicts, labels)
    predicts_bi = []
    labels_bi = []
    for ii in range(len(predicts)):
        pp = []
        ll = []
        for jj in range(len(predicts[ii])):
            if predicts[ii][jj] == 0 or predicts[ii][jj] == 3:
                pp.append(1)
            else:
                pp.append(0)
            if labels[ii][jj] == 0 or labels[ii][jj] == 3:
                ll.append(1)
            else:
                ll.append(0)
        predicts_bi.append(pp)
        labels_bi.append(ll)

    TP = 0.0
    FN = 0.0
    FP = 0.0
    for ii in range(len(labels_bi)):
        jj = 0
        while jj < len(labels_bi[ii]):
            if labels_bi[ii][jj] == 1:
                kk = jj + 1
                while kk < len(labels_bi[ii]) and labels_bi[ii][kk] == 0:
                    kk += 1
                temp1 = labels_bi[ii][jj:kk]
                temp2 = predicts_bi[ii][jj:kk]
                if all([temp1[zz]==temp2[zz] for zz in range(len(temp1))]) and (kk==len(labels_bi[ii]) or predicts_bi[ii][kk] != 0):
                    TP += 1
                else:
                    FN += 1
                jj = kk
            else:
                jj += 1
    for ii in range(len(predicts_bi)):
        jj = 0
        while jj < len(predicts_bi[ii]):
            if predicts_bi[ii][jj] == 1:
                kk = jj + 1
                while kk < len(predicts_bi[ii]) and predicts_bi[ii][kk] == 0:
                    kk += 1
                temp1 = labels_bi[ii][jj:kk]
                temp2 = predicts_bi[ii][jj:kk]
                if any([temp1[zz]!=temp2[zz] for zz in range(len(temp2))]) or (kk < len(predicts_bi[ii]) and labels_bi[ii][kk] == 0):
                    FP += 1
                jj = kk
            else:
                jj += 1
    if TP == 0:
        recall = 0
        precise = 0
    else:
        recall = TP/(TP+FN)
        precise = TP/(TP+FP)
    if recall == 0 and precise == 0:
        fscore = 0
    else:
        fscore = 2*precise*recall/(precise+recall)

    return accuracy, recall, precise, fscore

def prepare_data(seqs_x, seqs_l, classesnum=4, maxlen=None):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_l = [len(s) for s in seqs_l]

    if maxlen is not None:
        seqs_x = [s if len(s) <= maxlen else s[:maxlen] for s in seqs_x]
        seqs_l = [s if len(s) <= maxlen else s[:maxlen] for s in seqs_l]

        lengths_x = [len(s) for s in seqs_x]
        lengths_l = [len(s) for s in seqs_l]

    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x)
    maxlen_l = np.max(lengths_l)
    assert maxlen_x == maxlen_l

    x = np.zeros((n_samples, maxlen_x)).astype('int32')
    y = np.zeros((n_samples, maxlen_l)).astype('int32')
    mask = np.zeros((n_samples, maxlen_x)).astype('float32')
    for idx, [s_x, s_l] in enumerate(zip(seqs_x, seqs_l)):
        x[idx, :lengths_x[idx]] = s_x
        mask[idx, :lengths_x[idx]] = 1.
        y[idx, :lengths_x[idx]] = s_l

    lengths_x = np.asarray(lengths_x).astype('int32')
    return x, y, mask, lengths_x


def gettraindata(filepath='./data/msr_training.utf8'):
    orisens = []
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            orisens.append(line)
    print("ori sample num: %d" % len(orisens))
    labels = []
    char_sens = []
    word_sens = []
    for sen in orisens:
        words = sen.strip().split()
        if len(words) == 0:
            continue
        chars = []
        label = []
        for word in words:
            chars.extend([char for char in word])
            if len(word) == 1:
                label.append(3)
            else:
                label.append(0)
                for ii in range(len(word)-2):
                    label.append(1)
                label.append(2)
        assert len(label) == len(chars), sen
        labels.append(label)
        char_sens.append(chars)
        word_sens.append(words)
    print("%d samples after preprocess" % len(char_sens))

    return char_sens, labels, word_sens

def gettestdata(filepath='./data/msr_test_gold.utf8'):
    orisens = []
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            orisens.append(line)
    print("test ori sample num: %d" % len(orisens))
    labels = []
    char_sens = []
    for sen in orisens:
        words = sen.strip().split()
        if len(words) == 0:
            continue
        chars = []
        label = []
        for word in words:
            chars.extend([char for char in word])
            if len(word) == 1:
                label.append(3)
            else:
                label.append(0)
                for ii in range(len(word)-2):
                    label.append(1)
                label.append(2)
        assert len(label) == len(chars), sen
        labels.append(label)
        char_sens.append(chars)
    print("%d test samples after preprocess" % len(char_sens))
    return char_sens, labels

def readfilelc(filepath='./data/re/ZX/ZXD.labeled'):
    sents = []
    labels = []
    ldict = {'B':0, 'M':1, 'E':2, 'S':3}
    idx = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        sent = []
        label = []
        while True:
            buff = f.readline()
            idx += 1
            if buff.strip() == '' and len(sent) == 0:
                break
            if buff.strip() == '':
                sents.append(sent)
                labels.append(label)
                sent = []
                label = []
            else:
                temp = buff.replace('\n', '').split('\t')
                label.append(ldict[temp[0]])
                sent.append(temp[1])
    print(len(sents))
    return sents, labels

def readfilelcul(filepath='./data/nore/ZXRaw/ZXR16K.Mega.v1.1.labeled.v1'):
    sents = []
    labels = []
    ldict = {'B':0, 'M':1, 'E':2, 'S':3}
    idx = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        sent = []
        label = []
        while True:
            buff = f.readline()
            idx += 1
            if buff.strip() == '' and len(sent) == 0:
                break
            if buff.strip() == '':
                sents.append(sent)
                labels.append(label)
                sent = []
                label = []
            else:
                temp = buff.replace('\n', '').split()
                label.append(ldict[temp[0][0]])
                sent.append(temp[1])
    print(len(sents))
    return sents, labels

def getDict(orisens, min_count=0):
    chars = []
    for sen in orisens:
        chars.extend(sen)
    count = []
    count.extend(collections.Counter(chars).most_common())
    dictionary = dict()
    dictionary['<PAD>'] = 0#PAD
    dictionary['<UNK>'] = 1#UNK
    for word, c in count:
        if c < min_count:
            break
        dictionary[word] = len(dictionary)
    r_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    print('vocab size: %d' % len(dictionary))
    return dictionary

def getsensid(orisens, dictionary):
    senslen = [len(s) for s in orisens]
    maxlen = max(senslen)
    sensid = []
    for sen in orisens:
        temp = [dictionary[char] if char in dictionary else 1 for char in sen]
        sensid.append(temp)
    return sensid, maxlen

def data_split(orisens, oricwslabels, valid_bili=0.1):
    split_at = int((1-valid_bili) * len(orisens))
    validsens = orisens[split_at:]
    validcwslabels = oricwslabels[split_at:]

    trainsens = orisens[:split_at]
    traincwslabels = oricwslabels[:split_at]

    return trainsens, traincwslabels, validsens, validcwslabels

def data_random_select(orisens, oricwslabels, train_bili=0.25, rpath='./rpath/rtrain0.npy'):
    if os.path.exists(rpath):
        r = np.load(rpath)
    else:
        r = np.random.permutation(len(orisens))
        np.save(rpath, r)
    orisens = [orisens[ii] for ii in r]
    oricwslabels = [oricwslabels[ii] for ii in r]

    split_at = int(train_bili * len(orisens))

    validsens = orisens[split_at:]
    validcwslabels = oricwslabels[split_at:]

    trainsens = orisens[:split_at]
    traincwslabels = oricwslabels[:split_at]

    return trainsens, traincwslabels, validsens, validcwslabels

def get_pretrain_emb(dictionary, pretrain_file='./data/charVecForCWS.txt', emb_size=200):
    vocab_size = len(dictionary)
    pre_emb = np.random.random([vocab_size, emb_size]) * 2 - 1
    with open(pretrain_file, 'r', encoding='utf-8') as f:
        temp = f.readline().strip().split()
        wordnum = int(temp[0])
        veclen = int(temp[1])
        assert emb_size == veclen
        pre_num = 0
        for line in f:
            temp = line.strip().split()
            if temp[0] in dictionary:
                index = dictionary[temp[0]]
                try:
                    prevecs = np.asarray([float(num) for num in temp[1:]])
                    pre_emb[index, :] = prevecs
                except:
                    print(temp[0])
                pre_num += 1
        print('pretrain word num: %d' % pre_num)
    return pre_emb

def getworddict(word_dictpath='./data/cidian.txt', dsize=1):
    word_dict = set()
    with open(word_dictpath, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip()) > 1:
                word_dict.add(line.strip())
    if dsize < 1:
        word_dict = list(word_dict)
        word_dict = word_dict[:int(dsize*len(word_dict))]
        word_dict = set(word_dict)
    return word_dict

def getlabelfromfile(labelfile='./data/bilifakelabel.txt'):
    labels = []
    with open(labelfile, 'r', encoding='utf-8') as f:
        for line in f:
            label = [int(ll) for ll in line.strip().split()]
            labels.append(label)
    return labels

def output_CWS_result(predicts, testsens, r_dictionary, ofilepath='./result/segresult_cnncrf.txt'):
    seg_sens = []
    for ii in range(len(testsens)):
        sen = []
        begin = 0
        end = 1
        while end < len(testsens[ii]):
            if predicts[ii][end] == 0 or predicts[ii][end] == 3:
                sen.append(''.join([r_dictionary[c] for c in testsens[ii][begin:end]]))
                begin = end
            end += 1
        sen.append(''.join([r_dictionary[c] for c in testsens[ii][begin:end]]))
        seg_sens.append(' '.join(sen))

    with open(ofilepath, 'w', encoding='utf-8') as f:
        for sen in seg_sens:
            f.write(sen)
            f.write('\n')

#prob means unary score
def generate_kmax_path_crf(probs_allsens, transition, candi_num=64):
    print("generate %d max path" % candi_num)
    kmaxpaths_all = []
    kmaxscores_all = []
    for idx in range(len(probs_allsens)):
        if idx % 1000 == 0:
            print('complete: %d sens' % idx)
        probs = probs_allsens[idx]
        if probs.shape[1] ** probs.shape[0] <= candi_num:
            kmaxpaths = [[ii] for ii in range(probs.shape[1])]
            kmaxscores = [probs[0][ii] for ii in range(probs.shape[1])]
            for ii in range(1, probs.shape[0]):
                temppaths = []
                tempscores = []
                for jj in range(len(kmaxpaths)):
                    for kk in range(probs.shape[1]):
                        temppaths.append(kmaxpaths[jj] + [kk])
                        tempscores.append(kmaxscores[jj] + probs[ii][kk] + transition[kmaxpaths[jj][-1], kk])
                kmaxpaths = temppaths
                kmaxscores = tempscores
            sort_idx = np.argsort(kmaxscores)[::-1]
            kmaxpaths = [kmaxpaths[sort_idx[ii]] for ii in range(len(kmaxpaths))]
            kmaxscores = [kmaxscores[sort_idx[ii]] for ii in range(len(kmaxscores))]
            kmaxpaths_all.append(kmaxpaths)
            kmaxscores_all.append(kmaxscores)
        else:
            kmaxpaths = [[[ii]] for ii in range(probs.shape[1])]
            kmaxscores = [[probs[0][ii]] for ii in range(probs.shape[1])]
            for ii in range(1, probs.shape[0]):
                tempkmaxpaths = []
                tempkmaxscores = []
                for jj in range(probs.shape[1]):
                    temppaths = []
                    tempscores = []
                    for kk in range(len(kmaxpaths)):
                        for mm in range(len(kmaxpaths[kk])):
                            temppaths.append(kmaxpaths[kk][mm]+[jj])
                            tempscores.append(kmaxscores[kk][mm] + probs[ii][jj] + transition[kmaxpaths[kk][mm][-1], jj])
                    sort_idx = np.argsort(tempscores)[::-1]
                    xxxkmaxpaths = []
                    xxxkmaxscores = []
                    for mm in range(min(candi_num, len(temppaths))):
                        xxxkmaxpaths.append(temppaths[sort_idx[mm]])
                        xxxkmaxscores.append(tempscores[sort_idx[mm]])
                    tempkmaxpaths.append(xxxkmaxpaths)
                    tempkmaxscores.append(xxxkmaxscores)
                kmaxpaths = tempkmaxpaths
                kmaxscores = tempkmaxscores
            tempkmaxpaths = []
            tempkmaxscores = []
            for kk in range(len(kmaxpaths)):
                for mm in range(len(kmaxpaths[kk])):
                    tempkmaxpaths.append(kmaxpaths[kk][mm])
                    tempkmaxscores.append(kmaxscores[kk][mm])
            sort_idx = np.argsort(tempkmaxscores)[::-1]
            kmaxpaths = []
            kmaxscores = []
            for mm in range(min(candi_num, len(tempkmaxpaths))):
                kmaxpaths.append(tempkmaxpaths[sort_idx[mm]])
                kmaxscores.append(tempkmaxscores[sort_idx[mm]])
            kmaxpaths_all.append(kmaxpaths)
            kmaxscores_all.append(kmaxscores)
    return kmaxpaths_all, kmaxscores_all


def prepare_for_norm(probs_sens, testsens, output_dim=4):
    senslen = [len(sen) for sen in testsens]
    maxlen = max(senslen)
    n_samples = len(testsens)
    probs_input = np.zeros((n_samples, maxlen, output_dim)).astype('float32')
    for idx in range(n_samples):
        probs_input[idx, :senslen[idx], :] = probs_sens[idx]
    senslen = np.asarray(senslen)
    return probs_input, senslen
def cal_crf_norm(probs_allsens, transition, testsens):
    output_dim = transition.shape[0]

    inputs_norm = tf.placeholder(tf.float32, [None, None, output_dim], name='inputs_norm')
    seqlen_norm = tf.placeholder(tf.int32, [None], name='seqlen_norm')
    transition_norm = tf.placeholder(tf.float32, [output_dim, output_dim], name='transition_norm')
    log_norm = tf.contrib.crf.crf_log_norm(inputs_norm, seqlen_norm, transition_norm)
    norm = log_norm
    batch_size = 64
    norm_list = []
    show_thr = 0
    with tf.Session() as sess:
        for ii in range(0, len(probs_allsens), batch_size):
            endidx = min(ii+batch_size, len(probs_allsens))
            if endidx <= ii:
                break
            probs, seql = prepare_for_norm(probs_allsens[ii:endidx], testsens[ii:endidx], output_dim)
            feed_dict = {inputs_norm:probs, seqlen_norm:seql, transition_norm:transition}
            nn = sess.run(norm, feed_dict=feed_dict)
            for jj in range(len(nn)):
                norm_list.append(nn[jj])
            if len(norm_list) > show_thr:
                print("norm for %d sens" % len(norm_list))
                show_thr += 1000
    assert len(norm_list) == len(testsens)
    return norm_list

def generate_fake_label_prmsnk(testsens, r_dictionary, transition,
                        probpath='./result/bilitestprob.pkl',
                        olabelpath='./data/bilitestfakelabel.txt',
                        osenspath='./data/bilitestfakeseg.txt',
                        word_dict=None,
                        candi_num=1,
                        candi_k=64,
                        oscorepath='./data/bilitestfakescore.txt'):
    with open(probpath, 'rb') as f:
        probs_allsens = pkl.load(f)
    kmaxpaths_all, kmaxscores_all = generate_kmax_path_crf(probs_allsens, transition, candi_num=candi_num*candi_k)
    print(len(kmaxpaths_all[0]))

    fakelabels = []
    fakesens = []
    final_scores = []
    final_scores_m = []
    final_scores_d = []
    final_fake_sens = []
    norm_list = cal_crf_norm(probs_allsens, transition, testsens)
    for idx in range(len(testsens)):
        kmaxpaths = kmaxpaths_all[idx]
        kmaxscores = kmaxscores_all[idx]
        probs = probs_allsens[idx]
        scores_with_dict = []
        score_only_dict = []
        score_only_model = []
        for ii in range(len(kmaxpaths)):
            sen = []
            begin = 0
            end = 1
            assert len(kmaxpaths[ii]) == len(testsens[idx])
            pathnow = kmaxpaths[ii]
            score_model = kmaxscores[ii]
            score_model = np.exp(score_model - norm_list[idx])
            if np.isnan(score_model):
                score_model = 0
            while end < len(testsens[idx]):
                if pathnow[end] == 0 or pathnow[end] == 3:
                    sen.append(''.join([r_dictionary[c] for c in testsens[idx][begin:end]]))
                    begin = end
                end += 1
            sen.append(''.join([r_dictionary[c] for c in testsens[idx][begin:end]]))
            score_dict = 0
            for word in sen:
                if len(word) > 1:
                    if word in word_dict:
                        score_dict = score_dict + 1
            if len(sen) > 0:
                score_dict = score_dict / len(sen)

            score = score_model + score_dict
            score = np.exp(score)
            scores_with_dict.append(score)
            score_only_dict.append(score_dict)
            score_only_model.append(score_model)
        guiyi = np.sum(np.sort(scores_with_dict)[::-1][:candi_num])
        sort_idx = np.argsort(scores_with_dict)[::-1]
        for ii in range(min(candi_num, len(kmaxpaths))):
            fakelabels.append(kmaxpaths[sort_idx[ii]])
            final_scores.append(scores_with_dict[sort_idx[ii]]/guiyi)
            final_fake_sens.append(testsens[idx])
            final_scores_m.append(score_only_model[sort_idx[ii]])
            final_scores_d.append(score_only_dict[sort_idx[ii]])

        if idx % 1000 == 0:
            print(idx)
    with open(olabelpath, 'w', encoding='utf-8') as f:
        for label in fakelabels:
            f.write(' '.join([str(ll) for ll in label]))
            f.write('\n')
    with open(oscorepath, 'w', encoding='utf-8') as f:
        for ii in range(len(final_scores)):
            f.write("%.4f\t%.4f\t%.4f\n" % (final_scores[ii], final_scores_m[ii], final_scores_d[ii]))

    output_CWS_result(fakelabels, final_fake_sens, r_dictionary, ofilepath=osenspath)

    return final_fake_sens, fakelabels, final_scores
