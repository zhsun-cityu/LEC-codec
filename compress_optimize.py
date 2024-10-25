# this file is an optimized version of compress.py
# Refer to
# https://www.nayuki.io/page/reference-arithmetic-coding
# https://github.com/nayuki/Reference-arithmetic-coding
# https://github.com/mohit1997/DeepZip
#


import tensorflow as tf
import numpy as np
from collections import deque
import argparse

from tensorflow.python.ops.gen_array_ops import one_hot
from codec_args import get_argument
from utils.arithmetic_coding import arithmeticcoding_fast
import pickle
import json
import tqdm
import tempfile
import shutil
import os
import model_defination
from utils.dataset_utils import prepare_dataset_for_compression, get_one_hot, int2char_dict, degenerate_base_dict
from utils.utils import var_int_encode
import hashlib
import time
import math
import multiprocessing

def lcm(a, b):
    return abs(a*b) // math.gcd(a, b)

def get_label_index(chars, digit_arr):
    idx = 0
    power = len(digit_arr) - 1
    for digit in digit_arr:
        idx += math.pow(len(chars), power) * digit
        power -= 1
    return int(idx)

def infer_compression(digit_x, digit_y, onehot_x, compress_model, arguments, final_step=False, markov_model=False, lut=None, stride=1, sequence_y=True):
    assert len(digit_x)==len(digit_y)==len(onehot_x)
    # convert digit X into one-hot X
    chars = arguments.chars
    int2char = int2char_dict(chars)

    context_length = arguments.context_length
    batch_size = arguments.batch_size ##
    infer_batch_size = arguments.infer_batch_size

    alphabet_size = len(chars)

    # infer_l = int(math.floor((len(digit_x)/infer_batch_size)//stride)*stride*infer_batch_size)
    infer_l = arguments.infer_l
    l = infer_l

    num_iters = int(math.floor(l/batch_size-context_length)) ## no need divide stride, cause the result is stored in array
    # num_iters = arguments.num_iters
    # infer_num_iters = int(math.floor((l/infer_batch_size)/stride))
    infer_num_iters = arguments.infer_num_iters

    ind = np.array(range(batch_size)) * int(l/batch_size) ## ind is for encoding phrase
    infer_ind = np.array(range(infer_batch_size)) * int(infer_l/infer_batch_size)

    ### do NN inference first
    prob_vector_array = np.ones((infer_l, alphabet_size))
    if sequence_y:
        # cumul = np.zeros((batch_size, alphabet_size+1), dtype = np.uint64)
        for j in tqdm.tqdm(range(infer_num_iters), desc='Neural Network Computation'):
            prob_vector = compress_model.predict(onehot_x[infer_ind,:], batch_size=infer_batch_size)

            prob_vector = np.transpose(prob_vector, (1,0,2)) # transpose to (predict_num, bs, base_num)

            assert prob_vector.shape[0]==stride, print("zero axis of ", prob_vector.shape, "not equals to ", stride) ##

            # this part can be optimized
            for s in range(stride):
                prob_vector_array[infer_ind, ...] = prob_vector[s]
                infer_ind += 1

        # ## process residule part (e.g. 6*50 < 328, process the residule 28 bases)
        if (infer_num_iters * stride) < int(infer_l/infer_batch_size):

            # infer_ind = np.array(range(1, infer_batch_size+1)) * int(l/infer_batch_size) - stride
            resid_num = int(l/infer_batch_size) - (infer_num_iters*stride)

            prob_vector = compress_model.predict(onehot_x[infer_ind,:], batch_size=infer_batch_size)

            prob_vector = np.transpose(prob_vector, (1,0,2)) # transpose to (predict_num, bs, base_num)
            assert prob_vector.shape[0]==stride ##

            prob_vector = prob_vector[:resid_num]

            # infer_ind = len(infer_ind) - resid_num
            # infer_ind += (stride-resid_num)
            for s in range(resid_num):
                # prob_vector_array[-resid_num:, ...] = prob_vector[s]
                prob_vector_array[infer_ind, ...] = prob_vector[s]
                infer_ind += 1

        else:
            print("No residue part")
            pass

    else:
        raise NotImplementedError
    
    # print("Encoding......")
    # open compressed files and compress first few characters using uniform distribution
    f = [open(arguments.temp_file_prefix+'.' + str(i),'wb') for i in range(batch_size)]
    bitout = [arithmeticcoding_fast.BitOutputStream(f[i]) for i in range(batch_size)]
    enc = [arithmeticcoding_fast.ArithmeticEncoder(32, bitout[i]) for i in range(batch_size)]

    ## Encode context part
    if arguments.markov_model: # Use markov model to get the transition probability
        rf = open(arguments.prob_dict_file, 'rb')
        # base_prob = np.array([0.3086, 0.3127, 0.1315, 0.2471])
        k = pickle.load(rf) - 1
        base_prob = pickle.load(rf)
        prob_dict = pickle.load(rf)
        cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(base_prob*int(1e7) + 1)
        # context_array = np.zeros((batch_size, n))
        context_list = []
        for i in range(batch_size):
            context_list.append(deque())
            for j in range(k):
                enc[i].write(cumul, digit_x[ind[i],j])
                context_list[i].append(int2char[digit_x[ind[i],j]])

        for i in range(batch_size):
            for j in range(k, min(arguments.context_length, num_iters)):
                context = ''.join(context_list[i])
                prob = np.array(prob_dict[context])
                cumul[1:] = np.cumsum(prob*int(1e7) + 1)
                #enc[i].write(cumul, X[ind[i],j])
                enc[i].write(cumul, digit_x[ind[i],j])
                # if len(context_list[i]) > n:
                context_list[i].append(int2char[digit_x[ind[i],j]])
                context_list[i].popleft()
    else: # uniform initialize probability
        prob = np.ones(alphabet_size)/alphabet_size ## uniform probability initialization
        cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(prob*int(1e7) + 1)
        for i in range(batch_size):
            for j in range(min(arguments.context_length, num_iters)):
                enc[i].write(cumul, digit_x[ind[i],j])

    ## Encode NN inference part
    ## TODO: accelerate this part
    cumul = np.zeros((batch_size, alphabet_size+1), dtype = np.uint64)
    for j in tqdm.tqdm(range(num_iters), desc='Entropy Coding'): ## encode the prob_dist one-by-one
        prob_vector = prob_vector_array[ind, ...] # (batch_size, alphabet_size)

        cumul[:, 1:] = np.cumsum(prob_vector*int(1e7)+1, axis=1)
        for i in range(batch_size):
            enc[i].write(cumul[i, :], digit_y[ind[i]])
        ind += 1

    ## close files
    for i in range(batch_size):
        enc[i].finish()
        bitout[i].close()
        f[i].close()

    ### Encode last part
    f = open(arguments.temp_file_prefix+'.last','wb')
    bitout = arithmeticcoding_fast.BitOutputStream(f)
    enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)

    cumul = np.zeros(alphabet_size+1, dtype = np.uint64)

    ## encode the context part
    for i in tqdm.tqdm(range(l-context_length, len(onehot_x), stride),desc="Encode last part"):
        prob_vector = compress_model.predict(onehot_x[i][np.newaxis, ...], batch_size=1)

        prob_vector = np.transpose(prob_vector, (1,0,2)) # transpose to (predict_num, bs, base_num)
        assert prob_vector.shape[0]==stride ##

        for k in range(stride):
            if i+k ==len(onehot_x):
                break
            prob_dist = prob_vector[k]
            cumul[1:] = np.cumsum(prob_dist*int(1e7) + 1)
            enc.write(cumul, digit_y[i+k])

    enc.finish()
    bitout.close()
    f.close()
    return lut

def bi_infer_compression(digit_x, digit_y, onehot_x, r_digit_x, r_digit_y, r_onehot_x, compress_model_tuple, arguments, final_step=False, markov_model=False, lut=None, stride=1, sequence_y=True):
    assert len(digit_x)==len(r_digit_x) and len(digit_y)==len(r_digit_y) and len(onehot_x)==len(r_onehot_x)

    # convert digit X into one-hot X
    chars = arguments.chars
    int2char = int2char_dict(chars)

    context_length = arguments.context_length
    batch_size = arguments.batch_size // 2 ## halve the gob cause context is shared
    infer_batch_size = arguments.infer_batch_size
    alphabet_size = len(chars)

    fwd_model, bkwd_model = compress_model_tuple

    infer_l = arguments.infer_l
    l = infer_l

    num_iters = int(math.floor(l/(batch_size*2))-context_length)
    infer_num_iters = int(math.floor((l/(infer_batch_size))/stride))
    ind = np.array(range((batch_size * 2))) * int(l/(batch_size * 2)) ## ind is for encoding phrase
    infer_ind = np.array(range((infer_batch_size))) * int(infer_l/(infer_batch_size))
    r_onehot_x = r_onehot_x[-l:]

    if not sequence_y:
        raise NotImplementedError
    fwd_prob_vector_array = np.ones((infer_l, alphabet_size))
    bkwd_prob_vector_array = np.ones((infer_l, alphabet_size))
    
    ## predict forward part
    s_t = time.time()
    for j in tqdm.tqdm(range(infer_num_iters), desc='Forward predicting'):
        prob_vector = fwd_model.predict(onehot_x[infer_ind,:], batch_size=infer_batch_size)
        # if len(prob_vector)>2: ## if model predicted more than one base
        prob_vector = np.transpose(prob_vector, (1,0,2)) # transpose to (predict_num, bs, base_num)
        assert prob_vector.shape[0]==stride, print("zero axis of ", prob_vector.shape, "not equals to ", stride) ##

        # this part can be optimized
        for s in range(stride):
            fwd_prob_vector_array[infer_ind, ...] = prob_vector[s]
            infer_ind += 1
    e_t = time.time()
    duration = e_t - s_t

    print("INFO:Forward prediction cost:%.2f sec" % duration)
    s_t = time.time()
    infer_ind = np.array(range((infer_batch_size))) * int(infer_l/(infer_batch_size)) ## reset infer_ind
    ## predict backward part
    for j in tqdm.tqdm(range(infer_num_iters), desc='Backward predicting'):
        prob_vector = bkwd_model.predict(r_onehot_x[infer_ind,:], batch_size=infer_batch_size)
        # if len(prob_vector)>2: ## if model predicted more than one base
        prob_vector = np.transpose(prob_vector, (1,0,2)) # transpose to (predict_num, bs, base_num)
        assert prob_vector.shape[0]==stride, print("zero axis of ", prob_vector.shape, "not equals to ", stride) ##

        # this part can be optimized
        for s in range(stride):
            bkwd_prob_vector_array[infer_ind, ...] = prob_vector[s]
            infer_ind += 1
    e_t = time.time()
    duration = e_t - s_t
    print("INFO:Backward prediction cost:%.2f sec" % duration)


    f = [open(arguments.temp_file_prefix+'.' + str(i),'wb') for i in range(batch_size)]
    bitout = [arithmeticcoding_fast.BitOutputStream(f[i]) for i in range(batch_size)]
    enc = [arithmeticcoding_fast.ArithmeticEncoder(32, bitout[i]) for i in range(batch_size)]

    ### encode context part  (can be optimized)
    s_t = time.time()
    fwd_ind = ind[1::2].copy()
    bkwd_ind = ind[1::2].copy()
    if arguments.markov_model: # uniform initialize the probability
        rf = open(arguments.prob_dict_file, 'rb')
        k = pickle.load(rf) - 1
        base_prob = pickle.load(rf)
        prob_dict = pickle.load(rf)
        cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(base_prob*int(1e7) + 1)
        context_list = []
        for i in range(batch_size):
            context_list.append(deque())
            for j in range(k):
                enc[i].write(cumul, digit_x[fwd_ind[i],j])
                context_list[i].append(int2char[digit_x[fwd_ind[i],j]])

        for i in range(batch_size):
            for j in range(k, context_length):
                context = ''.join(context_list[i])
                prob = np.array(prob_dict[context])
                cumul[1:] = np.cumsum(prob*int(1e7) + 1)
                #enc[i].write(cumul, X[fwd_ind[i],j])
                enc[i].write(cumul, digit_x[fwd_ind[i],j])
                context_list[i].append(int2char[digit_x[fwd_ind[i],j]])
                context_list[i].popleft()

    else: # uniform initialize probability
        prob = np.ones(alphabet_size)/alphabet_size ## uniform probability initialization
        cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(prob*int(1e7) + 1)
        for i in range(batch_size):
            for j in range(context_length):
                enc[i].write(cumul, digit_x[fwd_ind[i],j])
    e_t = time.time()
    duration = e_t - s_t
    print("INFO:Context coding cost:%.2f sec" % duration)

    ## encode forward part
    s_t = time.time()
    cumul = np.zeros((batch_size, alphabet_size+1), dtype = np.uint64)
    for j in tqdm.tqdm(range(num_iters), desc='Entropy coding for forward part.'): ## encode the prob_dist one-by-one
        prob_vector = fwd_prob_vector_array[fwd_ind, ...] # (batch_size, alphabet_size)
        cumul[:, 1:] = np.cumsum(prob_vector*int(1e7)+1, axis=1)
        for i in range(batch_size):
            enc[i].write(cumul[i, :], digit_y[fwd_ind[i]])
        fwd_ind += 1
    e_t = time.time()
    duration = e_t - s_t
    print("INFO:Forward entropy coding cost:%.2f sec" % duration)

    ## encode backward part
    s_t = time.time()
    bkwd_ind -= 1 ## encode previous one first
    bkwd_prob_vector_array = bkwd_prob_vector_array[::-1] ## reverse for convenient index
    # bkwd_prob_vector_array = bkwd_prob_vector_array.round(4)
    r_digit_y = r_digit_y[::-1] ## reverse the r_digit_y for coding
    # bkwd_ind += context_length ## shift context length
    cumul = np.zeros((batch_size, alphabet_size+1), dtype = np.uint64)
    for j in tqdm.tqdm(range(num_iters+context_length), desc='Entropy coding for backward part.'): ## encode the prob_dist one-by-one
        prob_vector = bkwd_prob_vector_array[bkwd_ind, ...] # (batch_size, alphabet_size)
        cumul[:, 1:] = np.cumsum(prob_vector*int(1e7)+1, axis=1)
        for i in range(batch_size):
            enc[i].write(cumul[i, :], r_digit_y[bkwd_ind[i]])
        bkwd_ind -= 1
    e_t = time.time()
    duration = e_t - s_t
    print("INFO:Backward entropy coding cost:%.2f sec" % duration)

    ## close files
    for i in range(batch_size):
        enc[i].finish()
        bitout[i].close()
        f[i].close()

    ## encode the last part with forward prediction
    s_t = time.time()
    f = open(arguments.temp_file_prefix+'.last','wb')
    bitout = arithmeticcoding_fast.BitOutputStream(f)
    enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
    prob = np.ones(int(alphabet_size))/alphabet_size
    cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
    cumul[1:] = np.cumsum(prob*int(1e7) + 1)

    ## encode the context part with uniform distribution
    for j in range(arguments.context_length):
        enc.write(cumul, digit_x[0,j])

    for i in tqdm.tqdm(range(infer_l, len(onehot_x), stride), desc="Encode last part"):
        prob_vector = fwd_model.predict(onehot_x[i][np.newaxis, ...], batch_size=1)

        prob_vector = np.transpose(prob_vector, (1,0,2)) # transpose to (predict_num, bs, base_num)
        assert prob_vector.shape[0]==stride ##

        for k in range(stride):
            if i+k ==len(onehot_x):
                break
            prob_dist = prob_vector[k]
            cumul[1:] = np.cumsum(prob_dist*int(1e7) + 1)
            enc.write(cumul, digit_y[i+k])
    e_t = time.time()
    duration = e_t - s_t
    print("INFO:Last part cost:%.2f sec" % duration)

    enc.finish()
    bitout.close()
    f.close()
    return lut

def compress(arguments, compress_model, backward_model=None, bi_direction=False, lut=None, stride=1, sequence_y=True):
    # text = open(arguments.base_file, 'r').readline().rstrip() ## suppose the base file is only one line
    if bi_direction:
        if backward_model is None:
            print('Please indicate backward model. Abort.')
            exit()
        compress_model_tuple = (compress_model, backward_model)

    text = ''
    lines = open(arguments.base_file, 'r').readlines()
    for line in lines:
        text += line.rstrip()
    # replace_degenerate_bases
    for key in degenerate_base_dict:
        text = text.replace(key, degenerate_base_dict[key][0]) # use the first choice to replace

    text_len = len(text)
    context_length = arguments.context_length
    alphabet_size = len(arguments.chars)
    batch_size = arguments.batch_size
    if batch_size % 2 != 0 and batch_size != 1:
        batch_size = batch_size+1 ## make it to even number
    infer_batch_size = arguments.infer_batch_size

    if arguments.params_file != None:
        with open(arguments.params_file, 'r') as f:
            params = json.load(f)
    else:
        params = {}

    output_base_dir = os.path.dirname(arguments.output_file)
    if not os.path.isdir(output_base_dir):
        os.makedirs(output_base_dir)

    digit_text, digit_x, digit_y, onehot_x, _ = prepare_dataset_for_compression(arguments.base_file, fasta_flag=False, context_length=context_length, stride=stride, sequence_y=sequence_y)
    if bi_direction:
        r_digit_text, r_digit_x, r_digit_y, r_onehot_x, _ = prepare_dataset_for_compression(arguments.base_file, fasta_flag=False, context_length=context_length, stride=stride, sequence_y=sequence_y, reverse=True)

    #NOTE: if the following variables changed, the two inside the infer_compression should change correspondingly
    infer_l = int(math.floor( (len(digit_x)/infer_batch_size)//stride) * stride * infer_batch_size) ## this length is the nn model inference length
    infer_num_iters = int(math.floor((infer_l/infer_batch_size)/stride))

    arguments.infer_l = infer_l
    arguments.infer_num_iters = infer_num_iters

    start = time.time()

    if not bi_direction:
        lut = infer_compression(digit_x, digit_y, onehot_x, compress_model, arguments, lut=lut, stride=stride)

    else: ## use bi-directional prediction
        ## TODO: check if bi_infer_compression check length of digit_x
        # lut = bi_infer_compression(digit_x[:infer_l, :], digit_y[:infer_l], onehot_x[:infer_l], compress_model_tuple, arguments, lut=lut, stride=stride)
        lut = bi_infer_compression(digit_x, digit_y, onehot_x, r_digit_x, r_digit_y, r_onehot_x, compress_model_tuple, arguments, lut=lut, stride=stride)
        batch_size=batch_size//2 ## bit-stream files is halved

    ## write all bitstream file
    output_path = arguments.output_file+'.lec'
    f = open(output_path,'wb')
    for i in tqdm.tqdm(range(batch_size), desc='Export to file'): ####
        f_in = open(arguments.temp_file_prefix+'.'+str(i),'rb')
        byte_str = f_in.read()
        byte_str_len = len(byte_str)
        var_int_encode(byte_str_len, f)
        f.write(byte_str)
        f_in.close()

    f_in = open(arguments.temp_file_prefix+'.last','rb')
    byte_str = f_in.read()
    byte_str_len = len(byte_str)
    var_int_encode(byte_str_len, f)
    f.write(byte_str)
    f_in.close()
    f.close()
    print("Done.")
    duration = time.time() - start
    ratio = os.path.getsize(arguments.base_file)/os.path.getsize(output_path)
    print("Cost %f seconds. Compression ratio %f" %(duration, ratio))
    print("Write to %s" %output_path)

    ## write parameters
    ## model parameter
    params['model_name'] = arguments.model_name
    params['pretrained_model'] = arguments.pretrained_model
    params['bi_direction'] = False
    if bi_direction:
        params['bi_direction'] = True
        params['backward_model'] = arguments.backward_weights_file
    params['prob_dict_file'] = arguments.prob_dict_file
    ## compressing parameter
    params['sequence_length'] = text_len
    params['infer_l'] = infer_l
    # params['num_iters'] = num_iters
    params['infer_num_iters'] = infer_num_iters
    params['context_length'] = arguments.context_length
    params['predict_base_num'] = arguments.predict_base_num ## predict stride
    params['chars'] = arguments.chars
    params['batch_size'] = arguments.batch_size ## store the original batch_size
    params['infer_batch_size'] = arguments.infer_batch_size
    params['combined_num'] = arguments.combined_num
    params['markov_model'] = arguments.markov_model
    params['sequence_y'] = sequence_y
    
    with open(arguments.output_file+'.params','w') as f:
        json.dump(params, f, indent=4)

    return lut

if __name__ == "__main__":
    ## following is an example for pure base compression
    args = get_argument()
    output_path = args.output_path
    args.base_file = args.fasta_path

    if output_path is None:
        output_path = os.path.splitext(args.base_file)[0]
    args.output_file = output_path
    args.params_file = None

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # load model and pretrained-weights
    params = {'context_length':args.context_length, 'chars':args.chars, 'predict_base_num':args.predict_base_num, 'trainable':False}
    compress_model = getattr(model_defination, args.model_name)(params)
    compress_model.load_weights(args.pretrained_model)
    if args.bi_direction: ## bidirectional flag
        backward_model = getattr(model_defination, args.model_name)(params)
        backward_model.load_weights(args.backward_model)

    # create temp directory
    args.temp_dir = tempfile.mkdtemp()
    args.temp_file_prefix = os.path.join(args.temp_dir, "compressed")
    # lut = {}
    lut=None
    if args.bi_direction:
        # lut = bi_compress(args, (compress_model, backward_model), lut=lut, stride=args.predict_base_num, sequence_y=True)
        lut = compress(args, compress_model, backward_model=backward_model, bi_direction=True, lut=lut, stride=args.predict_base_num, sequence_y=True)
    else:
        lut = compress(args, compress_model, lut=lut, stride=args.predict_base_num, sequence_y=True)

    # lut = compress(args, compress_model, lut,stride=args.predict_base_num, sequence_y=True)
    shutil.rmtree(args.temp_dir)
    pickle.dump(lut, open('subset_lut.pkl', 'wb'))
    print("saved to %s" % output_path)
