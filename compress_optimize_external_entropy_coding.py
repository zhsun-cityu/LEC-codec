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
    # function description:
    # chars: alphabet;
    # digit_arr: a integer vector contains the index of each bases
    #   e.g. [2] represents G; [0,1,3] represents ACT
    #   digit_arr can contain multiple base
    # return: the index of the base vector, calculated as 4^2 *idx(A)+4^1 * idx(C)+4^0*idx(T)
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

    ibs_stride_lcm = lcm(infer_batch_size, stride)
    bs_stride_lcm = lcm(batch_size, stride)
    bs_ibs_stride_lcm = lcm(ibs_stride_lcm, batch_size)

    alphabet_size = len(chars)

    infer_l = int(math.floor((len(digit_x)/infer_batch_size)//stride)*stride*infer_batch_size)
    l = infer_l

    num_iters = int(math.floor(l/batch_size-context_length)) ## no need divide stride, cause the result is stored in array
    infer_num_iters = int(math.floor((l/infer_batch_size)/stride))

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

        ## process residule part (e.g. 6*50 < 328, process the residule 28 bases)
        if (infer_num_iters * stride) < int(infer_l/infer_batch_size):
            resid_num = int(l/infer_batch_size) - (infer_num_iters*stride)

            prob_vector = compress_model.predict(onehot_x[infer_ind,:], batch_size=infer_batch_size)

            prob_vector = np.transpose(prob_vector, (1,0,2)) # transpose to (predict_num, bs, base_num)
            assert prob_vector.shape[0]==stride ##

            prob_vector = prob_vector[:resid_num]

            for s in range(resid_num):
                prob_vector_array[infer_ind, ...] = prob_vector[s]
                infer_ind += 1

        else:
            print("No residue part")
            pass

    else:
        raise NotImplementedError
   ### NN inference finished

    # print("Encoding......")
    # open compressed files and compress first few characters using uniform distribution
    parallel_cumul_list = [[]] * batch_size
    parallel_symbol_list = [[]] * batch_size
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
                parallel_cumul_list[i].append(cumul)
                parallel_symbol_list[i].append(digit_x[ind[i], j])
                # enc[i].write(cumul, digit_x[ind[i],j])
                context_list[i].append(int2char[digit_x[ind[i],j]])

        for i in range(batch_size):
            for j in range(k, min(arguments.context_length, num_iters)):
                context = ''.join(context_list[i])
                prob = np.array(prob_dict[context])
                cumul[1:] = np.cumsum(prob*int(1e7) + 1)

                parallel_cumul_list[i].append(cumul)
                parallel_symbol_list[i].append(digit_x[ind[i], j])
                # enc[i].write(cumul, digit_x[ind[i],j])

                # if len(context_list[i]) > n:
                context_list[i].append(int2char[digit_x[ind[i],j]])
                context_list[i].popleft()

    else: # uniform initialize probability
        prob = np.ones(alphabet_size)/alphabet_size ## uniform probability initialization
        cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(prob*int(1e7) + 1)
        for i in range(batch_size):
            for j in range(min(arguments.context_length, num_iters)):
                parallel_cumul_list[i].append(cumul)
                parallel_symbol_list[i].append(digit_x[ind[i], j])
                # enc[i].write(cumul, digit_x[ind[i],j])

    ## Encode NN inference part
    ## TODO: accelerate this part
    cumul = np.zeros((batch_size, alphabet_size+1), dtype = np.uint64)
    for j in tqdm.tqdm(range(num_iters), desc='Entropy Coding'): ## encode the prob_dist one-by-one
        prob_vector = prob_vector_array[ind, ...] # (batch_size, alphabet_size)

        cumul[:, 1:] = np.cumsum(prob_vector*int(1e7)+1, axis=1)
        for i in range(batch_size):
            # enc[i].write(cumul[i, :], digit_y[ind[i]])
            parallel_cumul_list[i].append(cumul[i, :])
            parallel_symbol_list[i].append(digit_y[ind[i]])

        ind += 1

    last_cumul_list = []
    last_symbol_list = []

    ### Encode last part
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
            last_cumul_list.append(cumul)
            last_symbol_list.append(digit_y[i+k])            

    parallel_cumul_array=np.array([np.array(xi) for xi in parallel_cumul_list])
    parallel_symbol_array=np.array([np.array(xi) for xi in parallel_symbol_list])
    last_cumul_array = np.array(last_cumul_list)
    last_symbol_array = np.array(last_symbol_list)
    
    return parallel_cumul_array, parallel_symbol_array, last_cumul_array, last_symbol_array

def bi_infer_compression(digit_x, digit_y, onehot_x, compress_model_tuple, arguments, final_step=False, markov_model=False, lut=None, stride=1, sequence_y=True):
    # convert digit X into one-hot X
    chars = arguments.chars
    int2char = int2char_dict(chars)

    context_length = arguments.context_length
    batch_size = arguments.batch_size // 2 ## halve the gob cause context is shared
    alphabet_size = len(chars)

    fwd_model, bkwd_model = compress_model_tuple

    if not final_step:
        l = int( math.floor((len(digit_x))/(batch_size*2) ) * (batch_size*2) ) ## inference length
        num_iters = int(math.floor(l/(batch_size*2)-context_length)/stride)

        ind = np.array(range((batch_size*2))) * int(l/(batch_size*2))

        # open compressed files and compress first few characters to start markov chain
        # using uniform distribution
        f = [open(arguments.temp_file_prefix+'.' + str(i),'wb') for i in range(batch_size)]
        bitout = [arithmeticcoding_fast.BitOutputStream(f[i]) for i in range(batch_size)]
        enc = [arithmeticcoding_fast.ArithmeticEncoder(32, bitout[i]) for i in range(batch_size)]

        fwd_ind = ind[1::2].copy()
        ## do forward prediction
        print("forward predicting......")
        ### encode context part
        ### this part can be optimized
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
                for j in range(k, min(arguments.context_length, num_iters)):
                    context = ''.join(context_list[i])
                    prob = np.array(prob_dict[context])
                    cumul[1:] = np.cumsum(prob*int(1e7) + 1)
                    #enc[i].write(cumul, X[ind[i],j])
                    enc[i].write(cumul, digit_x[fwd_ind[i],j])
                    context_list[i].append(int2char[digit_x[fwd_ind[i],j]])
                    context_list[i].popleft()

        else: # uniform initialize probability
            prob = np.ones(alphabet_size)/alphabet_size ## uniform probability initialization
            cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
            cumul[1:] = np.cumsum(prob*int(1e7) + 1)
            for i in range(batch_size):
                for j in range(min(arguments.context_length, num_iters)):
                    enc[i].write(cumul, digit_x[ind[i],j])

        if sequence_y:
            cumul = np.zeros((batch_size, alphabet_size+1), dtype = np.uint64)
            # for j in tqdm.tqdm(range(0, num_iters - arguments.context_length)): # !! range should change, 0, num_iters, add stride
            for j in tqdm.tqdm(range(num_iters)):
                if lut is None:
                    prob_vector = fwd_model.predict(onehot_x[fwd_ind,:], batch_size=batch_size)
                else:
                    ## use lut constructing prob
                    # check digit_x
                    prob_vector = np.zeros((batch_size, alphabet_size))
                    infer_idx_list = []
                    for i in range(len(digit_x[fwd_ind, :])):
                        if digit_x[fwd_ind[i]].tostring() in lut:
                            prob_vector[i] = lut[digit_x[fwd_ind[i]].tostring()]
                        else:
                            infer_idx_list.append(i)
                    infer_idx_arr = np.array(infer_idx_list)
                    prob_vector[infer_idx_arr] = fwd_model.predict(onehot_x[infer_idx_arr], batch_size=len(infer_idx_list))

                    ## record in lut
                    for idx in infer_idx_arr:
                        lut[digit_x[fwd_ind[idx]].tostring()] = prob[idx]

                # if len(prob_vector)>2: ## if model predicted more than one base
                prob_vector = np.transpose(prob_vector, (1,0,2)) # transpose to (predict_num, bs, base_num)
                assert prob_vector.shape[0]==stride ##

                for k in range(len(prob_vector)):
                    prob_dist = prob_vector[k]
                    cumul[:,1:] = np.cumsum(prob_dist*int(1e7) + 1, axis = 1)

                    ## write to file
                    ## this part can be optimized
                    for i in range(batch_size):
                        enc[i].write(cumul[i,:], digit_y[fwd_ind[i]])
                    fwd_ind += 1

            ## process forward residual part of each gob
            if (num_iters * stride + context_length) < int(l/(batch_size * 2)):
                # process residue part
                fwd_ind = np.array(range(1, batch_size+1)) * int(l/(batch_size * 2)) - stride - context_length
                resid_num = int(l/(batch_size * 2)) - (num_iters*stride + context_length)
                prob_vector = fwd_model.predict(onehot_x[fwd_ind,:], batch_size=batch_size)
                # if len(prob_vector)>2: ## if model predicted more than one base
                prob_vector = np.transpose(prob_vector, (1,0,2)) # transpose to (predict_num, bs, base_num)
                assert prob_vector.shape[0]==stride ##
                prob_vector = prob_vector[-resid_num:]
                fwd_ind += (stride-resid_num)
                for k in range(len(prob_vector)):
                    prob_dist = prob_vector[k]
                    cumul[:,1:] = np.cumsum(prob_dist*int(1e7) + 1, axis = 1)

                    for i in range(batch_size):
                        enc[i].write(cumul[i,:], digit_y[fwd_ind[i]])
                    fwd_ind += 1
            else:
                print("No residue part")
                pass
        else:
            raise NotImplementedError

        ### do backward prediction, backward prediction is a bit more than the
        bkwd_ind = ind[1::2].copy() ## bkwd_ind should be the same as fwd_ind, but inference in another direction
        # construct backward data from forward data
        print("Backward predicting......")
        if sequence_y:
            cumul = np.zeros((batch_size, alphabet_size+1), dtype = np.uint64)
            ## TODO: check if num_iter or num_iter+1
            for j in tqdm.tqdm(range(num_iters+context_length)): ## encode context_length more bases compared with forward prediction
                if lut is None:
                    ## TODO: check onehot_x[bkwd_ind,:][::-1]
                    prob_vector = bkwd_model.predict(np.flip(onehot_x[bkwd_ind,:],axis=1), batch_size=batch_size)
                else:
                    ## use lut constructing prob
                    # check digit_x
                    prob_vector = np.zeros((batch_size, alphabet_size))
                    infer_idx_list = []
                    for i in range(len(digit_x[bkwd_ind, :])):
                        if digit_x[bkwd_ind[i]].tostring() in lut:
                            prob_vector[i] = lut[digit_x[bkwd_ind[i]].tostring()]
                        else:
                            infer_idx_list.append(i)
                    infer_idx_arr = np.array(infer_idx_list)
                    ## TODO: check onehot_x[infer_idx_arr][::-1]
                    prob_vector[infer_idx_arr] = bkwd_model.predict(np.flip(onehot_x[infer_idx_arr], axis=1), batch_size=len(infer_idx_list))

                    ## record in lut
                    for idx in infer_idx_arr:
                        lut[digit_x[bkwd_ind[idx]].tostring()] = prob[idx]

                prob_vector = np.transpose(prob_vector, (1,0,2))
                assert prob_vector.shape[0]==stride ##

                for k in range(len(prob_vector)):
                    prob_dist = prob_vector[k]
                    cumul[:,1:] = np.cumsum(prob_dist*int(1e7) + 1, axis = 1)

                    ## write to file
                    ## this part can be optimized
                    for i in range(batch_size):
                        # TODO: check onehot_x[bkwd_ind][i][0]
                        enc[i].write(cumul[i, :], np.argmax(onehot_x[bkwd_ind-1][i][0])) ## encode the
                    bkwd_ind -= 1 ## backward, thus minus 1

            ## process backward residual part of each gob
            ## TODO: check condition (num_iters * stride + context_length) < int(l/batch_size)
            if (num_iters * stride + context_length) < int(l/(batch_size*2)):
                # TODO: check bkwd_ind
                # bkwd_ind = np.array(range(1, batch_size+1)) * int(l/batch_size) - stride - context_length
                resid_num = int(l/(batch_size*2)) - (num_iters*stride + context_length)
                g_num=len(ind[1::2])
                bkwd_ind = ind[::2][:g_num]
                bkwd_ind+=resid_num

                ## TODO: check .predict(onehot_x[bkwd_ind,:][::-1], batch_size=batch_size)
                prob_vector = bkwd_model.predict(np.flip(onehot_x[bkwd_ind,:], axis=1), batch_size=batch_size)

                # if len(prob_vector)>2: ## if model predicted more than one base
                prob_vector = np.transpose(prob_vector, (1,0,2)) # transpose to (predict_num, bs, base_num)
                assert prob_vector.shape[0]==stride ##

                prob_vector = prob_vector[:resid_num]## only keep residual part
                for k in range(len(prob_vector)):
                    prob_dist = prob_vector[k]
                    cumul[:,1:] = np.cumsum(prob_dist*int(1e7) + 1, axis = 1)

                    for i in range(batch_size):
                        # enc[i].write(cumul[i,:], digit_y[bkwd_ind[i]])
                        enc[i].write(cumul[i, :], np.argmax(onehot_x[bkwd_ind-1][i][0])) ## encode the
                    bkwd_ind -= 1

            else:
                print("No residue part")
                pass
        else:
            raise NotImplementedError

        ## close files
        for i in range(batch_size):
            enc[i].finish()
            bitout[i].close()
            f[i].close()

    ## if the rest part not enough for a complete forward part and complete backward part,
    ##   use forward model to predict the rest part
    else:
        f = open(arguments.temp_file_prefix+'.last','wb')
        bitout = arithmeticcoding_fast.BitOutputStream(f)
        enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
        prob = np.ones(int(alphabet_size))/alphabet_size
        cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(prob*int(1e7) + 1)

        ## encode the context part with uniform distribution
        for j in range(arguments.context_length):
            enc.write(cumul, digit_x[0,j])

        ## encode the rest part with forward prediction
        for i in (range(0, len(onehot_x), stride)):
            if lut is None:
                prob_vector = compress_model.predict(onehot_x[i][np.newaxis, ...], batch_size=1)
            else:
                # TODO: use lut constructing prob_vector
                # pass
                if digit_x[i].tostring() in lut:
                    prob_vector = lut[digit_x[i].tostring()]
                else:
                    prob_vector = compress_model.predict(onehot_x[i][np.newaxis, ...], batch_size=1)
                    lut[digit_x[i].tostring()] = prob_vector

            if sequence_y:
                # if len(prob_vector)>2: ## if model predicted more than one base
                prob_vector = np.transpose(prob_vector, (1,0,2)) # transpose to (predict_num, bs, base_num)
                assert prob_vector.shape[0]==stride ##

                for k in range(len(prob_vector)):
                    prob_dist = prob_vector[k]
                    cumul[1:] = np.cumsum(prob_dist*int(1e7) + 1)
                    enc.write(cumul, digit_y[i])
                    i += 1
            else:
                raise NotImplementedError

        ## residual checking
        residual = len(onehot_x)%stride ## may have bug when stride is larger then context_length
        if residual != 0:
            index = len(onehot_x)-stride - 1
            prob_vector = compress_model.predict(onehot_x[index][np.newaxis, ...], batch_size=1)
            if sequence_y:
                # if len(prob_vector)>2: ## if model predicted more than one base
                prob_vector = np.transpose(prob_vector, (1,0,2)) # transpose to (predict_num, bs, base_num)
                assert prob_vector.shape[0]==stride ##
                for k in range(len(prob_vector)):
                    prob_dist = prob_vector[k]
                    cumul[1:] = np.cumsum(prob_dist*int(1e7) + 1)
                    enc.write(cumul, digit_y[i])
                    i += 1
            else:
                raise NotImplementedError

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
    if batch_size % 2 != 0  and batch_size != 1:
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

    ## model parameter
    params['model_name'] = arguments.model_name
    params['pretrained_model'] = arguments.pretrained_model
    params['prob_dict_file'] = arguments.prob_dict_file
    ## compressing parameter
    params['sequence_length'] = text_len
    params['context_length'] = arguments.context_length
    params['predict_base_num'] = arguments.predict_base_num ## predict stride
    params['chars'] = arguments.chars
    params['batch_size'] = arguments.batch_size
    params['infer_batch_size'] = arguments.infer_batch_size
    params['combined_num'] = arguments.combined_num
    params['markov_model'] = arguments.markov_model
    params['sequence_y'] = sequence_y
    ## save parameter to file for decoding
    with open(arguments.output_file+'.params','w') as f:
        json.dump(params, f, indent=4)

    digit_text, digit_x, digit_y, onehot_x, _ = prepare_dataset_for_compression(arguments.base_file, fasta_flag=False, context_length=context_length, stride=stride, sequence_y=sequence_y)

    infer_l = int(math.floor( (len(digit_x)/infer_batch_size))//stride * stride * infer_batch_size) ## this length is the nn model inference length

    start = time.time()

    if not bi_direction:
        parallel_cumul_array, parallel_symbol_array, last_cumul_array, last_symbol_array = infer_compression(digit_x, digit_y, onehot_x, compress_model, arguments, lut=lut, stride=stride)
        data_path = os.path.join(arguments.temp_dir, 'data.npy')
        batch_length = parallel_symbol_array.shape[1]
        last_length = last_symbol_array.shape[0]
        with open(data_path, 'wb') as f:
            np.save(f, batch_size)
            np.save(f, batch_length)
            np.save(f, last_length)
            np.save(f, parallel_cumul_array)
            np.save(f, parallel_symbol_array)
            np.save(f, last_cumul_array)
            np.save(f, last_symbol_array)
        print("encode information saved to %s" % data_path)
        # TODO: encode with c++ program ( this program should generate binary file directly from the cumul and symbols)

    else: ## use bi-directional prediction
        ## TODO: check if bi_infer_compression check length of digit_x
        lut = bi_infer_compression(digit_x[:infer_l, :], digit_y[:infer_l], onehot_x[:infer_l], compress_model_tuple, arguments, lut=lut, stride=stride)

        #### TODO area
        ## This should be better
        if infer_l < text_len-context_length:
            lut = bi_infer_compression(digit_x[infer_l:,:], digit_y[infer_l:], onehot_x[infer_l:], compress_model_tuple, arguments, lut=lut, final_step=True, stride=stride)
        else:
            ## encode rest of the stream with uniform distribution
            f = open(arguments.temp_file_prefix+'.last','wb')
            bitout = arithmeticcoding_fast.BitOutputStream(f)
            enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)

            base_prob = np.ones(alphabet_size)/alphabet_size
            cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
            cumul[1:] = np.cumsum(base_prob*int(1e7) + 1)
            for j in range(infer_l, text_len):
                enc.write(cumul, digit_text[j])
                enc.finish()
                bitout.close()
                f.close()

    ## write all bitstream to one file
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
        lut = bi_compress(args, (compress_model, backward_model), lut=lut, stride=args.predict_base_num, sequence_y=True)
    else:
        lut = compress(args, compress_model, lut=lut, stride=args.predict_base_num, sequence_y=True)

    # lut = compress(args, compress_model, lut,stride=args.predict_base_num, sequence_y=True)
    shutil.rmtree(args.temp_dir)
    pickle.dump(lut, open('subset_lut.pkl', 'wb'))
    print("saved to %s" % output_path)
