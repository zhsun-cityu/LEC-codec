import math
import numpy as np
import pickle
import tempfile
import shutil
from collections import deque
from tqdm import tqdm
import struct
import argparse
from utils.utils import var_int_decode
from utils.arithmetic_coding import arithmeticcoding_fast
from utils.dataset_utils import get_one_hot
import os

def get_prediction_with_hash_dict(model, context, alphabet_size, hash_dict):
    if len(context) != 1:
        raise NotImplementedError
    use_hash_table=0
    str_ctx = np.array2string(context[0], separator='')[1:-1][::2]
    if str_ctx in hash_dict:
        prob_vector = hash_dict[str_ctx]
        use_hash_table+=1
    else:
        onehot_context = get_one_hot(context, alphabet_size)
        prob_vector = model.predict(onehot_context, batch_size=1)
        hash_dict[str_ctx] = prob_vector
    return prob_vector, hash_dict, use_hash_table

def infer_decompression_with_hash_dict(arguments, model, markov_model=True, hash_dict=None):
    if arguments.bi_direction: 
        raise NotImplementedError
    seq_length = arguments.sequence_length
    context_length = arguments.context_length
    batch_size = arguments.batch_size
    infer_batch_size = arguments.infer_batch_size
    # alphabet_size = len(arguments.chars)
    chars = arguments.chars
    stride = arguments.predict_base_num
    alphabet_size = len(chars)
    model_name = arguments.model_name
    id2char_dict = arguments.id2char_dict
    # int2char = int2char_dict(arguments.chars)
    if 'infer_l' not in arguments:
        l = int(math.floor( ((seq_length-context_length)/infer_batch_size))//stride * stride * infer_batch_size) ## this length is the nn model inference length
        num_iters = int(math.floor(l/batch_size - context_length)/stride)
    else:
        l = arguments.infer_l
        num_iters = arguments.num_iters

    # infer_num_iters = arguments.infer_num_iters

    if batch_size != 1:
        raise NotImplementedError ## currently only support batchsize=1

    # l = int(math.floor( ((seq_length-context_length)/infer_batch_size))//stride * stride * infer_batch_size) ## this length is the nn model inference length
    # num_iters = int(math.floor(l/batch_size - context_length)/stride)

    series = np.zeros(seq_length, dtype = np.uint8)
    series_2d = np.ones((batch_size, int(l/batch_size)), dtype=np.uint8)

    # open compressed files and decompress first few characters e
    # uniform distribution
    f = [open(arguments.temp_file_prefix+'.'+str(i),'rb') for i in range(batch_size)]
    bitin = [arithmeticcoding_fast.BitInputStream(f[i]) for i in range(batch_size)]
    dec = [arithmeticcoding_fast.ArithmeticDecoder(32, bitin[i]) for i in range(batch_size)]

    context_list = []
    hash_table_use_count = 0
    if markov_model:
        rf = open(arguments.prob_dict_file, 'rb')
        n = pickle.load(rf)
        k = n-1
        base_prob = pickle.load(rf)
        prob_dict = pickle.load(rf)

        prob = np.ones(alphabet_size)/alphabet_size
        cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(base_prob*int(1e7) + 1)

        ## decode with prior probability
        for i in range(batch_size):
            cur_dq = deque()
            for j in range(k):
                series_2d[i, j] = dec[i].read(cumul, alphabet_size)
                # context_list[i].append(series_2d[i, j]) ## the decoded base is sent to the buffer
                cur_dq.append(series_2d[i, j])
            context_list.append(cur_dq)

        ### decode context partwith markov transition matrix
        for i in range(batch_size):
            # for j in range(k, min(arguments.context_length, num_iters)):
            for j in range(k, arguments.context_length):
                ctx_list = series_2d[i, j-k:j].tolist()
                ctx = ''
                for ctx_id in ctx_list:
                    ctx+=id2char_dict[ctx_id]

                prob = np.array(prob_dict[ctx])
                cumul[1:] = np.cumsum(prob*int(1e7) + 1)
                series_2d[i, j] = dec[i].read(cumul, alphabet_size)
                context_list[i].append(series_2d[i, j])

    else:  ## decode the context part with uniform probability
        prob = np.ones(alphabet_size)/alphabet_size
        cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(prob*int(1e7) + 1)
        for i in range(batch_size):
            for j in range(min(num_iters,context_length)):
                series_2d[i,j] = dec[i].read(cumul, alphabet_size)

    ## NN inference phrase
    if arguments.sequence_y:
        # decode rest of the string with learning based model
        cumul = np.zeros((batch_size, alphabet_size+1), dtype = np.uint64)
        for j in tqdm(range(0, num_iters)):
            context = series_2d[:,(stride*j):(stride*j) + context_length] ## series_2d: (batch_size, number_iters)
            # onehot_context = get_one_hot(context, alphabet_size)
            # prob_vector = model.predict(onehot_context, batch_size=batch_size)
            prob_vector, hash_dict, use_hash_table = get_prediction_with_hash_dict(model, context, alphabet_size, hash_dict)
            hash_table_use_count+=use_hash_table
            prob_vector = np.transpose(prob_vector, (1, 0, 2))  # (base_number, batch_size, chars_num)
            assert len(prob_vector)==stride
            for k in range(stride):
                prob = prob_vector[k]
                cumul[:,1:] = np.cumsum(prob*int(1e7) + 1, axis = 1)
                for i in range(batch_size): ## decode current symbol
                    series_2d[i, (stride*j) + context_length + k] = dec[i].read(cumul[i,:], alphabet_size) 
        
        ## process resid part of the encoder
        if (num_iters * stride + context_length) < int(l/batch_size):
            resid_num = int(l/batch_size) - (num_iters*stride + context_length)
            context = series_2d[:, int(l/batch_size)-resid_num-context_length:int(l/batch_size)-resid_num]
            prob_vector, hash_dict, use_hash_table = get_prediction_with_hash_dict(model, context, alphabet_size, hash_dict)
            hash_table_use_count+=use_hash_table

            # decode            
            prob_vector = np.transpose(prob_vector, (1, 0, 2))
            prob_vector = prob_vector[:resid_num]
            for k in range(len(prob_vector)):
                prob = prob_vector[k]
                cumul[:,1:] = np.cumsum(prob*int(1e7) + 1, axis = 1)
                for i in range(batch_size): ## decode current symbol
                    series_2d[i, int(l/batch_size)-resid_num+k] = dec[i].read(cumul[i,:], alphabet_size) 

        # close files
        for i in range(batch_size):
            bitin[i].close()
            f[i].close()
    else:
        raise NotImplementedError

    # return series_2d.reshape(-1)
    series[:l] = series_2d.reshape(-1)

    # decode resid part
    # series = np.zeros(seq_length, dtype = np.uint8)
    f = open(arguments.temp_file_prefix+'.last','rb')
    bitin = arithmeticcoding_fast.BitInputStream(f)
    dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin)
    
    cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
    if arguments.sequence_y:
        for i in tqdm(range(l, seq_length, stride),desc='decode last part'):
            # convert series[i:i+context_length] to one_hot_array
            digit_ctx = series[i-context_length:i]
            onehot_context = get_one_hot(series[i-context_length:i], alphabet_size)

            prob_vector, hash_dict, use_hash_table= get_prediction_with_hash_dict(model, np.array([digit_ctx]), alphabet_size, hash_dict)
            hash_table_use_count+=use_hash_table
            
            prob_vector = np.transpose(prob_vector, (1, 0, 2))
            # decode
            for k in range(len(prob_vector)):
                if i+k==seq_length:
                    break
                prob = prob_vector[k]
                cumul[1:] = np.cumsum(prob*int(1e7) + 1)
                series[i+k] = dec.read(cumul, alphabet_size)
    else:
        raise NotImplementedError

    total_infer = num_iters + 1 + len(range(l, seq_length, stride))
    bitin.close()
    f.close()
    return series, hash_dict, hash_table_use_count, total_infer

def bi_infer_decompression(arguments, model, backward_model, markov_model=True):
    seq_length = arguments.sequence_length
    context_length = arguments.context_length
    batch_size = arguments.batch_size
    batch_size = batch_size // 2
    infer_batch_size = arguments.infer_batch_size
    # alphabet_size = len(arguments.chars)
    chars = arguments.chars
    stride = arguments.predict_base_num
    alphabet_size = len(chars)
    model_name = arguments.model_name
    id2char_dict = arguments.id2char_dict
    # int2char = int2char_dict(arguments.chars)
    if 'infer_l' in arguments:
        infer_l = arguments.infer_l
    else:
        infer_l = int(math.floor( ((seq_length-context_length)/infer_batch_size))//stride * stride * infer_batch_size) ## this length is the nn model inference length

    # num_iters = arguments.num_iters
    infer_num_iters = arguments.infer_num_iters

    # l = int(math.floor( ((seq_length-context_length)/infer_batch_size))//stride * stride * infer_batch_size) ## this length is the nn model inference length
    fwd_num_iters = int(math.floor(infer_l/(batch_size * 2) - context_length)/stride)
    bkwd_num_iters = int(math.floor(infer_l/(batch_size * 2))/stride)
    series = np.zeros(seq_length, dtype = np.uint8)
    series_2d = np.ones((batch_size, int(infer_l/batch_size)), dtype=np.uint8)
    half_pos = int(infer_l/batch_size) // 2

    # open compressed files and decompress first few characters e
    # uniform distribution
    f = [open(arguments.temp_file_prefix+'.'+str(i),'rb') for i in range(batch_size)]
    bitin = [arithmeticcoding_fast.BitInputStream(f[i]) for i in range(batch_size)]
    dec = [arithmeticcoding_fast.ArithmeticDecoder(32, bitin[i]) for i in range(batch_size)]

    # ____ ____ _  _ ___ ____ _  _ ___ 
    # |    |  | |\ |  |  |___  \/   |  
    # |___ |__| | \|  |  |___ _/\_  |  
    context_list = []
    if markov_model:
        rf = open(arguments.prob_dict_file, 'rb')
        n = pickle.load(rf)
        k = n-1
        base_prob = pickle.load(rf)
        prob_dict = pickle.load(rf)

        prob = np.ones(alphabet_size)/alphabet_size
        cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(base_prob*int(1e7) + 1)

        ## decode with prior probability
        for i in range(batch_size):
            cur_dq = deque()
            for j in range(half_pos, half_pos+k):
                series_2d[i, j] = dec[i].read(cumul, alphabet_size)
                # context_list[i].append(series_2d[i, j]) ## the decoded base is sent to the buffer
                cur_dq.append(series_2d[i, j])
            context_list.append(cur_dq)

        ## decode with transition matrix
        for i in range(batch_size):
            # for j in range(k, min(arguments.context_length, num_iters)):
            for j in range(half_pos+k, half_pos+arguments.context_length):
                ctx_list = series_2d[i, j-k:j].tolist()
                ctx = ''
                for ctx_id in ctx_list:
                    ctx+=id2char_dict[ctx_id]

                prob = np.array(prob_dict[ctx])
                cumul[1:] = np.cumsum(prob*int(1e7) + 1)
                series_2d[i, j] = dec[i].read(cumul, alphabet_size)
                context_list[i].append(series_2d[i, j])

    else:  ## decode the context part with uniform probability
        prob = np.ones(alphabet_size)/alphabet_size
        cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(prob*int(1e7) + 1)
        for i in range(batch_size):
            for j in range(half_pos, half_pos+context_length):
                series_2d[i,j] = dec[i].read(cumul, alphabet_size)

    # ____ ____ ____ _ _ _ ____ ____ ___  
    # |___ |  | |__/ | | | |__| |__/ |  \ 
    # |    |__| |  \ |_|_| |  | |  \ |__/ 
                                        
    if not arguments.sequence_y:
        raise NotImplementedError
    # decode rest of the string with learning based model
    cumul = np.zeros((batch_size, alphabet_size+1), dtype = np.uint64)
    for j in tqdm(range(0, fwd_num_iters), desc='Decoding forward part'):
        context = series_2d[:,half_pos+(stride*j):half_pos+(stride*j) + context_length] ## series_2d: (batch_size, number_iters)
        onehot_context = get_one_hot(context, alphabet_size)
        prob_vector = model.predict(onehot_context, batch_size=batch_size)
        prob_vector = np.transpose(prob_vector, (1, 0, 2))  # (base_number, batch_size, chars_num)
        assert len(prob_vector)==stride
        for k in range(stride):
            prob = prob_vector[k]
            cumul[:,1:] = np.cumsum(prob*int(1e7) + 1, axis = 1)
            for i in range(batch_size): ## decode current symbol
                series_2d[i, half_pos+(stride*j) + context_length + k] = dec[i].read(cumul[i,:], alphabet_size)
        
    # ___  ____ ____ _  _ _ _ _ ____ ____ ___  
    # |__] |__| |    |_/  | | | |__| |__/ |  \ 
    # |__] |  | |___ | \_ |_|_| |  | |  \ |__/ 
    ctx_skip = context_length
    cumul = np.zeros((batch_size, alphabet_size+1), dtype = np.uint64)
    for j in tqdm(range(0, bkwd_num_iters), desc='Decoding backward part'):
        context = series_2d[:,half_pos-(stride*j):half_pos-(stride*j) + context_length] ## check if the stride is correct
        context = context[:, ::-1] ## context is reversed
        onehot_context = get_one_hot(context, alphabet_size)
        prob_vector = backward_model.predict(onehot_context, batch_size=batch_size)
        # prob_vector=prob_vector.round(4)
        prob_vector = np.transpose(prob_vector, (1, 0, 2))  # (base_number, batch_size, chars_num)
        prob_vector = prob_vector[::-1]
        assert len(prob_vector)==stride
        for k in range(stride):
            # if ctx_skip > 0: ## skip the context part
            #     ctx_skip -= 1
            #     continue
            prob = prob_vector[k]
            cumul[:,1:] = np.cumsum(prob*int(1e7) + 1, axis = 1)
            for i in range(batch_size): ## decode current symbol
                series_2d[i, half_pos-(stride*j) - k - 1] = dec[i].read(cumul[i,:], alphabet_size)

    # close files
    for i in range(batch_size):
        bitin[i].close()
        f[i].close()

    # return series_2d.reshape(-1)
    series[:infer_l] = series_2d.reshape(-1)

    # decode resid part
    # series = np.zeros(seq_length, dtype = np.uint8)
    f = open(arguments.temp_file_prefix+'.last','rb')
    bitin = arithmeticcoding_fast.BitInputStream(f)
    dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin)
    
    cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
    if not arguments.sequence_y:
        raise NotImplementedError

    for i in tqdm(range(infer_l, seq_length, stride),desc='Decode last part'):
        # convert series[i:i+context_length] to one_hot_array
        onehot_context = get_one_hot(series[i-context_length:i], alphabet_size)
        prob_vector = model.predict(onehot_context[np.newaxis, ...], batch_size=1)
        prob_vector = np.transpose(prob_vector, (1, 0, 2))
        for k in range(len(prob_vector)):
            if i+k==seq_length:
                break
            prob = prob_vector[k]
            cumul[1:] = np.cumsum(prob*int(1e7) + 1)
            series[i+k] = dec.read(cumul, alphabet_size)

    bitin.close()
    f.close()
    return series

def infer_decompression(arguments, model, backward_model=None, markov_model=True):
    seq_length = arguments.sequence_length
    context_length = arguments.context_length
    batch_size = arguments.batch_size
    infer_batch_size = arguments.infer_batch_size
    # alphabet_size = len(arguments.chars)
    chars = arguments.chars
    stride = arguments.predict_base_num
    alphabet_size = len(chars)
    model_name = arguments.model_name
    id2char_dict = arguments.id2char_dict
    # int2char = int2char_dict(arguments.chars)

    if 'infer_l' in arguments:
        infer_l = arguments.infer_l
    else:
        infer_l = int(math.floor( ((seq_length-context_length)/infer_batch_size))//stride * stride * infer_batch_size) ## this length is the nn model inference length

    # num_iters = arguments.num_iters
    # infer_num_iters = arguments.infer_num_iters

    # infer_l = int(math.floor( ((seq_length-context_length)/infer_batch_size))//stride * stride * infer_batch_size) ## this length is the nn model inference length
    num_iters = int(math.floor(infer_l/batch_size - context_length)/stride)

    series = np.zeros(seq_length, dtype = np.uint8)
    series_2d = np.ones((batch_size, int(infer_l/batch_size)), dtype=np.uint8)

    # open compressed files and decompress first few characters e
    # uniform distribution
    f = [open(arguments.temp_file_prefix+'.'+str(i),'rb') for i in range(batch_size)]
    bitin = [arithmeticcoding_fast.BitInputStream(f[i]) for i in range(batch_size)]
    dec = [arithmeticcoding_fast.ArithmeticDecoder(32, bitin[i]) for i in range(batch_size)]

    context_list = []
    if markov_model:
        rf = open(arguments.prob_dict_file, 'rb')
        n = pickle.load(rf)
        k = n-1
        base_prob = pickle.load(rf)
        prob_dict = pickle.load(rf)

        prob = np.ones(alphabet_size)/alphabet_size
        cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(base_prob*int(1e7) + 1)

        ## decode with prior probability
        for i in range(batch_size):
            cur_dq = deque()
            for j in range(k):
                series_2d[i, j] = dec[i].read(cumul, alphabet_size)
                # context_list[i].append(series_2d[i, j]) ## the decoded base is sent to the buffer
                cur_dq.append(series_2d[i, j])
            context_list.append(cur_dq)

        ## decode with transition matrix
        for i in range(batch_size):
            # for j in range(k, min(arguments.context_length, num_iters)):
            for j in range(k, arguments.context_length):
                ctx_list = series_2d[i, j-k:j].tolist()
                ctx = ''
                for ctx_id in ctx_list:
                    ctx+=id2char_dict[ctx_id]

                prob = np.array(prob_dict[ctx])
                cumul[1:] = np.cumsum(prob*int(1e7) + 1)
                series_2d[i, j] = dec[i].read(cumul, alphabet_size)
                context_list[i].append(series_2d[i, j])

    else:  ## decode the context part with uniform probability
        prob = np.ones(alphabet_size)/alphabet_size
        cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(prob*int(1e7) + 1)
        for i in range(batch_size):
            for j in range(min(num_iters,context_length)):
                series_2d[i,j] = dec[i].read(cumul, alphabet_size)
    if arguments.sequence_y:
        # decode rest of the string with learning based model
        cumul = np.zeros((batch_size, alphabet_size+1), dtype = np.uint64)
        for j in tqdm(range(0, num_iters)):
            context = series_2d[:,(stride*j):(stride*j) + context_length] ## series_2d: (batch_size, number_iters)
            onehot_context = get_one_hot(context, alphabet_size)
            prob_vector = model.predict(onehot_context, batch_size=batch_size)
            prob_vector = np.transpose(prob_vector, (1, 0, 2))  # (base_number, batch_size, chars_num)
            assert len(prob_vector)==stride
            for k in range(stride):
                prob = prob_vector[k]
                cumul[:,1:] = np.cumsum(prob*int(1e7) + 1, axis = 1)
                for i in range(batch_size): ## decode current symbol
                    series_2d[i, (stride*j) + context_length + k] = dec[i].read(cumul[i,:], alphabet_size) 
        
        ## process resid part of the encoder
        ## UPDATE: should be no resid part now
        if (num_iters * stride + context_length) < int(infer_l/batch_size):
            resid_num = int(infer_l/batch_size) - (num_iters*stride + context_length)
            context = series_2d[:, int(infer_l/batch_size)-resid_num-context_length:int(infer_l/batch_size)-resid_num]
            onehot_context = get_one_hot(context, alphabet_size)
            prob_vector = model.predict(onehot_context, batch_size=batch_size)
            prob_vector = np.transpose(prob_vector, (1, 0, 2))
            # prob_vector = prob_vector[-resid_num:]
            prob_vector = prob_vector[:resid_num]
            for k in range(len(prob_vector)):
                prob = prob_vector[k]
                cumul[:,1:] = np.cumsum(prob*int(1e7) + 1, axis = 1)
                for i in range(batch_size): ## decode current symbol
                    series_2d[i, int(infer_l/batch_size)-resid_num+k] = dec[i].read(cumul[i,:], alphabet_size) 
        else:
            print("No resid part to decode.")

        # close files
        for i in range(batch_size):
            bitin[i].close()
            f[i].close()
    else:
        raise NotImplementedError

    series[:infer_l] = series_2d.reshape(-1)

    # decode last part
    f = open(arguments.temp_file_prefix+'.last','rb')
    bitin = arithmeticcoding_fast.BitInputStream(f)
    dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin)
    
    cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
    if arguments.sequence_y:
        for i in tqdm(range(infer_l, seq_length, stride),desc='decode last part'):
            # convert series[i:i+context_length] to one_hot_array
            onehot_context = get_one_hot(series[i-context_length:i], alphabet_size)
            prob_vector = model.predict(onehot_context[np.newaxis, ...], batch_size=1)
            prob_vector = np.transpose(prob_vector, (1, 0, 2))
            for k in range(len(prob_vector)):
                if i+k==seq_length:
                    break
                prob = prob_vector[k]
                cumul[1:] = np.cumsum(prob*int(1e7) + 1)
                series[i+k] = dec.read(cumul, alphabet_size)
    else:
        raise NotImplementedError

    bitin.close()
    f.close()
    return series

def decompress(args, model, backward_model=None, hash_dict=None):
    args.temp_dir = tempfile.mkdtemp()
    args.temp_file_prefix = os.path.join(args.temp_dir, "compressed")

    ## get paramters from args

    seq_length = args.sequence_length
    if args.bi_direction:
        batch_size = args.batch_size // 2
    else:
        batch_size = args.batch_size
    infer_batch_size = args.infer_batch_size
    stride = args.predict_base_num
    context_length = args.context_length
    chars = args.chars
    alphabet_size = len(chars)
    hash_flag = args.hash_flag
    bi_direction = args.bi_direction
    
    # id2char_dict = args.id2char_dict
    id2char_dict = {i:c for i,c in enumerate(chars)}
    args.id2char_dict = id2char_dict
    f = open(args.input_file,'rb')
    for i in range(batch_size): ## read each batch from bitstream file
        f_out = open(args.temp_file_prefix+'.'+str(i),'wb')
        byte_str_len = var_int_decode(f)
        byte_str = f.read(byte_str_len)
        f_out.write(byte_str)
        f_out.close()
    f_out = open(args.temp_file_prefix+'.last','wb') ## read the residule bitstream file
    byte_str_len = var_int_decode(f)
    byte_str = f.read(byte_str_len)
    f_out.write(byte_str)
    f_out.close()
    f.close()

    series = np.zeros(seq_length,dtype=np.uint8)

    l = int(((seq_length-context_length)/infer_batch_size))//stride * stride * infer_batch_size
    
    num_iters = int(math.floor(l/batch_size - context_length)/stride)
    
    ## some encoding error for small sequence, will fix later
    if num_iters <= 0:
        print("small sequence encoding error, skipping %s" % os.path.basename(args.output_file_name))
        return hash_dict

    if hash_flag:
        if hash_flag and hash_dict is None:
            print("initialize hash table")
            hash_dict = {}
        series, hash_dict, hash_table_use_count, total_infer = infer_decompression_with_hash_dict(args, model, markov_model=True, hash_dict=hash_dict)
    elif not bi_direction: ## hash_dict will not update
        series = infer_decompression(args, model, markov_model=True)
    else: ## decoding with bi_directional inference
        series = bi_infer_decompression(args, model, backward_model, markov_model=True)

    ## write data to file ## need
    f = open(args.output_file_name,'w')
    
    out_list = []
    for s in series:
        out_list.append(id2char_dict[s])
    out_str =''.join(out_list)
    f.write(out_str)
    f.close()
    shutil.rmtree(args.temp_dir)

    # if not hash_dict:
    #     print("INFO: Hash table used %d time in this iteration.\nUsed percentage: %f.\nCurrent hash table size: %d" %(hash_table_use_count, hash_table_use_count/total_infer, len(hash_dict)))
    return hash_dict