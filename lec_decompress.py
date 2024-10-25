import pickle
import numpy as np
import argparse
from utils.arithmetic_coding import arithmeticcoding_fast
from utils.utils import var_int_decode
import json
import model_defination
import os, glob

import time
# from utils.dataset_utils import int2char_dict
from utils.dataset_utils import get_one_hot
from keras import backend as K
from decompress import decompress

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters for decompression')
    parser.add_argument("--file_dir",  type=str, required=True, help="directory of compressed files, should be the same species")
    parser.add_argument("--output_dir", type=str, required=True, help="output dir for decompressed file")
    parser.add_argument("--hash_flag", action='store_true', help="hash flag")
    args1 = parser.parse_args()

    file_dir = args1.file_dir
    output_dir = args1.output_dir
    # hash_flag = args1.hash_flag
    log_file = os.path.join(output_dir, 'decompress_log.txt')

    compressed_path_list = glob.glob(os.path.join(file_dir, '*.lec'))

    os.makedirs(output_dir, exist_ok=True)
    w_f = open(log_file, 'a')
    w_f.write('File name,Decompression time\n')

    prev_model_name = ' '
    hash_path = os.path.join(output_dir, 'hash_dict.pkl')
    if not os.path.isfile(hash_path):
        h_d = None
    else:
        h_d = pickle.load(open(hash_path, 'rb'))
    for compressed_path in compressed_path_list:
        # root, ext = os.path.splitext(compressed_path)
        # id = os.path.basename(root)
        param_path = compressed_path.replace('.lec', '.params')
        basename = os.path.basename(compressed_path)
        output_path = os.path.join(output_dir, basename.replace('.lec', ''))
        print('INFO: Decompress %s to %s' % (compressed_path,output_path))
        if os.path.exists(output_path):
            print("%s exists"%output_path)
            continue

        json_dict = json.load(open(param_path, 'r'))
        # json_dict = json.load(open(param_path, 'r'))
        # t_args = argparse.Namespace()
        # t_args.__dict__.update(json_dict)

        parser2 = argparse.ArgumentParser()
        t_args = argparse.Namespace()
        t_args.__dict__.update(json_dict)
        args = parser.parse_args(namespace=t_args)

        args.input_file = compressed_path
        args.output_file_name = output_path

        if 'bi_direction' not in args:
            args.bi_direction=False

        ## load model
        if args.model_name != prev_model_name:
            model_params = {'context_length':args.context_length, 'chars':args.chars, 'predict_base_num':args.predict_base_num, 'trainable':False}
            model = getattr(model_defination, args.model_name)(model_params)
            model.load_weights(args.pretrained_model)
            if args.bi_direction:
                bkwd_model = getattr(model_defination, args.model_name)(model_params)
                bkwd_model.load_weights(args.backward_model)
            prev_model_name = args.model_name

        start_time = time.time()
        try:
            if not args.bi_direction:
                h_d = decompress(args, model, h_d) # re-use the hash dict if not None
            else:
                h_d = decompress(args, model, backward_model=bkwd_model, hash_dict=h_d)
        except :
            print("INFO: Decompression Failed for %s." % args.input_file)

        # try:
        #     decompress(args, model)
        # except:
        #     w_f.write('%s,failed\n'%basename)
        #     continue
        end_time = time.time()
        duration=end_time-start_time
        print("%.2fs in total."%duration)
        w_f.write('%s,%s\n'%(basename, str(duration)))
        w_f.flush()

        ## save hash_table for reuse
        # print("Stage hash table to %s" % hash_path)
        # pickle.dump(h_d, open(hash_path, 'wb'))
