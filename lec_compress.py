import pickle
import model_defination
import sys
import os
import shutil
from Bio import SeqIO
from compress_optimize import compress

import tempfile
import tensorflow as tf
import argparse

## TODO: compress identifier
def get_codec_args():
    parser = argparse.ArgumentParser(description='Input')
    parser.add_argument('-fp', '--file_path', type=str, required=True, help='file path')
    parser.add_argument('-od', '--output_dir', type=str, default='compress_output', help='output directory')
    parser.add_argument('-sp', '--split', default=False, action='store_true', help='split output if input is multiple fasta file')
    parser.add_argument('-m', '--model_name', type=str, default='deepdna_larger', help='model name')
    parser.add_argument('-pm', '--pretrained_model', type=str, required=True, help='model file')
    parser.add_argument('-bpm', '--backward_weights_file', type=str, default=None, help='backward model file; -bd should be enabled')
    parser.add_argument('-ppf', '--prob_dict_file', type=str, default='model_output/human_mito/human_mito.pkl', help='prior probability file')
    parser.add_argument("-pbn", "--predict_base_num", dest="predict_base_num", default=1, type=int, help="How many bases to predict. Default 1")


    ## data relavant: context_length, combine_num, gob_size
    parser.add_argument('-cl', '--context_length', type=int, default=64,
                    help='context length of input model')
    parser.add_argument('-c', '--chars', type=str, default='ACGT', help='base chars')
    parser.add_argument('-bs', '--batch_size', type=int, default=500,
                    help='parallel size')
    parser.add_argument('-ibs', '--infer_batch_size', type=int, default=5000,
                    help='GPU inference batch size')
    parser.add_argument('-ebs', '--escape_batch', type=int, default=50,
                    help='Batch size for short sequence')
    parser.add_argument('-cn', '--combined_num', type=int, default=1,
                    help='context length of input model')
    parser.add_argument('-bd', '--bi_direction', default=False, action='store_true', help='use forward model and backward model to apply bi-directional prediciton')
    parser.add_argument('-mm', '--markov_model', default=True, action='store_true', help='use markov model to get the probability of context')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_codec_args()
    file_path = args.file_path
    if not os.path.exists(file_path):
        print("%s not exists, abort." % file_path)
        exit()

    temp_dir = tempfile.mkdtemp()
    args.temp_dir = os.path.join(temp_dir, 'stream')
    os.makedirs(args.temp_dir)

    # parase fasta file
    pure_base_list=[]

    basename= os.path.basename(file_path).split('.')[0]
    if file_path.endswith('.fasta'):
        records = SeqIO.parse(file_path, 'fasta')

        if not args.split:
            text = ''
            for record in records:
                id = record.id
                # pure_base_path = os.path.join(args.temp_dir,id+'.txt')
                text+=str(record.seq)
            pure_base_path = os.path.join(temp_dir, '%s.txt'%basename)
            pure_base_list.append(pure_base_path)

            w_f = open(pure_base_path, 'w')
            w_f.write(text)
            w_f.close()
        else:
            for record in records:
                id = record.id
                pure_base_path = os.path.join(temp_dir,id+'.txt')
                w_f = open(pure_base_path, 'w')
                w_f.write(str(record.seq))
                w_f.close()
                pure_base_list.append(pure_base_path)
    elif file_path.endswith('.txt'):
        shutil.copyfile(file_path, os.path.join(temp_dir, os.path.basename(file_path)))
        pure_base_list.append(os.path.join(temp_dir, os.path.basename(file_path)))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # load compress model and pretrained-weights
    params = {'context_length':args.context_length, 'chars':args.chars, 'predict_base_num':args.predict_base_num, 'trainable':False}
    compress_model = getattr(model_defination, args.model_name)(params)
    compress_model.load_weights(args.pretrained_model)

    if args.bi_direction: ## bidirectional flag
        backward_model = getattr(model_defination, args.model_name)(params)
        backward_model.load_weights(args.backward_weights_file)

    lut = None
    for pure_base in pure_base_list:
        args.base_file = pure_base
        output_path = os.path.join(args.output_dir, os.path.basename(pure_base))
        if os.path.exists(output_path+'.lec'):
            print("%s exists,continue" % output_path)
            continue
        args.output_file = output_path
        args.params_file = None
        # create temp directory
        args.temp_file_prefix = os.path.join(args.temp_dir, "compressed")

        print("Compressing %s" % os.path.basename(pure_base))
        if args.bi_direction:
            compress(args, compress_model, backward_model=backward_model, lut=lut, stride=args.predict_base_num, bi_direction=True)
        else:
            compress(args, compress_model, lut=lut, stride=args.predict_base_num)

    shutil.rmtree(temp_dir)
