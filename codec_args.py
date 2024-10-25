import argparse

def get_argument(): ## default parameter

    parser = argparse.ArgumentParser(description="Parameter.")
    parser.add_argument("-fp", "--fasta_path", dest="fasta_path", required=True, type=str, 
        help="single or multi-fasta file path for training.")
    parser.add_argument("-op", "--output_path", dest="output_path", default=None, type=str, 
        help="output path when do compression")

    parser.add_argument("-pm", "--pretrained_model", dest="pretrained_model", default=None, type=str, 
        help="pretrained model for initialization. If None, train from zero.")    
    parser.add_argument("-bpm", "--backward_model", dest="backward_model", default=None, type=str, 
        help="backward model for initialization. If None, train from zero.")    

    parser.add_argument("-m", "--model_name", dest="model_name", default="deepdna_multiout", type=str, 
        help="the name of model used for training. check in model_defination")

    parser.add_argument("-lr", "--learning_rate", dest="learning_rate", default=0.0001, type=float,
        help="initial learning rate use for training.")

    parser.add_argument("-bs", "--batch_size", dest="batch_size", default=10, type=int,
        help="batch size of data used for training.")
    parser.add_argument("-ibs", "--infer_batch_size", dest="infer_batch_size", default=5000, type=int, help="batch size used in gpu inference phrase")

    parser.add_argument("-pbn", "--predict_base_num", dest="predict_base_num", default=1, type=int, 
        help="How many bases to predict. Default 1")
    parser.add_argument('-sqy', '--sequence_y', default=True, action='store_true',
                    help='predict sequence y. the data is arranged as (None, )')
    
    parser.add_argument('-t', '--trainable', default=False, action='store_true',
                    help='indicate if feature is trainable')

    parser.add_argument('-ppf', '--prob_dict_file', type=str, default='model_output/human_mito/human_mito.pkl', help='prior probability file')
    parser.add_argument('-cn', '--combined_num', type=int, default=1,
                    help='context length of input model')
    parser.add_argument('-mm', '--markov_model', default=True, action='store_true',
                    help='use markov model to get the probability of context')

    # default following
    parser.add_argument("-e", "--epoch", default=500, dest="epoch", type=int, 
        help="training epoch.")
    parser.add_argument("-cl", "--context_length", dest="context_length", default=64, type=int, 
        help="total base number in context.")
    parser.add_argument("-l", "--loss_name", dest="loss_name", default="cross_entropy_loss", type=str,
        help="name of loss function.")
    parser.add_argument("-mkf", "--minimize_kmer_freq", dest="minimize_kmer_freq", default=2, type=int,
        help="minimal kmer frequency.")
    parser.add_argument("-bd", "--bi_direction", action='store_true', default=False, dest="bi_direction", 
        help="The dataset is reversed if add this.")
    parser.add_argument("-c", "--chars", dest="chars", default="ACGT", type=str,
        help="different bases")

    args = parser.parse_args()
    # print(args);exit()

    return args
