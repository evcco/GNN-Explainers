import time
import argparse
from texttable import Texttable

def parse_args():
    t = time.strftime('%b %d %Y %H:%M:%S', time.localtime(time.time()))
    parser = argparse.ArgumentParser(description="Run Causal Screening and other explainers.")
    parser.add_argument('--dataset_num', nargs='?', default='[3]',
                        help='Choose dataset(s).')
    parser.add_argument('--idx_explainers', nargs='?', default='[9]',
                        help='Choose explainer(s).')
    parser.add_argument('--num_test', type=int, default=10,
                        help='Number of graph instances to test.')

    parser.add_argument('--large_scale', type=int, default=1,
                        help='Whether to use .')
    parser.add_argument('--log_name', type=str, default='',
                        help='Log file name. Default: TIME.')
    parser.add_argument('--log_path', type=str, default='log/',
                        help='Path to store log file.')

    parser.add_argument('--draw_graph', type=int, default=0,
                    help='Whether to visulize explaination result(s).')
    parser.add_argument('--vis_ratio', type=float, default=0.2,
                        help='Visualize top vis_ratio edges. Active when draw_graph == True')

    parser.add_argument('--n_cluster', type=int, default=5,
                        help='Cluster size.')
    parser.add_argument('--ice_ratio', type=float, default=0.0,
                        help="Visualize top ice_ratio edges' ICE distribution.")
    parser.add_argument('--random_seed', type=int, default=2021,
                        help='Random seed to run the code.')
    parser.add_argument('--topk', type=int, default=5,
                        help='Evaluate precison@topk for BA-3 dataset.')

    parser.add_argument('--eval_ACC', type=int, default=1,
                        help='Whether to evaluate Accuracy for the explaination results.')
    parser.add_argument('--ACC_ratio_lst', nargs='?',default='[0.1 * (i+1) for i in range(10)]',
                        help='.')
    parser.add_argument('--eval_CTS', type=int, default=1,
                        help='Whether to evaluate Constractivity for the explaination results.')
    parser.add_argument('--eval_SC', type=int, default=1,
                        help='Whether to do Saliency Check for explainer(s).')

    return parser.parse_args()


