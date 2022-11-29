import argparse
import os
from datetime import date, timedelta
from statistics import mean, median, stdev

import tensorflow as tf


def check_args(parsed_args):
    if parsed_args.batch_size < 1:
        parsed_args.batch_size = 1
    return parsed_args


def parse_args(args):
    today = str(date.today() + timedelta(days=0))
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--snapshot', help='Resume training from a snapshot.', default=None)
    parser.add_argument('--mode', help='Mode in which a program should be run.', default='Train')
    parser.add_argument('--model-name', help='Name of a model from a dictionary.', default='CustomNetFPN')
    parser.add_argument('--dataset-path', help='Path of dataset json files.', default=None)
    parser.add_argument('--train-json', help='Train set json file name.', default=None)
    parser.add_argument('--val-json', help='Validation set json file name.', default=None)
    parser.add_argument('--test-json', help='Test set json file name.', default=None)
    parser.add_argument('--freeze-backbone', help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--batch-size', help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=200)
    parser.add_argument('--steps', help='Number of steps per epoch.', type=int, default=10000)
    parser.add_argument('--snapshot-path',
                        help='Path to store snapshots of models during training',
                        default='checkpoints\\{}'.format(today))
    parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output',
                        default='logs\\{}'.format(today))
    parser.add_argument('--no-snapshots', help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation', help='Disable per epoch evaluation.', dest='evaluation',
                        action='store_false')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--input-size', help='Rescale the image so the smallest side is min_side.', type=int,
                        default=512)
    parser.add_argument('--compute-val-loss', help='Compute validation loss during training', dest='compute_val_loss',
                        action='store_true')

    # Fit generator arguments
    parser.add_argument('--multiprocessing', help='Use multiprocessing in fit_generator.', action='store_true')
    parser.add_argument('--workers', help='Number of generator workers.', type=int, default=1)
    parser.add_argument('--max-queue-size', help='Queue length for multiprocessing workers in fit_generator.', type=int,
                        default=10)
    print(vars(parser.parse_args(args)))
    return check_args(parser.parse_args(args))


def initialize_tf_keras_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def print_times(times):
    print("Inference speed:")
    print("#############################")
    print("Total predictions:" + str(len(times)))
    print("#############################")
    print("Maximum taken: " + str(max(times)))
    print("Minimum taken: " + str(min(times)))
    print("Average: " + str(mean(times)))
    print("Median: " + str(median(times)))
    print("Standart deviation: " + str(stdev(times)))

