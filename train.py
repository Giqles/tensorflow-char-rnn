import argparse
import codecs
import json
import logging
import os
import shutil
import sys

import numpy as np
from char_rnn_model import *

TF_VERSION = int(tf.__version__.split('.')[1])

def main(args):
    parser = argparse.ArgumentParser()

    # Data and vocabulary file
    parser.add_argument('--data_file', type=str,
                        default='data/tiny_shakespeare.txt',
                        help='data file')

    parser.add_argument('--encoding', type=str,
                        default='utf-8',
                        help='the encoding of the data file.')

    # Parameters for saving models.
    parser.add_argument('--output_dir', type=str, default='output',
                        help=('directory to store final and'
                              ' intermediate results and models.'))

    # Parameters to configure the neural network.
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='size of RNN hidden state vector')
    parser.add_argument('--embedding_size', type=int, default=0,
                        help='size of character embeddings')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--num_unrollings', type=int, default=10,
                        help='number of unrolling steps.')
    parser.add_argument('--model', type=str, default='lstm',
                        help='which model to use (rnn, lstm or gru).')

    # Parameters to control the training.
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='minibatch size')
    parser.add_argument('--train_frac', type=float, default=0.9,
                        help='fraction of data used for training.')
    parser.add_argument('--valid_frac', type=float, default=0.05,
                        help='fraction of data used for validation.')
    parser.add_argument('--early_stopping', type=int, default=5,
                        help='number of epochs to tolerate without improvement')
    # test_frac is computed as (1 - train_frac - valid_frac).
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout rate, default to 0 (no dropout).')

    parser.add_argument('--input_dropout', type=float, default=0.0,
                        help=('dropout rate on input layer, default to 0 (no dropout),'
                              'and no dropout if using one-hot representation.'))

    # Parameters for gradient descent.
    parser.add_argument('--max_grad_norm', type=float, default=5.,
                        help='clip global grad norm')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
                        help='initial learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate')

    # Parameters for logging.
    parser.add_argument('--log_to_file', dest='log_to_file', action='store_true',
                        help=('whether the experiment log is stored in a file under'
                              '  output_dir or printed at stdout.'))
    parser.set_defaults(log_to_file=False)

    parser.add_argument('--progress_freq', type=int,
                        default=100,
                        help=('frequency for progress report in training'
                              ' and evalution.'))

    parser.add_argument('--verbose', type=int,
                        default=0,
                        help=('whether to show progress report in training'
                              ' and evalution.'))

    # Parameters to feed in the initial model and current best model.
    parser.add_argument('--init_model', type=str,
                        default='',
                        help=('initial model'))
    parser.add_argument('--best_model', type=str,
                        default='',
                        help=('current best model'))
    parser.add_argument('--best_valid_ppl', type=float,
                        default=np.Inf,
                        help=('current valid perplexity'))

    # Parameters for using saved best models.
    parser.add_argument('--init_dir', type=str, default='',
                        help='continue from the outputs in the given directory')

    # Parameters for debugging.
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='show debug information')
    parser.set_defaults(debug=False)

    # Parameters for unittesting the implementation.
    parser.add_argument('--test', dest='test', action='store_true',
                        help=('use the first 1000 character to as data'
                              ' to test the implementation'))
    parser.set_defaults(test=False)

    args = parser.parse_args(args)

    # Specifying location to store model, best model and tensorboard log.
    args.save_model = os.path.join(args.output_dir, 'save_model/model')
    args.save_best_model = os.path.join(args.output_dir, 'best_model/model')
    args.tb_log_dir = os.path.join(args.output_dir, 'tensorboard_log/')
    args.vocab_file = ''

    # Create necessary directories.
    if args.init_dir:
        args.output_dir = args.init_dir
    else:
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        for paths in [args.save_model, args.save_best_model,
                      args.tb_log_dir]:
            os.makedirs(os.path.dirname(paths))

    # Specify logging config.
    log_name = os.path.basename(args.output_dir) + '_experiment_log'
    log_fname = log_name + '.txt'
    if args.log_to_file:
        args.log_file = os.path.join(args.output_dir, log_fname)
    else:
        args.log_file = 'stdout'

    # Set logging file.
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    if args.log_file == 'stdout':
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.FileHandler(args.log_file)
    log_format = '%(asctime)s %(levelname)s:%(message)s'
    date_format = '%I:%M:%S'
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    print(('=' * 60))
    print(('All final and intermediate outputs will be stored in %s/' % args.output_dir))
    print(('All information will be logged to %s' % args.log_file))
    print(('=' * 60 + '\n'))

    if args.debug:
        logging.info('args are:\n%s', args)

    # Prepare parameters.
    if args.init_dir:
        with open(os.path.join(args.init_dir, 'result.json'), 'r') as f:
            result = json.load(f)
        params = result['params']
        args.init_model = result['latest_model']
        best_model = result['best_model']
        best_valid_ppl = result['best_valid_ppl']
        if 'encoding' in result:
            args.encoding = result['encoding']
        else:
            args.encoding = 'utf-8'
        args.vocab_file = os.path.join(args.init_dir, 'vocab.json')
    else:
        params = {'batch_size': args.batch_size,
                  'num_unrollings': args.num_unrollings,
                  'hidden_size': args.hidden_size,
                  'max_grad_norm': args.max_grad_norm,
                  'embedding_size': args.embedding_size,
                  'num_layers': args.num_layers,
                  'learning_rate': args.learning_rate,
                  'model': args.model,
                  'dropout': args.dropout,
                  'input_dropout': args.input_dropout}
        best_model = ''
    logger.info('Parameters are:\n%s\n', json.dumps(params, sort_keys=True, indent=4))

    # Read and split data.
    logger.info('Reading data from: %s', args.data_file)
    with codecs.open(args.data_file, 'r', encoding=args.encoding) as f:
        text = f.read()

    if args.test:
        text = text[:1000]
    logger.info('Number of characters: %s', len(text))

    if args.debug:
        n = 10
        logger.info('First %d characters: %s', n, text[:n])

    logger.info('Creating train, valid, test split')
    train_size = int(args.train_frac * len(text))
    valid_size = int(args.valid_frac * len(text))
    test_size = len(text) - train_size - valid_size
    train_text = text[:train_size]
    valid_text = text[train_size:train_size + valid_size]
    test_text = text[train_size + valid_size:]

    if args.vocab_file:
        vocab_index_dict, index_vocab_dict, vocab_size = load_vocab(args.vocab_file, args.encoding)
    else:
        logger.info('Creating vocabulary')
        vocab_index_dict, index_vocab_dict, vocab_size = create_vocab(text)
        vocab_file = os.path.join(args.output_dir, 'vocab.json')
        save_vocab(vocab_index_dict, vocab_file, args.encoding)
        logger.info('Vocabulary is saved in %s', vocab_file)
        args.vocab_file = vocab_file

    params['vocab_size'] = vocab_size
    logger.info('Vocab size: %d', vocab_size)

    # Create batch generators.
    batch_size = params['batch_size']
    num_unrollings = params['num_unrollings']
    train_batches = BatchGenerator(train_text, batch_size, num_unrollings, vocab_size,
                                   vocab_index_dict, index_vocab_dict)
    # valid_batches = BatchGenerator(valid_text, 1, 1, vocab_size,
    #                                vocab_index_dict, index_vocab_dict)
    valid_batches = BatchGenerator(valid_text, batch_size, num_unrollings, vocab_size,
                                   vocab_index_dict, index_vocab_dict)

    test_batches = BatchGenerator(test_text, 1, 1, vocab_size,
                                  vocab_index_dict, index_vocab_dict)

    if args.debug:
        logger.info('Test batch generators')
        logger.info(batches2string(next(train_batches), index_vocab_dict))
        logger.info(batches2string(next(valid_batches), index_vocab_dict))
        logger.info('Show vocabulary')
        logger.info(vocab_index_dict)
        logger.info(index_vocab_dict)

    # Create graphs
    logger.info('Creating graph')
    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('training'):
            train_model = CharRNN(is_training=True, use_batch=True, **params)
        tf.get_variable_scope().reuse_variables()
        with tf.name_scope('validation'):
            valid_model = CharRNN(is_training=False, use_batch=True, **params)
        with tf.name_scope('evaluation'):
            test_model = CharRNN(is_training=False, use_batch=False, **params)
            saver = tf.train.Saver(name='checkpoint_saver')
            best_model_saver = tf.train.Saver(name='best_model_saver')

    logger.info('Model size (number of parameters): %s\n', train_model.model_size)
    logger.info('Start training\n')

    result = {}
    result['params'] = params
    result['vocab_file'] = args.vocab_file
    result['encoding'] = args.encoding

    try:
        # Use try and finally to make sure that intermediate
        # results are saved correctly so that training can
        # be continued later after interruption.
        with tf.Session(graph=graph) as session:
            # Version 8 changed the api of summary writer to use
            # graph instead of graph_def.
            if TF_VERSION >= 8:
                graph_info = session.graph
            else:
                graph_info = session.graph_def

            train_writer = tf.train.SummaryWriter(args.tb_log_dir + 'train/', graph_info)
            valid_writer = tf.train.SummaryWriter(args.tb_log_dir + 'valid/', graph_info)

            # load a saved model or start from random initialization.
            if args.init_model:
                saver.restore(session, args.init_model)
            else:
                tf.initialize_all_variables().run()

            # Set epochs_without_improvement counter
            epochs_without_improvement = 0

            for i in range(args.num_epochs):
                logger.info('=' * 19 + ' Epoch %d ' + '=' * 19 + '\n', i)
                logger.info('Training on training set')
                # training step
                ppl, train_summary_str, global_step = train_model.run_epoch(
                    session,
                    train_size,
                    train_batches,
                    is_training=True,
                    verbose=args.verbose,
                    freq=args.progress_freq)
                # record the summary
                train_writer.add_summary(train_summary_str, global_step)
                train_writer.flush()
                # save model
                saved_path = saver.save(session, args.save_model,
                                                    global_step=train_model.global_step)
                logger.info('Latest model saved in %s\n', saved_path)
                logger.info('Evaluate on validation set')

                # valid_ppl, valid_summary_str, _ = valid_model.run_epoch(
                valid_ppl, valid_summary_str, _ = valid_model.run_epoch(
                    session,
                    valid_size,
                    valid_batches,
                    is_training=False,
                    verbose=args.verbose,
                    freq=args.progress_freq)

                # save and update best model
                if (not best_model) or (valid_ppl < best_valid_ppl):
                    best_model = best_model_saver.save(
                        session,
                        args.save_best_model,
                        global_step=train_model.global_step)
                    best_valid_ppl = valid_ppl
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                valid_writer.add_summary(valid_summary_str, global_step)
                valid_writer.flush()
                logger.info('Best model is saved in %s', best_model)
                logger.info('Best validation ppl is %f\n', best_valid_ppl)
                result['latest_model'] = saved_path
                result['best_model'] = best_model
                # Convert to float because numpy.float is not json serializable.
                result['best_valid_ppl'] = float(best_valid_ppl)
                result_path = os.path.join(args.output_dir, 'result.json')
                if os.path.exists(result_path):
                    os.remove(result_path)
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2, sort_keys=True)

                # Stop training if early_stopping exceeded
                if epochs_without_improvement >= args.early_stopping:
                    logger.info('Early stopping tolerance exceeded.')
                    logger.info('Stopping training after epoch %d', i)
                    logger.info('\n')
                    break

            logger.info('Latest model is saved in %s', saved_path)
            logger.info('Best model is saved in %s', best_model)
            logger.info('Best validation ppl is %f\n', best_valid_ppl)
            logger.info('Evaluate the best model on test set')
            saver.restore(session, best_model)
            test_ppl, _, _ = test_model.run_epoch(session, test_size, test_batches,
                                                  is_training=False,
                                                  verbose=args.verbose,
                                                  freq=args.progress_freq)
            result['test_ppl'] = float(test_ppl)
    finally:
        result_path = os.path.join(args.output_dir, 'result.json')
        if os.path.exists(result_path):
            os.remove(result_path)
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, sort_keys=True)


def create_vocab(text):
    unique_chars = list(set(text))
    vocab_size = len(unique_chars)
    vocab_index_dict = {}
    index_vocab_dict = {}
    for i, char in enumerate(unique_chars):
        vocab_index_dict[char] = i
        index_vocab_dict[i] = char
    return vocab_index_dict, index_vocab_dict, vocab_size


def load_vocab(vocab_file, encoding):
    with codecs.open(vocab_file, 'r', encoding=encoding) as f:
        vocab_index_dict = json.load(f)
    index_vocab_dict = {}
    vocab_size = 0
    for char, index in list(vocab_index_dict.items()):
        index_vocab_dict[index] = char
        vocab_size += 1
    return vocab_index_dict, index_vocab_dict, vocab_size


def save_vocab(vocab_index_dict, vocab_file, encoding):
    with codecs.open(vocab_file, 'w', encoding=encoding) as f:
        json.dump(vocab_index_dict, f, indent=2, sort_keys=True)

if __name__ == '__main__':
    main(sys.argv[1:])
