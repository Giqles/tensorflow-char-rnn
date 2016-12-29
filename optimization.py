from bayes_opt import BayesianOptimization
import argparse
import pickle as pkl
import train
import json
import logging
import os
import sys

def main(args):
    parser = argparse.ArgumentParser()

    # Name
    parser.add_argument('--name', type=str)
    # Corpus
    parser.add_argument('--corpus', type=str)

    # Parse arguments
    args = parser.parse_args(args)

    resfile = 'output/' + args.name + '/res.pkl'

    # Some of this is taken from nadavbh12's fork of the original repo.
    def evaluate(**params):
        # Get the trial iteration
        nonlocal trial
        trial += 1
        # Get the start_text and text length
        nonlocal start_text
        nonlocal text_length

        # Set the output directory, also where sample reads from
        odir = 'output/' + args.name + '/_trial_' + str(int(trial))

        # Pass the arguments for the training routine
        trargs = ['--data_file='    + args.corpus,
                  '--train_frac=0.9',
                  '--num_epochs=50',
                  '--early_stopping=5',
                  '--log_to_file',
                  '--num_layers='     + str(int(params['num_layers'])),
                  '--hidden_size='    + str(int(params['hidden_size'])),
                  '--num_unrollings=' + str(int(params['num_unrollings'])),
                  '--dropout='        + str(float(params['dropout'])),
                  '--max_grad_norm='  + str(float(params['max_grad_norm'])),
                  '--learning_rate='  + str(float(params['learning_rate'])),
                  '--output_dir='     + odir]

        # And do some training
        train.main(trargs)

        # Grab the logger from the training routine.
        log_name = '_trial_' + str(int(trial)) + '_experiment_log'
        logger = logging.getLogger(log_name)
        logger.setLevel(logging.INFO)

        # Bit of heading
        logger.info('===' * 20)
        logger.info('Sample texts:')
        logger.info('')

        # After training, make and store some sample text from the best model
        for temp in [0.3, 0.4, 0.7, 1.0]:
            smargs = ['--init_dir=' + odir,
                      '--start_text=' + start_text,
                      '--temperature=' + temp,
                      '--length=' + str(int(text_length))]
            sample_text = sample.main(smargs)
            logger.info('---' * 20)
            logger.info('Temperature: ' + str(temp))
            logger.info('---' * 20)
            logger.info(sample_text)

        # Load up the final optimization result to pass back to the optimizer
        with open(os.path.join(odir, 'result.json'), 'r') as f:
            result = json.load(f)

        # Optimization is maximization problem:
        return result['best_valid_ppl'] * -1

    # Define optimization problem
    searchSpace = {'num_layers': (2, 3),
                   'hidden_size': (128, 512),
                   'num_unrollings': (20, 80),
                   'dropout': (0.0, 0.3),
                   'max_grad_norm': (1, 10),
                   'learning_rate': (0.0001, 0.01)}

    bo = BayesianOptimization(evaluate, searchSpace)

    # Specify other arguments to pass to the optimization function
    start_text='The meaning of life is'
    text_length=1000

    # Load up previous optimization data if available
    if os.path.isfile(resfile):
        with open(resfile, 'rb') as f:
            res = pkl.load(f)
        trial = len(res)
        new = False
    # Else set up from scratch
    else:
        trial = 0
        new = True

    # And start parameter search
    try:
        # If new, do some exploration and some init points
        if new:
            # Do some sane exploration
            exploreSpace = {'num_layers': [2, 2, 3, 3],
                            'hidden_size': [128, 256, 256, 512],
                            'num_unrollings': [20, 30, 40, 50],
                            'dropout': [0.1, 0.1, 0.2, 0.2],
                            'max_grad_norm': [5, 5, 5, 5],
                            'learning_rate': [0.001, 0.001, 0.001, 0.001]}
            bo.explore(exploreSpace)

            # Do some optimization, with init points
            bo.maximize(init_points=5, n_iter=20, kappa=3.29)
        else:
            # Initialize with the earlier data
            bo.initialize(res)

            # Add some init points if we don't have enough
            if trial < 5:
                inits = 5 - trial
            else:
                inits = 0

            # Do some optimization
            bo.maximize(init_points=inits, n_iter=20, kappa=3.29)

    finally:
        # Manipulate the trial data into the right format to feed back to the
        # initialize method
        allres = {}
        for i in range(len(bo.Y)):
            allres[bo.Y[i]] = dict(zip(bo.keys, bo.X[i]))
        # Save wherever we got to:
        with open(resfile, 'wb') as f:
            pkl.dump(allres, f)

if __name__ == '__main__':
    main(sys.argv[1:])
