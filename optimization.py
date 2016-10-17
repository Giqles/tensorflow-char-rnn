from bayes_opt import BayesianOptimization
import pickle as pkl
import train
import json
import os

def optimizer(name):

    resfile = 'output/' + name + '/res.pkl'

    # Some of this is taken from nadavbh12's fork of the original repo.
    def optim(**params):
        nonlocal trial
        trial += 1
        odir = 'output/' + name + '/_trial_' + str(int(trial))

        args = ['--data_file=data/corpus_clean.txt',
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

        train.main(args)
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

    bo = BayesianOptimization(optim, searchSpace)

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
            bo.maximize(init_points=5, n_iter=20, kappa=3.29, verbose=True)
            print(bo.Y)
        else:
            # Add in the initialization
            bo.initialize(res)
            # Do some optimization
            bo.maximize(init_points=0, n_iter=9, kappa=3.29, verbose=True)

    finally:
        # Manipulate the trial data into the right format to feed back to the
        # initialize method
        allres = {}
        for i in range(len(bo.Y)):
            allres[bo.Y[i]] = dict(zip(bo.keys, bo.X[i]))
        # Save wherever we got to:
        with open(resfile, 'wb') as f:
            pkl.dump(allres, f)

name = 'agenda-bot'

# Run the optimization
optimizer(name)
