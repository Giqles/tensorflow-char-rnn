from bayes_opt import BayesianOptimization
import pickle as pkl
import train
import os

# Some of this is taken from nadavbh12's fork of the original repo.
# The subfolders system needs work to function properly..
trial = 0
def optim(params):
    name = 'agenda-bot'
    trial += 1
    odir = 'outputs/' + name + '/_trial_' + str(int(trial))

    args = ['--data_file=data/corpus_clean.txt',
            '--train_frac=0.9',
            '--num_epochs=3',
            '--save_best_only',
            '--num_layers='     + str(int(params['num_layers'])),
            '--hidden_size='    + str(int(params['hidden_size'])),
            '--num_unrollings=' + str(int(params['num_unrollings'])),
            '--dropout='        + str(float(params['dropout'])),
            '--max_grad_norm='  + str(float(params['max_grad_norm'])),
            '--learning_rate='  + str(float(params['learning_rate'])),
            '--output_dir='     + odir]

    clf = train.main(args)
    with open(os.path.join(odir, 'result.json'), 'r') as f:
        result = json.load(f)

    # Optimization is maximization problem:
    return result['best_valid_ppl'] * -1

# Load up previous optimization data if available
if os.path.isfile('outputs/agenda-bot/res.pkl'):
    bo = pkl.load('outputs/agenda-bot/res.pkl')
    new = False
# Else set up from scratch
else:
    searchSpace = { 'num_layers': (2, 3),
                    'hidden_size': (128, 512),
                    'num_unrollings': (20, 80),
                    'dropout:' (0.0, 0.3),
                    'max_grad_norm': (1, 10),
                    'learning_rate': (0.0001, 0.01)}
    bo = BayesianOptimization(optim, searchSpace)
    new = True

# And start parameter search
if new:
    # Do some sane exploration
    exploreSpace = {'num_layers': [2, 2, 3, 3],
                    'hidden_size': [128, 256, 256, 512],
                    'num_unrollings': [20, 30, 40, 50],
                    'dropout': [0.1, 0.1, 0.2, 0.2],
                    'max_grad_norm': [5, 5, 5, 5],
                    'learning_rate': [0.001, 0.001, 0.001, 0.001]}
    bo.explore(exploreSpace)
    # And start maximization
    bo.maximize(init_points=5, n_iter=15, kappa=3.29, verbose=True)

else:
    # Carry on from where we left off
    bo.maximize(init_points=0, n_iter=15, kappa=3.29, verbose=True)

# Save wherever we got to:
pkl.dump(bo, 'outputs/agenda-bot/res.pkl')
