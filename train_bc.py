import argparse
import pickle

import d3rlpy

with open('random_dataset_flattened.pkl', 'rb') as readFile:
    # Serialize and save the data to the file
    random_dataset = pickle.load(readFile)

dataset = random_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--gpu", type=int)
args = parser.parse_args()

d3rlpy.seed(args.seed)

bc = d3rlpy.algos.DiscreteBCConfig(
).create(device=args.gpu)

bc.fit(
    dataset,
    n_steps=5000,
    n_steps_per_epoch=250,
)

bc.save('bc_random.d3')
