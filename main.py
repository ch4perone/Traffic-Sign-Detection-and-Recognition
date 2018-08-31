import argparse
import os

parser = argparse.ArgumentParser(description='Supervised Street Sign Assignment (s4a): Machine Learning method for '
                                             'learning and recognition of street signs')
parser.add_argument('--train', action='store_true',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()

if args.train:
    os.system('python3 trainModel.py')