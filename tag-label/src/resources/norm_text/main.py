import os
import argparse
from src.processing import run

parser = argparse.ArgumentParser()

parser.add_argument("-t", "--text", help="Input text to normalize", default="A random string.", type=str)
args = parser.parse_args()

if __name__ == '__main__':
    text = run(args.text)
    print(text)