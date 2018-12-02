from collections import defaultdict
from itertools import chain
import regex
import numpy as np
import pandas as pd

pluck = lambda dict, *args: (dict[arg] for arg in args) #pylint: disable-msg=e1136

def read_data(path):
    file = pd.read_csv(path, header=None)
    for _, row in file.iterrows():
        yield row.values[0]

def parse_row(row):
    return regex.search(
        r'(?:\d+)\s(?:(?P<question>.*)\t(?P<answer>.*)\t(?P<support>.*))|(?:(?P<fact_num>\d+)\s(?P<fact_text>.*))', 
        row)

def get_task(row):
    fact1 = parse_row(next(row))['fact_text']
    fact2 = parse_row(next(row))['fact_text']
    question, answer = pluck(parse_row(next(row)).groupdict(), 'question', 'answer')
    return [fact1, fact2, question, answer]

def parse_data(path):
    data = read_data(path)
    tasks = []

    try:
        while True:
            tasks.append(get_task(data))
    except: StopIteration

    return pd.DataFrame(
        data=tasks,
        columns=['fact1', 'fact2', 'question', 'answer'])

if __name__ == "__main__":
    print parse_data('data/babI/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt')
