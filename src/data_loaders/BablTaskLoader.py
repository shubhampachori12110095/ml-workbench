from collections import defaultdict
from itertools import chain
import regex
import numpy as np
import pandas as pd

pluck = lambda dict, *args: (dict[arg] for arg in args) #pylint: disable-msg=e1136

def _read_data(path):
    file = pd.read_csv(path, header=None)
    for _, row in file.iterrows():
        yield row.values[0]

def _parse_row(row):
    return regex.search(
        r'(?:\d+)\s(?:(?P<question>.*)\t(?P<answer>.*)\t(?P<support>.*))|(?:(?P<fact_num>\d+)\s(?P<fact_text>.*))', 
        row)

def _get_task(row):
    fact1 = _parse_row(next(row))['fact_text']
    fact2 = _parse_row(next(row))['fact_text']
    question, answer = pluck(_parse_row(next(row)).groupdict(), 'question', 'answer')
    return [fact1, fact2, question, answer]

def parse_data(path):
    data = _read_data(path)
    tasks = []

    try:
        while True:
            tasks.append(_get_task(data))
    except: StopIteration

    return pd.DataFrame(
        data=tasks,
        columns=['fact1', 'fact2', 'question', 'answer'])