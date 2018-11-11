from collections import defaultdict
import regex
import numpy as np
import pandas as pd

def parse_data(path='data/babI/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt'):
    '''
    read the babI task data into arrays. Sample data:

    facts[10]                 {
                                '1': 'Mary went to the bedroom.', 
                                '2': 'John journeyed to the bathroom.'
                            } 
    questions[10]             Where is John?  
    answers[10]               bathroom 
    supporting_facts[10]      2
    '''

    file = pd.read_csv(path, header=None)

    task_count  = 0
    facts = defaultdict(dict)
    questions = defaultdict(str)
    answers = defaultdict(str)
    supporting_facts = defaultdict(str)

    for key, row in file.iterrows():
        line = row.values[0]
        match = regex.search(r'(?:\d+)\s(?:(?P<question>.*)\t(?P<answer>.*)\t(?P<support>.*))|(?:(?P<fact_num>\d+)\s(?P<fact_text>.*))', line)
        
        if match['fact_text'] is not None:
            num, text = match['fact_num'], match['fact_text']
            facts[task_count][num] = text

        else:
            questions[task_count] = match['question']
            answers[task_count] = match['answer']
            supporting_facts[task_count] = match['support']
            task_count += 1

    return facts, questions, answers, supporting_facts
