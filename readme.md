setup & install venv:
- `virtualvenv venv`
- `source venv/bin/activate`

install custom ipykernel for jupyter notebook:
- `ipython kernel install --user --name=ml-reference-implementations`
- now the venv kernel can be selected in jupyter notebook

testing:
- `pip install -r requirements.txt`
- `python -m unittest discover`

Question / Answer Datasets:
- squad2.0 from stanford https://rajpurkar.github.io/SQuAD-explorer/
- babi from facebook https://research.fb.com/downloads/babi/
- Allen AI Science 
- Quiz Bowl
- CNN / Daily Mail 
- MCTest
- TriviaQA
- SimpleQuestions
- WikiQA

Dialogue Datasets:
- OpenSubtitles dataset 18 
- Movie Dialog dataset 19
- Cornell Movie DialoguesCorpus

Semantic Role Labeling Datasets:
- Proposition Bank

Pronoun Resolution Datasets:
- OntoNotes 5.0
- TRAINS93

source: https://arxiv.org/pdf/1805.09461.pdf

how to represent words:
- 1 hot vectors
- word embedding
- ..?

how to encode text:
- remember the previous and the next words in the sequence
- lstm? gru?

attention mechanism:
- which words to focus on
- memory networks