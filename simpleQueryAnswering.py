import sys, os
import numpy
path = os.path.abspath(os.path.join("stanford_corenlp_python"))
sys.path.append(path)

from practnlptools.tools import Annotator
annotator=Annotator()

from stanford_corenlp_python.corenlp import *
from nltk.stem import PorterStemmer, WordNetLemmatizer
porter = PorterStemmer()

wordnet_lemmatizer = WordNetLemmatizer()

#from corenlp import StandfordCoreNLP
import nltk
import nltk.data
import collections
import yesno
import json
from bs4 import BeautifulSoup

# Setup
corenlp = StanfordCoreNLP(corenlp_path="./stanford_corenlp_python/stanford-corenlp-full-2014-08-27/")
sent_detector = nltk.data.load("tokenizers/punkt/english.pickle")

# Hardcoded word lists
yesnowords = ["can", "could", "would", "is", "does", "has", "was", "were", "had", "have", "did", "are", "will"]
commonwords = ["the", "a", "an", "is", "are", "were", "."]
questionwords = ["who", "what", "where", "when", "why", "how", "whose", "which", "whom"]
motion_verbs = ['move', 'travel', 'journeyed', 'went', 'goes']

motion_verbs = [porter.stem(w) for w in motion_verbs]

# Take in a tokenized question and return the question type and body
def processquestion(qwords, question):
    
    # Find "question word" (what, who, where, etc.)
    questionword = ""
    qidx = -1

    for (idx, word) in enumerate(qwords):
        if word.lower() in questionwords:
            questionword = word.lower()
            qidx = idx
            break
        elif word.lower() in yesnowords:
            return ("YESNO", qwords)

    if qidx < 0:
        return ("MISC", qwords)

    if qidx > len(qwords) - 3:
        target = qwords[:qidx]
    else:
        target = qwords[qidx+1:]
    type = "MISC"

    annotation = annotator.getAnnotations(question, dep_parse=True)
    pos = annotation['pos']
    # Determine question type
    if questionword in ["who", "whose", "whom"]:
        type = "S-PER"
    elif questionword == "where":
        type = "S-LOC"
        target = [tag[0] for tag in pos if tag[1] == "NN" or tag[1] == "NNP"]
    elif questionword == "when":
        type = "TIME"
    elif questionword == "how":
        if target[0] in ["few", "little", "much", "many"]:
            type = "QUANTITY"
            target = target[1:]
        elif target[0] in ["young", "old", "long"]:
            type = "TIME"
            target = target[1:]

    # Trim possible extra helper verb
    if questionword == "which":
        target = target[1:]
    if target[0] in yesnowords:
        target = target[1:]
    
    # Return question data
    return (type, questionword, target)

def getSubject(pos):
    ret = []
    for tag in pos:
        if tag[1] in ['NN', 'NNP']:
            ret.append(tag[0])
    return ret 

def similar(word, checklist):
    for check in checklist:
        if word in check:
            return 1
    return 0


def processFiles(articlefilename, questionsfilename=None):

    if questionsfilename == None:
        questionsfilename = articlefilename

    article, question, answer = [], [], []
    # Process article file
    with open(articlefilename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line.replace("\n", ".")
            if "?" in line:
                q, ans, no = line.split('\t')
                question.append(" ".join(q.split()[1:]))
                answer.append(ans)

            else:
                line = " ".join(line.split()[1:])
                article.append(line)

    return article, question, answer
    
    # article = BeautifulSoup(article).get_text()
    # article = ''.join([i if ord(i) < 128 else ' ' for i in article])
    # article = article.replace("\n", " . ")
    # article = sent_detector.tokenize(article)

    # print article
    # # Process questions file
    # questions = open(questionsfilename, 'r').read()
    # questions = questions.decode('utf-8')
    # questions = questions.splitlines()

def main():
    # Get command line arguments
    debug = 0
    articlefilename = sys.argv[1]
    questionsfilename = sys.argv[2]
    article, questions, answers = processFiles(articlefilename, questionsfilename)
    if debug:
        print article, questions, answers
    
    # Iterate through all questions
    for question in questions:

        # Tokenize question
        print "Q:", question
        qwords = nltk.word_tokenize(question.replace('?', ''))
        questionPOS = nltk.pos_tag(qwords)

        # Process question
        (type, questionword, target) = processquestion(qwords, question)

        # Answer yes/no questions
        if type == "YESNO":
            yesno.answeryesno(article, qwords)
            continue

        # Get sentence keywords
        searchwords = set(target).difference(commonwords)
        dict = collections.Counter()

        answer = []
        # Find most relevant sentences
        for (i, sent) in enumerate(article):
            sentwords = nltk.word_tokenize(sent)
            wordmatches = set(filter(set(searchwords).__contains__, sentwords))
            dict[sent] = len(wordmatches)
            if dict[sent] == 0:
                continue

            annotation = annotator.getAnnotations(sent, dep_parse=True)
            
            pos = annotation['pos']
            ner = annotation['ner']
            srl = annotation['srl']

            subject = getSubject(pos)
            
            if debug:
                print subject, target
                print ner, srl

            if set(target) & set(subject):
                for tag in ner:
                    if type == tag[1]:
                        answer.append(tag[0])

                srltag = [k for k, v in srl[0].iteritems()]
                v =  porter.stem(srl[0]["V"])
                for i, stag in enumerate(srltag):
                    if srl[0][stag] in subject and questionword == "where" and similar(v, motion_verbs):
                        answer.append(srl[0][srltag[i+1]])

        if answer:
            print answer[-1]


  
if __name__ == '__main__':
    main()


    # Focus on 10 most relevant sentences
    # for (sentence, matches) in dict.most_common(10):
    #     parse = json.loads(corenlp.parse(sentence))
    #     sentencePOS = nltk.pos_tag(nltk.word_tokenize(sentence))
    #     print sentencePOS, parse["sentences"][0]["words"]

    #     # Attempt to find matching substrings
    #     searchstring = ' '.join(target)
    #     if searchstring in sentence:
    #         startidx = sentence.index(target[0])
    #         endidx = sentence.index(target[-1])
    #         answer = sentence[:startidx]
    #         done = True
    
    #     # Check if solution is found
    #     if done:
    #         continue

    #     # Check by question type
    #     answer = ""
    #     for worddata in parse["sentences"][0]["words"]:
            
    #         # Mentioned in the question
    #         if worddata[0] in searchwords:
    #             continue
            
    #         if type == "PERSON":
    #             if worddata[1]["NamedEntityTag"] == "PERSON":
    #                 answer = answer + " " + worddata[0]
    #                 done = True
    #             elif done:
    #                 break

    #         if type == "PLACE":
    #             if worddata[1]["NamedEntityTag"] == "LOCATION":
    #                 answer = answer + " " + worddata[0]
    #                 done = True
    #             elif done:
    #                 break

    #         if type == "QUANTITY":
    #             if worddata[1]["NamedEntityTag"] == "NUMBER":
    #                 answer = answer + " " + worddata[0]
    #                 done = True
    #             elif done:
    #                 break

    #         if type == "TIME":
    #             if worddata[1]["NamedEntityTag"] == "NUMBER":
    #                 answer = answer + " " + worddata[0]
    #                 done = True
    #             elif done:
    #                 answer = answer + " " + worddata[0]
    #                 break
            
    # if done:
    #     print answer

    
