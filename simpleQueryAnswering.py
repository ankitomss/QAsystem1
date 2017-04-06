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
stopwords = set(["the", "a", "an", "is", "are", "were", ".", "there", "to"])
questionwords = ["who", "what", "where", "when", "why", "how", "whose", "which", "whom"]
motion_verbs = ['moved', 'travelled', 'journeyed', 'went', 'goes']
possess_verbs = ['has', 'got', 'find', 'found', 'get',  'grabbed', 'took']
left_verbs = ['put down', 'dropped', 'discarded', 'left']

motion_verbs = [porter.stem(w) for w in motion_verbs]
possess_verbs = [porter.stem(w) for w in possess_verbs]
left_verbs = [porter.stem(w) for w in left_verbs]
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
    ner = annotation['ner']

    target, attributes = [], []
    # Determine question type
    if questionword in ["who", "whose", "whom"]:
        type = "S-PER"
    elif questionword == "where":
        type = "S-LOC"
        targetDone = False
        for tag in pos:
            if (tag[1] == "NN" or tag[1] == "NNP") and not targetDone:
                target = [tag[0]]
                targetDone = True

            elif tag[1] == "NN" or tag[1] == "NNP" or tag[1] == "IN":
                attributes.append(tag)
        
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
    
    targetType = []
    for t in target:
        for ntag in ner:
            if t == ntag[0]:
                targetType.append(ntag[1])

    # Return question data
    return (type, questionword, target, targetType, attributes)

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

def markObjectLocation(keeper):
    if keeper not in location: return 

    for obj in list(has[keeper]):
        if obj not in location or location[keeper][-1] != location[obj][-1]:
            location[obj].append(location[keeper][-1])

def processLine(line, article, question, answer):
    
    qflag = 0
    line.replace("\n", ".")
    if "?" in line:
        qflag = 1
        q, ans, no = line.split('\t')
        question.append(" ".join(q.split()[1:]))
        answer.append(ans)

    else:
        line = line.split()[1:]
        line = [w for w in line if w not in stopwords]
        line = " ".join(line)
        
        article.append(line)

        annotation = annotator.getAnnotations(line, dep_parse=True)
        pos = annotation['pos']
        ner = annotation['ner']
        srl = annotation['srl']

        srltag = [k for k, v in srl[0].iteritems()]
        v = porter.stem(srl[0]["V"])

        if similar(v, motion_verbs):
            objects = [p[0] for p in pos if p[1] == 'NN' or p[1] == 'NNP']
            location[objects[0]].append(objects[1])
            markObjectLocation(objects[0])
        elif similar(v, possess_verbs):
            objects = [p[0] for p in pos if p[1] == 'NN' or p[1] == 'NNP']
            has[objects[0]].add(objects[1])
            invert_has[objects[1]] = objects[0]
            markObjectLocation(objects[0])
        elif similar(v, left_verbs):
            objects = [p[0] for p in pos if p[1] == 'NN' or p[1] == 'NNP']
            has[objects[0]].discard(objects[1])
            invert_has[objects[1]] = ""

    return qflag
    

debug = 0
location = collections.defaultdict(list)
act = collections.defaultdict(list)
has = collections.defaultdict(set)
invert_has = collections.defaultdict(list)

def main():
    # Get command line arguments
    articlefilename = sys.argv[1]
    if len(sys.argv) == 3:
        questionsfilename = sys.argv[2]
    article, questions, answers = [], [], []
    right, wrong = 0, 0

    if debug:
        print article, questions, answers
    
    with open(articlefilename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        qflag = processLine(line, article, questions, answers)
        if not qflag:
            continue
        question = questions[-1]
        answer = []

        # Tokenize question
        print "Q:", question
        qwords = nltk.word_tokenize(question.replace('?', ''))
        questionPOS = nltk.pos_tag(qwords)

        # Process question
        (type, questionword, target, targetType, attributes) = processquestion(qwords, question)
        # Answer yes/no questions
        if type == "YESNO":
            yesno.answeryesno(article, qwords)
            continue

        attributes_words = [a[0] for a in attributes]
        if questionword == "where" and "before" in attributes_words:
            before = [a[0] for a in attributes if a[1] == "NN"][0]
            loc = location[target[-1]]
            for i in xrange(len(loc)-1, -1, -1):
                if loc[i] == before and i != 0:
                    answer = [loc[i-1]]
                    break
        elif questionword == "where":
            if target[-1] in location:
                answer = [location[target[-1]][-1]]


        if answer:
            exp, actual = answer[-1], answers[-1]

            if actual in exp:
                right += 1
            else:
                wrong += 1

            print answer[-1], "-->", answers[-1], "Accuracy ", (float(right)*100) /(float(right) + float(wrong)) 

    print "Accuracy of the task ", (float(right)*100) /(float(right) + float(wrong))


def findEntity(article, target, targetType, questionword, stop=None):
    # Find most relevant sentences
    answer, newtarget, newtargetType = [], [], targetType
    targetChange = False
    dict = collections.Counter()
    for (i, sent) in enumerate(article):
        sentwords = nltk.word_tokenize(sent)
        wordmatches = set(filter(set(target).__contains__, sentwords))
        
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
                if srl[0][stag] in subject and questionword == "where" :

                    if 'PER' in "".join(targetType) and similar(v, motion_verbs):
                        answer.append(srl[0][srltag[i+1]])

                    elif similar(v, possess_verbs) and 'PER' not in "".join(targetType):
                        newtarget += [n[0] for n in ner if n[1] == 'S-PER']
                        newtargetType = ['PER']
                        targetChange = True


    if targetChange:
        answer = findEntity(article, newtarget[-1:], newtargetType, questionword, None)

    return answer

if __name__ == '__main__':
    main()
    
