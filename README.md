## This project is taken from originally from 
##https://github.com/raoariel/NLP-Question-Answer-System
## This will include changes on top of this repository for project purpose improving and enhancing the methodology used in QA system developed.

# NLP-Question-Answer-System
----------------------------------
We will use NLTK to apply part of speech labeling and the Stanford parser for entity relationship modeling. Using an external package to take care of the details of implementation will allow us to focus on tweaking our algorithms on a higher level. We will use these tools to parse text before both distinguishing between the tasks of asking and answering.
Because pronouns are fundamentally ambiguous, we will consider using a probabilistic model for anaphora resolution. Then, we can build an offline database of entity-factoid pairs that we can query for answering.

Asking
----------------------------------
Since we have already tagged the text when parsing, we can then identify the candidate subjects of each question in an article and build a collection of question templates. We will then extract meta-information (such as the “Categories” section) from the Wikipedia HTML structure to topically generate questions for sections of text. Given a new article, we will consider the attributes of each subject in the text and apply the most probable question template based on all critical words in each sentence.

Answering
----------------------------------
Figuring out what type of question (yes/no, location, date, etc) is being asked will be useful for determining which relationships between words we should be considering. 
Parsing sentences into phrases and then deciding the functionality of the phrase will be useful for answering questions based on types. For example, a prepositional phrase that describes location will be useful for answering a location question while maintaining proper grammar.
If information or relationships in the article are successfully extracted, then this information will be delivered as the answer. Otherwise we will retrieve the most likely sentence, treating the keywords of the question as a vector that we are trying to match, and extract the most salient section of this retrieved sentence as the answer. We will likely use term frequency to rank sentences within a document.
Both the asking and answering modules will share the structured data extracted from parsing text and use wrappers over useful NLP algorithms from NLTK/other libraries. Otherwise the two components’ system designs will be independent.

Evaluation 
----------------------------------
We will be automatically evaluating each question generated for grammar and syntax, to ensure fluency. We will also evaluate answers for surface-level factual accuracy and adherence to the information need of the corresponding question, in addition to grammar and syntax. We will use the quality of candidate answers as a ranking criterion for the questions we generate.
