import spacy

nlp = spacy.load('en_core_web_sm')

doc = nlp(text='Checking on the parts of speech and tokenization')
n = 10
for token in doc:
    print(token.text,'\t', token.lemma_,'\t', token.pos_,'\t',
          token.tag_,'\t', token.dep_,'\t',token.shape_,'\t',
          token.is_alpha,'\t', token.is_stop)


# TEXT Actual text or word being processed
# LeMMa root form of the word being processed
# POS part-of-speech of the wordtagthey express the part-of-speech (e.g., Verb) and some amount of morphological information (e.g., that the verb is past tense).
# Dep syntactic dependency (i.e., the relation between tokens)
# Shape shape of the word (e.g., the capitalization, punctuation, digits format)
# Alpha is the token an alpha character?
# Stop is the word a stop word or part of a stop list?


#### NAMED ENTITY RECOGNITION
test_string = u"Google has its headquarters in Mountain View, California having revenue amounted to 109.65 billion US dollars"
doc_ner = nlp(text=test_string)

my_string = u"Mark Zuckerberg born May 14, 1984 in New York is an American technology entrepreneur and philanthropist best known for co-founding and leading Facebook as its chairman and CEO."
doc = nlp(my_string)
for ent in doc.ents:
    print(ent.text, ent.label_)



print('#### NAMED ENTITY RECOGNITION')
for entities in doc_ner.ents:
    print(entities.text, entities.label_)


class using_spacy:
    def __init__(self, model, text):
        self.model = model
        self.text = text
        self.nlp = spacy.load(self.model)
        self.doc = self.nlp(self.text)


    def PartsofSpeechTagging(self):
        for token in self.doc:
            print(token.text, '\t', token.lemma_, '\t', token.pos_, '\t',
                  token.tag_, '\t', token.dep_, '\t', token.shape_, '\t',
                  token.is_alpha, '\t', token.is_stop)

    def NamedEntityRecognition(self):
        print('#### NAMED ENTITY RECOGNITION')
        for entities in doc_ner.ents:
            print(entities.text, entities.label_)
        pass

    pass
