import spacy

nlp = spacy.load('en_core_web_trf')

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

