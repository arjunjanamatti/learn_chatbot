import spacy

nlp = spacy.load('en_core_web_trf')

doc = nlp(text='Check on the parts of speech and tokenization')

# for token in doc:
#     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
#             token.shape_, token.is_alpha, token.is_stop)

for token in doc:
    print(token.text, token.lemma_, token.pos_)