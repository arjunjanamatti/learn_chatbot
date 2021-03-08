import spacy
from spacy import displacy

class using_spacy:
    def __init__(self, model, text):
        self.model = model
        self.text = text
        self.nlp = spacy.load(self.model)
        self.doc = self.nlp(self.text)


    def PartsofSpeechTagging(self):
        print('#### PARTS OF SPEECH')
        for token in self.doc:
            print(token.text, '\t', token.lemma_, '\t', token.pos_, '\t',
                  token.tag_, '\t', token.dep_, '\t', token.shape_, '\t',
                  token.is_alpha, '\t', token.is_stop)

    def NamedEntityRecognition(self):
        print('#### NAMED ENTITY RECOGNITION')
        for entities in self.doc.ents:
            print(entities.text, entities.label_)
        pass

    def DisplayDependencyParsing(self):
        displacy.serve(self.doc, style='dep')
        pass

    pass

model_1 = 'en_core_web_sm'
model_2 = 'en_core_web_trf'
text_1 = 'Checking on the parts of speech and tokenization'
text_2 = "Google has its headquarters in Mountain View, California having revenue amounted to 109.65 billion US dollars"
text_3 = "Mark Zuckerberg born May 14, 1984 in New York is an American technology entrepreneur and philanthropist best known for co-founding and leading Facebook as its chairman and CEO."

# instance_1 = using_spacy(model_1, text_1)
# instance_1.PartsofSpeechTagging()
#
# instance_2 = using_spacy(model_2, text_2)
# instance_2.NamedEntityRecognition()
#
# instance_3 = using_spacy(model_1, text_3)
# instance_3.NamedEntityRecognition()
#
instance_3 = using_spacy(model_2, text_3)
instance_3.NamedEntityRecognition()
instance_3.DisplayDependencyParsing()