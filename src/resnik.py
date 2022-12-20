# os
import os

# third party
import stanza, nltk
from nltk.corpus import wordnet as wn

# my code
from story.Story import parse_file

STANZA_2_NLTK = { "PROPN": "NOUN", "VERB": "VERB", }

def main():
    # stories = parse_file("stories/movie/reducedMovie.story")
    
    # print(len(stories))
    # for i in range(len(stories)):
    #     print(stories[i])

    story1 = "John and Sally waited in line for ticket.\nJohn purchased two movie ticket.\nJohn and Sally walked up to the concession stand.\nJohn bought Sally a large popcorn.\nJohn and Sally entered the dark theater.\nSally chose seat at the front of the auditorium.\nJohn turned off the cellphone.\nSally ate the popcorn.\nJohn was bored during come attraction.\nJohn and Sally laughed during the movie.\nSally was surprised by the movie ending.\nSally and John left when the movie finish."

    story2 = "John and Sally got in line for the movie ticket.\nJohn and Sally bought the ticket.\nJohn and Sally got in line for popcorn.\nJohn and Sally bought popcorn.\nJohn and Sally walked down the theater corridor.\nJohn and Sally looked for the movie.\nJohn and Sally found the movie.\nJohn and Sally entered the auditorium.\nJohn and Sally found their seat.\nJohn and Sally sat down.\nJohn and Sally watched the movie.\nJohn and Sally left the movie theater."

    s1s1 = "John and Sally waited in line for ticket."
    s2s1 = "John and Sally got in line for the movie ticket."
    
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')
    nlp = stanza.Pipeline(lang='en',processors='tokenize,mwt,pos,lemma,depparse,ner')

    doc1 = nlp(s1s1)
    print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}\tupos: {word.upos}\txpos: {word.xpos}' for sent in doc1.sentences for word in sent.words], sep='\n')
    print("===")
    print(*[f'entity: {ent.text}\ttype: {ent.type}' for sent in doc1.sentences for ent in sent.ents], sep='\n')

    print("Finished parsing story 1.")


    doc2 = nlp(s2s1)
    # print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc2.sentences for word in sent.words], sep='\n')

    print("Finished parsing story 2.")

    # print(wn.synsets("Sally")[2].definition())
    
    similarities = []
    simcount = 0
    for sent1 in doc1.sentences:
        for word1 in sent1.words:
            
            if word1.head == 0: # root, ignore and continue
                continue

            for sent2 in doc2.sentences:
                for word2 in sent2.words:
                    if word2.head == 0: # root, ignore and continue
                        continue
                    # "When two grammatical relations are of different types, their similarity is zero"
                    if word1.deprel != word2.deprel: # if the relations don't match, don't look at them
                        continue
                    
                    simcount += 1

                    # word1 root, dependent
                    w1root, w1dep = word1.text, sent1.words[word1.head-1]

                    # word2 root, dependent
                    w2root, w2dep = word2.text, sent2.words[word2.head-1]

                    # dog.res_similarity(cat, brown_ic)
                    print("===")
                    print(w1root.root.res_similarity(w2root, story1))
                    print(w1root.root.res_similarity(w2root, story2))
                    
                    # "When the two relations belong to the same type, the similarity is the average of the word similarity between the governors and the word similarity between the dependents"
                    # rootsim = 0 / 2
                    # The semantic similarity between two words is computed based on WordNet (Miller, 1995). Empirically, we found the Resnik (1995) word similarity function to be the most useful.
                    # ssim = 0 / 2

    print(simcount)

    # print()

    # doc3 = nlp("Sally and John stood in line at the movie theater.")
    # print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc3.sentences for word in sent.words], sep='\n')

    # for story in stories:
    #     pass
    
###

if __name__ == "__main__":
    main()
