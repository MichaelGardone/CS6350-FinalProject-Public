# os
import os

# third party
import spacy, numpy

# my code
from cni_methods import WordVecModel as WVM
from util import file_read as fr

#TFIDF = term frequency inverse document frequency?

def main():
    # TODO: Change the model to lg
    parser = spacy.load('en_core_web_md')

    wv_model = WVM.WordVecModel(debug=True)
	
    # wv_model.generate_temporal_structure("stories/movie/reduced/reducedMovie.story", parser)
    # gold = fr.parse_gold("stories/movie/reduced/movieSemanticGold.gold")
	
    wv_model.generate_temporal_structure("stories/simple/RtSimpleStories.story", parser)
    gold = fr.parse_gold("stories/simple/RtSimpleGold.gold")
	
    print("Finished")

    # Purity
    total, per_cluster = wv_model.get_purity(gold)
    
    print("Total Purity:", total)
    for i in per_cluster:
        print(f"\tCluster {i}: {per_cluster[i]}")

    return 0
###

if __name__ == "__main__":
    main()
