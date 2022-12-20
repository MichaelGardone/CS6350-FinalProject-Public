from scipy.special import comb

def single_tail_confidence(graph, stories, clusters, e1_e2_threshold):
    """
    The single-tail confidence method is the literal translation from the Crowdsourcing
    Narrative Intelligence paper. We accept before(e1, e2) if it is above 50% probability,
    via a single-tail probability distribution, otherwise we accept before(e2, e1).

    As a NOTE this is believed to be an incredibly strong hypothesis and a strong requirement,
    see also q_and_k_confidence and only_k_confidence methods.
    """
    confidence_library = {}

    for e1 in clusters.keys(): 
        e1_cluster = clusters[e1]
        for e2 in clusters.keys():
            if graph.has_edge(e1, e2): # no duplicate edges
                continue
            if e1 == e2: # self-loop prevention
                continue
                
            e2_cluster = clusters[e2]
                
            n = 0 # number of observations supporting support either before(e1,e2) or before(e2,e1)
            k = 0 # number of observations supporting before(e1,e2)

            for s1 in e1_cluster:
                for s2 in e2_cluster:
                    # ignore if s1 and s2 are not in the same story
                    if s1[1] != s2[1]:
                        continue
                        
                    # before(e1, e2) found
                    if stories[s1[1]].index(s1[0]) < stories[s1[1]].index(s2[0]):
                        n += 1
                        k += 1
                    # before(e2, e1) found
                    else:
                        n += 1
                
                confidence = 0
                exp = 2 ** n
                for i in range(k):
                    confidence += comb(n, i) * 1 / exp
                
                # output = "Confidence " + str(e1) + " < " + str(e2) + ": " + str(confidence) + " (" + ("before(e1,e2)" if confidence >= confidence_threshold else "before(e2,e1)") + ")"
                # print(output)

                # if confidence is passed, add before(e1, e2)
                if confidence > e1_e2_threshold:
                    graph.add_edge(e1, e2)
                    confidence_library[(e1,e2)] = confidence
                # otherwise, always add before(e2, e1)
                else:
                    graph.add_edge(e2, e1)
                    confidence_library[(e2,e1)] = confidence
                # low confidence edges will be removed!
        
    return confidence_library
###