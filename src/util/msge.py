import numpy
import networkx as nx

def compute_DN(stories, clusters):
    """
    The normative distance from e1 to e2 averaged over the entire set of narratives.
    For each input narrative that includes sentence s1 from the cluster representing e1
    and sentence s2 from the cluster representing e2, the distance (i.e. number of interstitial
    sentences plus one) between s1 and s2 is dN(s1, s2). DN(e1, e2) is thus the average of
    dN(s1, s2) over all such input narratives.
    """
    DN = {}

    for e1 in clusters.keys():
        cluster1 = clusters[e1]

        for e2 in clusters.keys():
            if e1 == e2: # skip if the clusters are the same
                continue
            
            cluster2 = clusters[e2]

            total = 0
            interstitial = 0

            for s1 in cluster1:
                for s2 in cluster2:
                    if s1[1] != s2[1]:
                        continue
                    
                    total += 1

                    s1index = stories[s1[1]].index(s1[0])
                    s2index = stories[s2[1]].index(s2[0])

                    interstitial += int(numpy.abs(s2index - s1index))
                ##
            ##

            interstitial += 1
            
            if total > 0:
                DN[(e1,e2)] = interstitial / total
            else:
                # If we have no sentences between these two edges, record as 0
                DN[(e1,e2)] = 0
        ##
    ##

    return DN
###
    
def compute_DG(graph, DN):
    """
    The number of events on the shortest path from e1 to e2 on the graph (e1 excluded).
    """
    DG = {}

    for key in DN.keys():
        if nx.has_path(graph, key[0], key[1]) == False:
            DG[(key[0], key[1])] = 0    # skip if there
        else:
            # print(f"\t({key[0]},{key[1]}) = {shortest_paths[key][key2]} ({len(shortest_paths[key][key2])})")
            DG[(key[0], key[1])] = nx.shortest_path_length(graph, key[0], key[1]) - 1
        ##
    ##

    return DG
###
    
def compute_MSGE(DN, DG, P, lenP):
    MSGE = 0
    
    # P is the set of all ordered event pairs (e1, e2) such that
    #   e2 is reachable from e1 or that they are unordered
    invP = 1 / lenP

    for e1 in P:
        inner_accum = 0
        for e2 in P:
            # Skip if either doesn't have it
            if (e1,e2) not in DG or (e1,e2) not in DN: continue

            inner_accum += (DG[(e1,e2)] - DN[(e1,e2)]) ** 2
        MSGE += inner_accum
    ##
    
    return invP * MSGE
###
