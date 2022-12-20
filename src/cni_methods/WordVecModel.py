import spacy, numpy
from spacy.vectors import Vectors
from sklearn.cluster import OPTICS
import networkx as nx
from scipy.special import comb
from copy import deepcopy

# Utility methods provided in src/util
from util import file_read as fr
from util import gen_graph as gg
from util import msge as msge
from util import confidence as conf

class WordVecModel():
    """
    Using Word Vectors as the primary source of modeling. This differs significantly
    from the paper, which used Wordnet + Resnik but produces a decent enough result in
    terms of graph sanity. It is signifiantly faster than the Stanza-based model, which makes
    sense as spaCy (the backing library of this specific model) uses distributions rather than
    a deep neural network for parsing; of course, this is at the cost of accuracy in labeling
    and the word vector itself.

    HYPER PARAMETERS:
    - e1e2_confidence => Minimum confidence to add the edge before(e1,e2)\n
    - e2e1_confidence => Minimum confidence to add the edge before(e2,e1).
            confidence_func must be 3 to be used.\n
    - min_samples     => The minimum number of samples required for OPTICS to consider a grouping of
            sentences cluster.\n
    - stepsize        => The maximum slope used in the xi algorithm.\n
    - purity_minimum  => The minimum amount allowed for a cluster to be considered "pure" enough.
            This is the minimum PERCENTAGE allowed, e.g. pluster is considered impure if the purity
            calculation comes out to <X% of sentences. If optimize_purity is set, it will attempt to
            optimize the contents of the cluster. Otherwise, the cluster is dropped and considered "unpure".\n
    """

    def __init__(self, e1e2_confidence=0.5, e2e1_confidence=0.5,
                        confidence_func=0,
                        use_purity=False, allowUnclustered=False, debug=False) -> None:
        """
        :param: e1e2_confidence => Minimum confidence to add the edge before(e1,e2).
        :param: e2e1_confidence => Minimum confidence to add the edge before(e2,e1). confidence_func must be 3 to be used.
        :param: confidence_func => Which version of the confidence function to use when doing initial graph generation.
            - 0 = Default, as (vaguely) described in the paper. This will add at least one edge.
            - 1 = Without the else statement described in the paper.
            - 2 = With a separate count of before(e2, e1).
            - 3 = With a separate count of before(e2, e1) with its own confidence threshold.
        
        === DEBUG PARAMS ===

        :param: use_purity, Default=False       => Maybe?
        :param: allowUnclustered, Default=False => Allow the unclustered (-1) events to be used in the hieararchy.
            For peace of heart and mind, the author recommends against this.
        :param: debug, Default=False            => Allow debug outputs from the visualizers. This can significantly slow down
            execution speed as multiple graphs are generated, one at each major checkpoint.
        """

        self._e1_e2_threshold = e1e2_confidence
        self._e2_e1_threshold = e2e1_confidence
        self._unclustered_impact = allowUnclustered
        self._debug = debug
        self._OPTICS_min_samples = 0.025
        self._OPTICS_stepsize = 0.05
        self._use_purity = use_purity

        """
            Which version of the confidence function to use when doing initial graph generation.
            0 = Default, as (vaguely) described in the paper
            1 = Without the else statement described in the paper
            2 = With a separate count of before(e2, e1)
            3 = With a separate count of before(e2, e1) with its own confidence threshold
        """
        self._confidence_function = confidence_func

        self._stories = None
        self._parser = None
        self._flattened = None
        self._graph = nx.DiGraph()
        self._clusters = None
    ###

    def _confidence0(self, clusters):
        confidence_library = {}

        for e1 in clusters.keys():
        
            e1_cluster = clusters[e1]

            for e2 in clusters.keys():
                if self._graph.has_edge(e1, e2): # no duplicate edges
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
                        if self._stories[s1[1]].index(s1[0]) < self._stories[s1[1]].index(s2[0]):
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
                if confidence > self._e1_e2_threshold:
                    self._graph.add_edge(e1, e2)
                    confidence_library[(e1,e2)] = confidence
                # otherwise, always add before(e2, e1)
                else:
                    self._graph.add_edge(e2, e1)
                    confidence_library[(e2,e1)] = confidence
                # low confidence edges will be removed!
        
        return confidence_library
    ###

    def _confidence1(self, clusters):
        confidence_library = {}

        for e1 in clusters.keys():
        
            e1_cluster = clusters[e1]

            for e2 in clusters.keys():
                if self._graph.has_edge(e1, e2): # no duplicate edges
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
                        if self._stories[s1[1]].index(s1[0]) < self._stories[s1[1]].index(s2[0]):
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
                if confidence > self._e1_e2_threshold:
                    self._graph.add_edge(e1, e2)
                    confidence_library[(e1,e2)] = confidence
                # low confidence edges will be removed!
        
        return confidence_library
    ###

    def _confidence2(self, clusters):
        confidence_library = {}

        for e1 in clusters.keys():
        
            e1_cluster = clusters[e1]

            for e2 in clusters.keys():
                if self._graph.has_edge(e1, e2): # no duplicate edges
                    continue
                if e1 == e2: # self-loop prevention
                    continue
                
                e2_cluster = clusters[e2]
                
                n = 0 # number of observations supporting support either before(e1,e2) or before(e2,e1)
                k = 0 # number of observations supporting before(e1,e2)
                q = 0 # number of observations supporting before(e2, e1)

                for s1 in e1_cluster:
                    for s2 in e2_cluster:
                        # ignore if s1 and s2 are not in the same story
                        if s1[1] != s2[1]:
                            continue
                        
                        # before(e1, e2) found
                        if self._stories[s1[1]].index(s1[0]) < self._stories[s1[1]].index(s2[0]):
                            n += 1
                            k += 1
                        # before(e2, e1) found
                        else:
                            n += 1
                            q += 1
                
                # this doesn't change between the two
                exp = 2 ** n

                conf_e1e2 = 0
                for i in range(k):
                    conf_e1e2 += comb(n, i) * 1 / exp
                
                conf_e2e1 = 0
                for i in range(q):
                    conf_e2e1 += comb(n, i) * 1 / exp
                
                # output = "Confidence " + str(e1) + " < " + str(e2) + ": " + str(confidence) + " (" + ("before(e1,e2)" if confidence >= confidence_threshold else "before(e2,e1)") + ")"
                # print(output)

                # if confidence is passed, add before(e1, e2)
                if conf_e1e2 > self._e1_e2_threshold:
                    self._graph.add_edge(e1, e2)
                    confidence_library[(e1,e2)] = conf_e1e2
                # otherwise, always add before(e2, e1)
                if conf_e2e1 > self._e1_e2_threshold:
                    self._graph.add_edge(e2, e1)
                    confidence_library[(e2,e1)] = conf_e2e1
                # low confidence edges will be removed!
        
        return confidence_library
    ###

    def _confidence3(self, clusters):
        confidence_library = {}

        for e1 in clusters.keys():
        
            e1_cluster = clusters[e1]

            for e2 in clusters.keys():
                if self._graph.has_edge(e1, e2): # no duplicate edges
                    continue
                if e1 == e2: # self-loop prevention
                    continue
                
                e2_cluster = clusters[e2]
                
                n = 0 # number of observations supporting support either before(e1,e2) or before(e2,e1)
                k = 0 # number of observations supporting before(e1,e2)
                q = 0 # number of observations supporting before(e2, e1)

                for s1 in e1_cluster:
                    for s2 in e2_cluster:
                        # ignore if s1 and s2 are not in the same story
                        if s1[1] != s2[1]:
                            continue
                        
                        # before(e1, e2) found
                        if self._stories[s1[1]].index(s1[0]) < self._stories[s1[1]].index(s2[0]):
                            n += 1
                            k += 1
                        # before(e2, e1) found
                        else:
                            n += 1
                            q += 1
                
                # this doesn't change between the two
                exp = 2 ** n

                conf_e1e2 = 0
                for i in range(k):
                    conf_e1e2 += comb(n, i) * 1 / exp
                
                conf_e2e1 = 0
                for i in range(q):
                    conf_e2e1 += comb(n, i) * 1 / exp
                
                # output = "Confidence " + str(e1) + " < " + str(e2) + ": " + str(confidence) + " (" + ("before(e1,e2)" if confidence >= confidence_threshold else "before(e2,e1)") + ")"
                # print(output)

                # if confidence is passed, add before(e1, e2)
                if conf_e1e2 > self._e1_e2_threshold:
                    self._graph.add_edge(e1, e2)
                    confidence_library[(e1,e2)] = conf_e1e2
                # otherwise, always add before(e2, e1)
                if conf_e2e1 > self._e2_e1_threshold:
                    self._graph.add_edge(e2, e1)
                    confidence_library[(e2,e1)] = conf_e2e1
                # low confidence edges will be removed!
        
        return confidence_library
    ###

    def _prune_graph(self, confidence_library):
        loops = nx.simple_cycles(self._graph)
        for loop in loops:
            # print(loop)
            min_conf = 1000
            min_pair = ()
            #find least confident edge and remove it
            for i, node in enumerate(loop):
                if i + 1 < len(loop):
                    if confidence_library[(loop[i], loop[i+1])] < min_conf:
                        min_conf = confidence_library[(loop[i], loop[i+1])]
                        min_pair = (node, loop[i+1])
                else:
                    if confidence_library[(loop[i], loop[0])] < min_conf:
                        min_conf = confidence_library[(loop[i], loop[0])]
                        min_pair = (node, loop[0])
            
            # only remove if the edge still exists, it might not because of prior pruning
            if self._graph.has_edge(min_pair[0], min_pair[1]):
                self._graph.remove_edge(min_pair[0], min_pair[1])
    ###

    def _OPTICS_predictions(self, data):
        clusterer = OPTICS(min_samples=float(self._OPTICS_min_samples), xi=self._OPTICS_stepsize)
        predictions = clusterer.fit_predict(data)

        clusters = None
        if self._unclustered_impact:
            # unclustered events should be ordered
            # really probably not, but I'll leave this here as a thought exercise
            clusters = {int(key): [] for key in predictions}
        else:
            # ignore unclustered events
            clusters = {int(key): [] for key in predictions}
            clusters.pop(-1)
        
        for i, sent in enumerate(self._flattened_sents):
            if self._unclustered_impact == False and predictions[i] == -1:
                continue

            clusters[predictions[i]].append(sent)
        
        # TODO: Cluster purity

        return clusters
    ###

    def _average_word_vectors(self) -> numpy.vstack:
        story_vectors = []

        for story in self._stories:
            for sent in story:
                curr_sent = []
                sent = sent.replace(".", "").strip()
                
                for word in sent.split():
                    word_vec = self._parser(word).vector
                    # print(word_vec)
                    curr_sent.append(word_vec)
                
                word_avg = numpy.mean(numpy.array(curr_sent), axis=0)
                story_vectors.append(word_avg)

        return numpy.vstack(story_vectors)
    ###

    def generate_temporal_structure(self, stories, parser) -> None:
        # Pre-processing of data
        self._parser = parser
        self._stories = fr.parse_file(stories)
        self._flattened_sents = [(sent, story_i) for story_i in range(len(self._stories))\
                                            for sent in self._stories[story_i]]

        # Get the mean word vector of each sentence
        avg_story_vectors = self._average_word_vectors()

        # Generate the clusters (read: events) that each sentence belongs to
        self._clusters = self._OPTICS_predictions(avg_story_vectors)

        # TODO: Post-cluster pruning + reorganization

        nodes = []

        for ni in self._clusters.keys():
            nodes.append((ni, {'title':f'{ni}'}))

        # self._graph.add_nodes_from(self._clusters.keys())
        self._graph.add_nodes_from(nodes)

        conf_lib = None
        if self._confidence_function == 0:
            conf_lib = self._confidence0(self._clusters)
        elif self._confidence_function == 1:
            conf_lib = self._confidence1(self._clusters)
        elif self._confidence_function == 2:
            conf_lib = self._confidence2(self._clusters)
        elif self._confidence_function == 3:
            conf_lib = self._confidence3(self._clusters)
        
        # At this point, the initial graph is finished and can be dumped and viewed
        if self._debug:
            gg.drawGraph(self._graph, "output/unpruned_graph.html")
        
        self._prune_graph(conf_lib)

        # At this point, a slimmed down version of the initial graph is finished and can be dumped and viewed
        if self._debug:
            gg.drawGraph(self._graph, "output/pruned_graph.html")

        # Purification by MSGE
        self._SGIA()

        # Done! Draw the graph out!
        gg.drawGraph(self._graph, "output/final_graph.html")
    ###

    def _SGIA(self):
        """
        [S]cript [G]raph [I]mprovemnt [A]lgorithm. Table 3 from the Crowdsourcing Narrative Intelligence paper.
        """
        DN = msge.compute_DN(self._stories, self._clusters)
        DG = msge.compute_DG(self._graph, DN) # this may change with updates

        P  = []
        for e in DN:
            if e[0] not in P:
                P.append(e[0])
        
        # current baseline, compare against this
        error = msge.compute_MSGE(DN, DG, P, len(P))

        # Q := all of events (e1, e2) such that e2 is reachable from e1 or unordered
        Q = DN.keys() # << all potential orderings, including not in the graph
        Q = sorted(Q, key=lambda e: DN[e] - DG[e], reverse=True)
        events = self._clusters
        
        graph = deepcopy(self._graph)

        for e12 in Q:
            #   E := the set of event ei that satisfy DG(e1, ei) = DN(e1, e2) – 1 
            E = [e for e in events if e != e12[0] and DG[(e12[0], e)] == DN[e12] - 1]
            #   foreach ei ∈ E do:
            for e in E:
                # If edge ei → e2 is not in the graph and adding it to the graph will not create a cycle do:
                copygraph = deepcopy(graph)
                if nx.has_path(copygraph, e, e12[1]) == False:
                    # Add ei → e2 to the graph
                    copygraph.add_edge(e, e12[1])

                    # if no cycle was found, did we make MSGE better?
                    if len(list(nx.simple_cycles(copygraph))) == 0:
                        ndg = msge.compute_DG(copygraph, DN)
                        nerr = msge.compute_MSGE(DN, ndg, P, len(P))

                        # if the new error is better than current error, use that graph
                        if nerr < error:
                            graph = copygraph
                            error = nerr
            ##
        ##
        
        self._graph = graph
        # return graph --> graph is an object and directly modified
        # Nothing to return, in a class, use self._graph
    ###

    def get_purity(self, gold):
        # cluster_i : purity score
        clust_purity = {key:0 for key in self._clusters.keys()}
        # total purity over the story
        tp = 0

        for cj in clust_purity.keys():
            ccj = len(self._clusters[cj])

            # strip what story that's from, don't care
            all_sents = [s[0] for s in self._clusters[cj]]
            
            # what cluster does this fit the best from gold?
            best = -1
            best_gc = None
            for gc in gold:
                inter = len(list(set(all_sents) & set(gc)))
                if inter > best:
                    best = inter
                    best_gc = gc
            ##
            clust_purity[cj] = 1 / ccj * len(best_gc)
        ##
        
        N = len(self._flattened_sents)

        for cj in clust_purity.keys():
            tp += len(self._clusters[cj]) * clust_purity[cj]
        
        tp *= 1 / N

        return tp, clust_purity
    ##

    def get_graph(self) -> nx.DiGraph():
        return self._graph
    ###

###