"""
The cluster pruner is a specific post-step to come between the clustering and initial script
generation. This class provides a few methods, but please read the specific function contract
to understand each one's purpose.

- purify_clusters => Will simply prune clusters down as much as possible. If allowed, it will destroy clusters
                    if any attempt at pruning them would reduce them below a certain threshold. Members of the clusters
                    will just be sent to the -1 cluster (or representative unclustered elements).
                    NOTE: This method is simply to improve information extraction, if it is present.
- purify_on_step => Will attempt to purify clusters based on where in the story the sentences are. That is,
                    if two sentences are similarly related (e.g. "John waited for his food", "John paid for his food")
                    but come after one or the other, it will put this in a later cluster.
"""

# imports

# code

def prune_clusters(clusters, min_size, cluster_purity, unclustered_key=-1):
    pass
