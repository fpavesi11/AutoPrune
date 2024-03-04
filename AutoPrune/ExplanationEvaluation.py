import torch
from tqdm import tqdm
import numpy as np
# EXPLANATION SIZE: depth of decision tree, non-zero weights

# COVERAGE: Number of instances not covered by any rule
def CoverageCluster(model, data_loader, device='cpu'):
    model.to(device)
    model.eval()

    n_not_covered = 0  # <-- number of inputs for which output is zero because they fit in no rule
    tot = 0
    with torch.no_grad():
        for val_x, val_y in tqdm(data_loader):
            batch_size = val_x.size(0)
            tot += batch_size

            val_x = val_x.to(device)

            clusters = val_x[:, -1]
            inputs = val_x[:, :-1]

            num_clusters = len(model)
            num_rules = len(model[0][0])

            final_out = torch.empty((val_x.size(0), 1), device=val_x.device)  # <-- we only have binary classification
            for cluster in range(num_clusters):
                index = clusters == cluster
                cluster_mask = index.unsqueeze(-1).repeat_interleave(inputs.size(-1), -1)
                hidden_rules, rule_weight, dropout, weights_dropout = model[cluster]
                weights_masks = weights_dropout(hidden_rules, epoch=None, loss=None)

                rule_outputs = []
                for r in range(num_rules):
                    r_activated = hidden_rules[r](inputs * cluster_mask, weights_masks[r])
                    rule_outputs.append(r_activated)
                rule_outputs = torch.cat(rule_outputs, dim=-1)
                final_out[index] = rule_outputs.sum(-1).unsqueeze(-1)[index]
            n_not_covered += (final_out.squeeze() == 0).sum()
        perc_not_covered = n_not_covered / tot
    return n_not_covered, perc_not_covered


# RULE FREQUENCY: number of instances satisfying the rule
def RuleFrequencyCluster(model, data_loader, device='cpu'):
    model.to(device)
    model.eval()

    rule_freq = {}
    cluster_keys = [('cluster_' + str(i)) for i in range(len(model))]
    rule_keys = [('rule_' + str(i)) for i in range(len(model[0][0]))]  # <-- all clusters have same number of rules
    for cluster in cluster_keys:
        rule_freq[cluster] = {}
        for rule in rule_keys:
            rule_freq[cluster][rule] = 0.

    num_eval_batches = 0

    with torch.no_grad():
        for val_x, val_y in tqdm(data_loader):
            num_eval_batches += 1
            batch_size = val_x.size(0)

            val_x = val_x.to(device)

            clusters = val_x[:, -1]
            inputs = val_x[:, :-1]

            for j, cluster in enumerate(cluster_keys):
                index = clusters == j
                obs_in_cluster = sum(index)  # <--- otherwise we cannot catch the number of obs passing through
                cluster_mask = index.unsqueeze(-1).repeat_interleave(inputs.size(-1), -1)
                hidden_rules, rule_weight, dropout, weights_dropout = model[j]
                weights_masks = weights_dropout(hidden_rules, epoch=None, loss=None)

                for r, rule_name in enumerate(rule_keys):
                    r_passage = hidden_rules[r](inputs * cluster_mask,
                                                weights_masks[r])  # <----------------pass weights mask to linear module
                    perc_used = (
                                            r_passage.squeeze() != 0).sum() / obs_in_cluster  # <-- percentage of obs in cluster passing through this rule
                    rule_freq[cluster][rule_name] += perc_used
                    rule_freq[cluster][rule_name] /= num_eval_batches

    return rule_freq


# TOTAL NUMBER OF FEATURES USED BY THE MODEL


"""
FOLLOWING METRICS ARE ARGUABLE
"""
# DEGREE OF OVERLAPPING (CLUSTER LEVEL)
def OverlappingCluster(model):
    cluster_overlapping = []
    cluster_overlapping_perc = []
    for cluster in range(len(model)):
        base = len(model[cluster][0]) * len(model[cluster][-1].all_masks[0])
        for rule in range(len(model[cluster][0])):
            if rule == 0:
                rule_weight = model[cluster][0][rule].weight.data * model[cluster][-1].all_masks[rule]
            else:
                rule_weight *= model[cluster][0][rule].weight.data * model[cluster][-1].all_masks[rule]
        n_active_features = (rule_weight > 0).sum()
        perc_active = n_active_features/base
        perc_overlapping = 1 - perc_active
        cluster_overlapping.append(n_active_features)
        cluster_overlapping_perc.append(perc_overlapping)
    return cluster_overlapping, cluster_overlapping_perc

# DEGREE OF OVERLAPPIN (OVERALL)
def OverallFeatureOverlapping(model, translation_vocabs):
    all_words = []
    base = 0
    for num_cluster in range(len(model)):
        for num_rule in range(len(model[num_cluster][0])):
            idx = (model[num_cluster][-1].all_masks[num_rule] != 0).squeeze()
            all_pos = torch.arange(len(idx))[idx]
            base += len(all_pos)
            words = [translation_vocabs[num_cluster][i.item()] for i in all_pos]
            all_words.extend(words)
    all_words_array = np.array(all_words)
    perc_unique_words = len(np.unique(all_words_array))/base
    overlapping = 1 - perc_unique_words
    return overlapping





##### Both faithfulness and sensitivity does not have much sense. Their validity is instrinsic
# FAITHFULNESS Correlation Between model's performance drop when removing certain feature from the input
# and relevance score

# SENSITIVITY






