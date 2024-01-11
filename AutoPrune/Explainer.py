import torch

class Explainer:
    def __init__(self, model, translator=None, cluster=False):
        self.model = model
        self.cluster = cluster
        self.translator = translator

    def Explain(self, x, y=None):
        int_out, all_masks, hidden_weights, rule_weights = self.model.ExplainPrediction(x)

        if self.cluster:
            x = x[:-1]
            n_cluster = x[-1]

        # HIDDEN RULES PROCESSING
        hidden_passage = []
        hidden_passage_sent = []
        for r in range(len(hidden_weights)):
            hidden_passage.append(x * hidden_weights[r] * all_masks[r])
            sents = {'positive': [], 'negative': []}
            positions = torch.arange(len(all_masks[r].squeeze()))[all_masks[r].squeeze() != 0]
            for pos in positions:
                pos = int(pos)
                pos_sent = hidden_weights[r].squeeze()[pos] >= 0
                seen = int(x.squeeze()[pos])
                if pos_sent == 1:
                    sents['positive'].append([pos, seen])
                else:
                    sents['negative'].append([pos, seen])
            hidden_passage_sent.append(sents)

        # TRANSLATE (Optional)
        if self.translator is not None:
            for j, word_idx in enumerate(sents['positive']):
                word = self.translator(word_idx, n_cluster)
                sents['positive'][j] = word
            for j, word_idx in enumerate(sents['negative']):
                word = self.translator(word_idx, n_cluster)
                sents['negative'][j] = word


        # OUT PROCESSING - SIGN
        for j, rule_weight in enumerate(int_out[0].squeeze()):
            """if rule_weight < 0: #switch sentiments if negative weight sign
                positive = hidden_passage_sent[j]['positive']
                negative = hidden_passage_sent[j]['negative']
                hidden_passage_sent[j]['positive'] = negative
                hidden_passage_sent[j]['negative'] = positive"""
            if torch.round(int_out[-1].squeeze()) == 0:  # switch sentiments if prediction is not 1
                positive = hidden_passage_sent[j]['positive']
                negative = hidden_passage_sent[j]['negative']
                hidden_passage_sent[j]['positive'] = negative
                hidden_passage_sent[j]['negative'] = positive

        # setting all weights to positive
        # rule_weights = torch.abs(int_out[0])
        # OUT PROCESSING - WEIGHT
        perc_rule_weights = torch.abs(rule_weights) / torch.sum(torch.abs(rule_weights))

        # PRODUCE STRINGS
        rule_explanations = []
        for r in range(len(hidden_weights)):
            seen_pos_words = []
            not_seen_pos_words = []
            for sent in hidden_passage_sent[r]['positive']:
                word = sent[0]
                if sent[1] == 1:
                    seen_pos_words.append(word)
                else:
                    not_seen_pos_words.append(word)
            seen_neg_words = []
            not_seen_neg_words = []
            for sent in hidden_passage_sent[r]['negative']:
                word = sent[0]
                if sent[1] == 1:
                    seen_neg_words.append(word)
                else:
                    not_seen_neg_words.append(word)

            rule_string = ''
            # seen positive words
            for j, word in enumerate(seen_pos_words):
                if j == 0:
                    rule_string += 'Because I saw ' + str(word)
                    if j == len(seen_pos_words) - 1:
                        rule_string += '. '
                elif j == len(seen_pos_words) - 1:
                    rule_string += ' and ' + str(word) + '. '
                else:
                    rule_string += ', ' + str(word)
            # not seen negative words
            for j, word in enumerate(not_seen_neg_words):
                if j == 0:
                    rule_string += "Because I didn't saw " + str(word)
                    if j == len(not_seen_neg_words) - 1:
                        rule_string += '. '
                elif j == len(not_seen_neg_words) - 1:
                    rule_string += ' and ' + str(word) + '. '
                else:
                    rule_string += ', ' + str(word)
            # seen negative words
            for j, word in enumerate(seen_neg_words):
                if j == 0:
                    rule_string += 'Despite I saw ' + str(word)
                    if j == len(seen_neg_words) - 1:
                        rule_string += '. '
                elif j == len(seen_neg_words) - 1:
                    rule_string += ' and ' + str(word) + '. '
                else:
                    rule_string += ', ' + str(word)
            # not seen positive words
            for j, word in enumerate(not_seen_pos_words):
                if j == 0:
                    rule_string += "Despite I didn't saw " + str(word)
                    if j == len(not_seen_pos_words) - 1:
                        rule_string += '. '
                elif j == len(not_seen_pos_words) - 1:
                    rule_string += ' and ' + str(word) + '. '
                else:
                    rule_string += ', ' + str(word)

            rule_string += ': Contribution =  ' + str(round(int_out[0].squeeze()[r].detach().item(), 4))
            rule_string += ', Rule weight = ' + str(round(rule_weights.squeeze()[r].detach().item(), 4))
            # rule_string += ' (' + str(round(perc_rule_weights.squeeze()[r].detach().item()*100, 2)) + '% of total).'
            rule_explanations.append(rule_string)

        return rule_explanations
