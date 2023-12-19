from Layers import *


########################################################################################################################
"""
ALMOST OHNN
"""


class almostOneHotNN(nn.Module):
    def __init__(self, input_size, num_rules, num_neurons, initial_weight_val=0.2, final_activation=None,
                 rule_bias=False, force_positive_out_init=True, rule_activation=None,
                 out_bias=False, out_weight_constraint=None, dropout_perc=0.5, burn_in=10, dtype=torch.float64):

        super(almostOneHotNN, self).__init__()
        self.num_rules = num_rules
        self.out_weight_constraint = out_weight_constraint
        self.rule_activation = rule_activation()
        self.num_neurons = num_neurons

        # hidden rules
        self.hidden_rules = nn.ModuleList([
            OneActLinear(in_features=input_size, out_features=1, weight_val=initial_weight_val, bias=rule_bias,
                         dtype=dtype) for _ in
            range(num_rules)
        ])

        # rules' weight
        self.rule_weight = ruleDense(input_size=num_rules, num_neurons=1, weight_constraint=out_weight_constraint,
                                     bias=out_bias, force_positive_init=force_positive_out_init, dtype=dtype)

        self.dropout = CustomDimensionalDropout(dropout_perc)

        self.weights_dropout = CustomFeatureDropout(dropout_perc, num_neurons, burn_in)

        self.final_activation = None
        if final_activation is not None:
            self.final_activation = final_activation()

    def forward(self, x, epoch=None):
        # Input passes through all hidden rule dense modules, each output
        # is then summed up and concatenated
        # --> rule_outputs is n_obs x n_rules

        # weight dropout (activated after burn_in period)
        weights_masks = self.weights_dropout(self.hidden_rules, epoch)

        rule_outputs = []
        for r in range(self.num_rules):
            r_activated = self.rule_activation(self.hidden_rules[r](x, weights_masks[r])).unsqueeze(
                1)  # <----------------pass weights mask to linear module
            rule_outputs.append(r_activated)
        rule_outputs = torch.cat(rule_outputs, dim=-1)

        # dropout
        # rule_outputs = self.dropout(rule_outputs, self.get_rules_weights_params())

        # rules pass through a dense layer in which a positive weight to each rule is assigned
        # --> out is n_obs x 1 (we are still limited to binary classification)
        out = self.rule_weight(rule_outputs)

        if self.final_activation is not None:
            out = self.final_activation(out)

        return out

    def get_hidden_rules_params(self, apply_act=True):
        hidden_weights = []
        hidden_bias = None
        for rule in range(self.num_rules):
            weight = self.hidden_rules[rule].weight.detach()
            hidden_weights.append(weight)
            if self.hidden_rules[rule].bias is not None:
                hidden_bias = self.hidden_rules[rule].rule_bias.detach()
        return hidden_weights, hidden_bias

    def get_rules_weight(self, apply_act=True):
        rule_weight = self.rule_weight.linear.weight.detach()
        if apply_act and self.out_weight_constraint is not None:
            rule_weight = self.out_weight_constraint()(rule_weight)
        return rule_weight

    def ExplainPrediction(self, x):
        # Input passes through all hidden rule dense modules, each output
        # is then summed up and concatenated
        # --> rule_outputs is n_obs x n_rules
        intermediate_out = []
        rule_outputs = []
        for r in range(self.num_rules):
            r_activated = self.rule_activation(self.hidden_rules[r](x)).unsqueeze(1)
            rule_outputs.append(r_activated)
        rule_outputs = torch.cat(rule_outputs, dim=-1)

        intermediate_out.append(rule_outputs)

        # rules pass trough a dense layer in which a positive weight to each rule is assigned
        # --> out is n_obs x 1 (we are still limited to binary classification)
        out = self.rule_weight(rule_outputs)

        intermediate_out.append(out)

        if self.final_activation is not None:
            out = self.final_activation(out)

            intermediate_out.append(out)

        return intermediate_out

    def get_hidden_rules_params(self):
        return self.hidden_rules.parameters()

    def get_rules_weights_params(self):
        return self.rule_weight.parameters()

    def get_all_parameters(self):
        return self.parameters()

    def PruneNeurons(self, topk):
        for rule in range(self.hidden_rules):
            lower_limit = torch.topk(self.hidden_rules[rule].weight.data, k=topk)[0][topk - 1]
            mask = self.hidden_rules[rule].weight.data >= lower_limit
            self.hidden_rules[rule] = self.hidden_rules[rule].weight.data * mask



########################################################################################################################
"""
CLUSTER OHNN
#
#
#
#
#
#
"""


class ClusterOneHotNN(nn.Module):
    def __init__(self, input_size, n_clusters, num_rules, num_neurons, initial_weight_val=0.2, final_activation=None,
                 rule_bias=False, force_positive_out_init=True, rule_activation=None,
                 out_bias=False, out_weight_constraint=None, dropout_perc=0.5, burn_in=10, dtype=torch.float64):

        super(ClusterOneHotNN, self).__init__()
        self.n_clusters = n_clusters
        self.num_rules = num_rules
        self.out_weight_constraint = out_weight_constraint
        self.rule_activation = rule_activation()
        self.num_neurons = num_neurons

        # hidden rules
        self.model = nn.ModuleList([])
        for cluster in range(n_clusters):
            hidden_rules = nn.ModuleList([
                OneActLinear(in_features=input_size, out_features=1, weight_val=initial_weight_val, bias=rule_bias,
                             dtype=dtype) for _ in
                range(num_rules)
            ])

            # rules' weight
            rule_weight = ruleDense(input_size=num_rules, num_neurons=1, weight_constraint=out_weight_constraint,
                                    bias=out_bias, force_positive_init=force_positive_out_init, dtype=dtype)

            dropout = CustomDimensionalDropout(dropout_perc)

            weights_dropout = CustomFeatureDropout(dropout_perc, num_neurons, burn_in)
            pipeline = nn.ModuleList([hidden_rules, rule_weight, dropout, weights_dropout])
            self.model.append(pipeline)

        self.final_activation = None
        if final_activation is not None:
            self.final_activation = final_activation()

    def forward(self, x, epoch=None):
        # Input passes through all hidden rule dense modules, each output
        # is then summed up and concatenated
        # --> rule_outputs is n_obs x n_rules

        # weight dropout (activated after burn_in period)
        clusters = x[:, -1]
        inputs = x[:, :-1]

        final_out = torch.empty((x.size(0), 1))  # <-- we only have binary classification
        for cluster in range(self.n_clusters):
            index = clusters == cluster
            cluster_mask = index.unsqueeze(-1).repeat_interleave(inputs.size(-1), -1)
            hidden_rules, rule_weight, dropout, weights_dropout = self.model[cluster]
            weights_masks = weights_dropout(hidden_rules, epoch)

            rule_outputs = []
            for r in range(self.num_rules):
                r_activated = self.rule_activation(hidden_rules[r](inputs * cluster_mask, weights_masks[r])).unsqueeze(1)  # <----------------pass weights mask to linear module
                rule_outputs.append(r_activated)
            rule_outputs = torch.cat(rule_outputs, dim=-1)

            out = rule_weight(rule_outputs)
            final_out[index] = out[index].squeeze(-1)

        #final_out = torch.sum(final_out)

        if self.final_activation is not None:
            final_out = self.final_activation(final_out)

        return final_out

    def __get_hidden_rules_params(self, apply_act=True):
        hidden_weights = []
        hidden_bias = None
        for cluster in range(self.n_clusters):
            hidden_rules, rule_weight, dropout, weights_dropout = self.model[cluster]
            for rule in range(self.num_rules):
                weight = hidden_rules[rule].weight.detach()
                hidden_weights.append(weight)
                if hidden_rules[rule].bias is not None:
                    hidden_bias = hidden_rules[rule].rule_bias.detach()
        return hidden_weights, hidden_bias

    def __get_rules_weight(self, apply_act=True):
        rule_weights = []
        for cluster in range(self.n_clusters):
            hidden_rules, rule_weight, dropout, weights_dropout = self.model[cluster]
            rule_weight = rule_weight.linear.weight.detach()
            if apply_act and self.out_weight_constraint is not None:
                rule_weight = self.out_weight_constraint()(rule_weight)
            return rule_weight

    def ExplainPrediction(self, x):
        # Input passes through all hidden rule dense modules, each output
        # is then summed up and concatenated
        # --> rule_outputs is n_obs x n_rules
        intermediate_out = []
        rule_outputs = []
        for r in range(self.num_rules):
            r_activated = self.rule_activation(self.hidden_rules[r](x)).unsqueeze(1)
            rule_outputs.append(r_activated)
        rule_outputs = torch.cat(rule_outputs, dim=-1)

        intermediate_out.append(rule_outputs)

        # rules pass trough a dense layer in which a positive weight to each rule is assigned
        # --> out is n_obs x 1 (we are still limited to binary classification)
        out = self.rule_weight(rule_outputs)

        intermediate_out.append(out)

        if self.final_activation is not None:
            out = self.final_activation(out)

            intermediate_out.append(out)

        return intermediate_out

    def get_hidden_rules_params(self):
        hidden_rules_list = nn.ModuleList([])
        for cluster in range(self.n_clusters):
            hidden_rules, rule_weight, dropout, weights_dropout = self.model[cluster]
            hidden_rules_list.append(hidden_rules)
        return hidden_rules.parameters()

    def get_rules_weights_params(self):
        rule_weight_list = nn.ModuleList([])
        for cluster in range(self.n_clusters):
            hidden_rules, rule_weight, dropout, weights_dropout = self.model[cluster]
            rule_weight_list.append(rule_weight)
        return rule_weight_list.parameters()

    def get_all_parameters(self):
        return self.parameters()

    def PruneNeurons(self, topk):
        for rule in range(self.hidden_rules):
            lower_limit = torch.topk(self.hidden_rules[rule].weight.data, k=topk)[0][topk - 1]
            mask = self.hidden_rules[rule].weight.data >= lower_limit
            self.hidden_rules[rule] = self.hidden_rules[rule].weight.data * mask