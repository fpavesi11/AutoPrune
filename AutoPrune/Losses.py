from AutoPrune.Layers import *

class AlmostPenalizedLoss(nn.Module):
    def __init__(self, loss_fn, l1_lambda_hidden_rules=0.0, l2_lambda_hidden_rules=0.0, l1_lambda_rules_weights=0.0,
                 l2_lambda_rules_weights=0.0,
                 gini_lambda_rules_weights=0.0, gini_lambda_hidden_rules=0.0):
        super(AlmostPenalizedLoss, self).__init__()
        self.loss_fn = loss_fn
        self.l1_lambda_hidden_rules = l1_lambda_hidden_rules
        self.l2_lambda_hidden_rules = l2_lambda_hidden_rules
        self.l1_lambda_rules_weights = l1_lambda_rules_weights
        self.l2_lambda_rules_weights = l2_lambda_rules_weights
        self.gini_lambda_rules_weights = gini_lambda_rules_weights
        self.gini_lambda_hidden_rules = gini_lambda_hidden_rules

    def forward(self, y_pred, y_true, hidden_rules_parameters, rules_weights_parameters):
        base_loss = self.loss_fn(y_pred, y_true)

        l1_regularization_rules_parameters = 0
        l2_regularization_rules_parameters = 0
        ineq_hidden_penalty = 0

        if hidden_rules_parameters is not None:
            for param in hidden_rules_parameters:
                if self.l1_lambda_hidden_rules > 0.0:
                    l1_regularization_rules_parameters += torch.norm(param, 1)
                if self.l2_lambda_hidden_rules > 0.0:
                    l2_regularization_rules_parameters += torch.norm(param, 2)
                if self.gini_lambda_hidden_rules > 0.0:  # GINI REGULARIZATION
                    ineq_hidden_penalty += Gini(param)

        l1_regularization_rules_weights_parameters = 0
        l2_regularization_rules_weights_parameters = 0
        ineq_weight_penalty = 0

        if rules_weights_parameters is not None:
            for param in rules_weights_parameters:
                if self.l1_lambda_rules_weights > 0.0:
                    l1_regularization_rules_weights_parameters += torch.norm(param, 1)
                if self.l2_lambda_rules_weights > 0.0:
                    l2_regularization_rules_weights_parameters += torch.norm(param, 2)

                if self.gini_lambda_rules_weights > 0.0:  # <--- GINI REGULARIZATION
                    ineq_weight_penalty += Gini(param)

        hidden_rules_penalization = self.l1_lambda_hidden_rules * l1_regularization_rules_parameters + self.l2_lambda_hidden_rules * l2_regularization_rules_parameters
        rules_weights_penalization = self.l1_lambda_rules_weights * l1_regularization_rules_weights_parameters + self.l2_lambda_rules_weights * l2_regularization_rules_weights_parameters
        gini_rules_penalization = ineq_weight_penalty * self.gini_lambda_rules_weights
        gini_hidden_penalization = ineq_hidden_penalty * self.gini_lambda_hidden_rules * (
            -1)  # <--- opposite sign: we want them to be as dissimilar as possible

        loss = base_loss + hidden_rules_penalization + rules_weights_penalization + gini_rules_penalization + gini_hidden_penalization
        return loss