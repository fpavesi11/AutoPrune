from AutoPrune.Models import *


"""
REGULAR MODEL TRAINER
"""

class RegularModelTrainer:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.history = {'train_loss': [],
                        'train_accuracy': [],
                        'validation_loss': [],
                        'validation_accuracy': [],
                        'f1_class0': [],
                        'f1_class1': []}
        self.state_dict = []
        self.best_configuration = None
        self.observer_history = {}
        for rule in range(self.model.num_rules):
            name = 'rule_' + str(rule)
            self.observer_history[name] = []
        self.observer_history['rule_weight'] = []
        # because observer history will store both forward and backward weights,
        # we split them into forward and backward
        self.forward_history = {}
        self.backward_history = {}
        for rule in range(self.model.num_rules):
            name = 'rule_' + str(rule)
            self.forward_history[name] = []
            self.backward_history[name] = []
        self.forward_history['rule_weight'] = []
        self.backward_history['rule_weight'] = []

        self.grad_history = {}
        for rule in range(self.model.num_rules):
            name = 'rule_' + str(rule)
            self.grad_history[name] = []
        self.grad_history['rule_weight'] = []

    def observer(self):  # keeps track of weights evolution
        for rule in range(self.model.num_rules):
            name = 'rule_' + str(rule)
            self.observer_history[name].append(self.model.hidden_rules[rule].weight.detach().clone())
        self.observer_history['rule_weight'].append(self.model.rule_weight.linear.weight.detach().clone())

    def grad_observer(self):
        for rule in range(self.model.num_rules):
            name = 'rule_' + str(rule)
            self.grad_history[name].append(self.model.hidden_rules[rule].weight.grad.detach().clone())
        self.grad_history['rule_weight'].append(self.model.rule_weight.linear.weight.grad.detach().clone())

    def get_weight_history(self, num_rule, apply_act=True):
        if isinstance(num_rule, int):
            num_rule = 'rule_' + str(num_rule)
        if not apply_act:
            hist = [self.observer_history[num_rule][i].unsqueeze(0) for i in
                    range(len(self.observer_history[num_rule]))]
        else:
            act = self.model.rule_weight_constraint
            hist = [act()(self.observer_history[num_rule][i]).unsqueeze(0) for i in
                    range(len(self.observer_history[num_rule]))]
        hist = torch.cat(hist, dim=0)
        return hist

    def train_model(self, train_data_loader, num_epochs, device, learning_rate, val_data_loader=None,
                    store_weights_history=True, store_weights_grad=True, display_log='batch'):

        if display_log not in ['batch', 'epoch', 'quiet']:
            raise ValueError('Options for display_log are "batch", "epoch" or "quiet"')

        if device != 'cpu':
            gc.collect()
            torch.cuda.empty_cache()

        self.optimizer = self.optimizer(self.model.parameters(), lr=learning_rate)

        min_loss = 1e5

        # epoch log check
        epoch_range = range(num_epochs)
        if display_log == 'epoch':
            epoch_range = tqdm(epoch_range)

        training_loss = None
        loss = None
        for epoch in epoch_range:
            self.model.train()

            total_loss = 0
            correct = 0
            total = 0

            # batch log check
            train_iterations = train_data_loader
            if display_log == 'batch':
                train_iterations = tqdm(train_iterations)

            for x_train, y_train in train_iterations:

                self.model.to(device)

                x_train, y_train = x_train.float(), y_train.float()

                self.optimizer.zero_grad()


                if loss is not None:
                    # it should not give error
                    # batch_total is previous batch_total
                    training_loss = loss.item()/batch_total

                batch_total = len(y_train)
                total += batch_total

                x_train = x_train.to(device)
                y_train = y_train.to(device)

                outputs = self.model(x_train,
                                     epoch,
                                     training_loss).squeeze()  # <---------------------pass the epoch to model for burn in

                predicted = torch.round(outputs)

                batch_correct = (predicted.squeeze() == y_train.squeeze()).sum().item()
                correct += batch_correct

                loss = self.loss_fn(outputs.squeeze(), y_train.squeeze(),
                                    hidden_rules_parameters=self.model.get_hidden_rules_params(),
                                    rules_weights_parameters=self.model.get_rules_weights_params())

                total_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                f1 = f1_score(predicted.to('cpu').detach().numpy(),
                              y_train.to('cpu').detach().numpy(),
                              average=None,
                              zero_division=0,
                              labels=np.array([0, 1]))

                # batch log description
                if display_log == 'batch':
                    train_iterations.set_description(
                        'batch_loss: {:.4f}, batch_accuracy: {:.4f}, f1 score 0: {:.4f}, f1 score 1: {:.4f}'.format(
                            loss.item() / batch_total, batch_correct / batch_total, f1[0], f1[1]))

            avg_loss = total_loss / total
            accuracy = correct / total

            if val_data_loader is not None:
                val_loss, val_accuracy, val_f1_0, val_f1_1 = self.eval_model(val_data=val_data_loader,
                                                                             device=device)
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)
                self.history['val_f1_0'].append(val_f1_0)
                self.history['val_f1_1'].append(val_f1_1)
            else:
                val_loss, val_accuracy, val_f1_0, val_f1_1 = None, None, None, None

            self.history['train_loss'].append(avg_loss)
            self.history['train_accuracy'].append(accuracy)

            if store_weights_history:
                self.observer()  # <-- store weight history

            if store_weights_grad:
                self.grad_observer()  # <-- stores weights gradient

            if val_data_loader is not None:
                if val_loss < min_loss:
                    min_loss = val_loss
                    save_model = self.model.clone()
                    self.best_configuration = [save_model.to('cpu').state_dict(), val_loss, epoch]

            if display_log == 'epoch':
                epoch_range.set_description(
                    'epoch_loss: {:.4f}, epoch_accuracy: {:.4f}'.format(avg_loss, accuracy))

    def eval_model(self, val_data, device='cpu'):
        if device != 'cpu':
            gc.collect()
            torch.cuda.empty_cache()

        self.model.to(device)
        self.model.eval()

        eval_loss = 0
        total_correct = 0
        total = 0
        f1_0 = 0
        f1_1 = 0
        num_iter = 0

        with torch.no_grad():
            for val_x, val_y in tqdm(val_data):
                total += len(val_y)
                val_x = val_x.to(device)
                val_y = val_y.to(device)

                outputs = self.model(val_x).squeeze()

                batch_pred = torch.round(outputs)

                loss = self.loss_fn(outputs.float(), val_y.float(),
                                    hidden_rules_parameters=self.model.get_hidden_rules_params(),
                                    rules_weights_parameters=self.model.get_rules_weights_params())

                eval_loss += loss.item()

                batch_correct = (batch_pred == val_y).sum().item()
                total_correct += batch_correct

                f1 = f1_score(batch_pred.to('cpu').detach().numpy(),
                              val_y.to('cpu').detach().numpy(),
                              average=None,
                              zero_division=0,
                              labels=np.array([0, 1]))

                f1_0 += f1[0]
                f1_1 += f1[1]
                num_iter += 1

        eval_loss = eval_loss / total
        accuracy = total_correct / total
        f1_0 /= num_iter
        f1_1 /= num_iter

        return eval_loss, accuracy, f1_0, f1_1

    def predict(self, val_data, device, prob=True):
        if device != 'cpu':
            gc.collect()
            torch.cuda.empty_cache()

        self.model.to(device)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for val_x, val_y in tqdm(val_data):
                val_x = val_x.to(device)
                val_y = val_y.to(device)
                outputs = self.model(val_x).squeeze()
                if not prob:
                    outputs = torch.round(outputs)
                predictions.append(outputs.to('cpu'))
        predictions = torch.cat(predictions, axis=0)
        return predictions


#####################################################################################################################à


class ClusterModelTrainer:
    def __init__(self, model, loss_fn, optimizer):
        raise DeprecationWarning('This trainer is no longer valid for cluster-model training, '
                                 'RegularModelTrainer is now valid also for cluster-model')
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.history = {'train_loss': [],
                        'train_accuracy': [],
                        'validation_loss': [],
                        'validation_accuracy': [],
                        'f1_class0': [],
                        'f1_class1': []}
        self.state_dict = []
        self.best_configuration = None
        self.observer_history = {}
        for rule in range(self.model.num_rules):
            name = 'rule_' + str(rule)
            self.observer_history[name] = []
        self.observer_history['rule_weight'] = []
        # because observer history will store both forward and backward weights,
        # we split them into forward and backward
        self.forward_history = {}
        self.backward_history = {}
        for rule in range(self.model.num_rules):
            name = 'rule_' + str(rule)
            self.forward_history[name] = []
            self.backward_history[name] = []
        self.forward_history['rule_weight'] = []
        self.backward_history['rule_weight'] = []

        self.grad_history = {}
        for rule in range(self.model.num_rules):
            name = 'rule_' + str(rule)
            self.grad_history[name] = []
        self.grad_history['rule_weight'] = []

    def observer(self):  # keeps track of weights evolution
        for rule in range(self.model.num_rules):
            name = 'rule_' + str(rule)
            self.observer_history[name].append(self.model.hidden_rules[rule].weight.detach().clone())
        self.observer_history['rule_weight'].append(self.model.rule_weight.linear.weight.detach().clone())

    def grad_observer(self):
        for rule in range(self.model.num_rules):
            name = 'rule_' + str(rule)
            self.grad_history[name].append(self.model.hidden_rules[rule].weight.grad.detach().clone())
        self.grad_history['rule_weight'].append(self.model.rule_weight.linear.weight.grad.detach().clone())

    def get_weight_history(self, num_rule, apply_act=True):
        if isinstance(num_rule, int):
            num_rule = 'rule_' + str(num_rule)
        if not apply_act:
            hist = [self.observer_history[num_rule][i].unsqueeze(0) for i in
                    range(len(self.observer_history[num_rule]))]
        else:
            act = self.model.rule_weight_constraint
            hist = [act()(self.observer_history[num_rule][i]).unsqueeze(0) for i in
                    range(len(self.observer_history[num_rule]))]
        hist = torch.cat(hist, dim=0)
        return hist

    def train_model(self, train_data_loader, num_epochs, device, learning_rate, val_data_loader=None,
                    store_weights_history=True, store_weights_grad=True, display_log='batch'):

        if display_log not in ['batch', 'epoch', 'quiet']:
            raise ValueError('Options for display_log are "batch", "epoch" or "quiet"')

        if device != 'cpu':
            gc.collect()
            torch.cuda.empty_cache()

        self.optimizer = self.optimizer(self.model.parameters(), lr=learning_rate)

        min_loss = 1e5

        # epoch log check
        epoch_range = range(num_epochs)
        if display_log == 'epoch':
            epoch_range = tqdm(epoch_range)

        for epoch in epoch_range:
            self.model.train()

            total_loss = 0
            correct = 0
            total = 0

            # batch log check
            train_iterations = train_data_loader
            if display_log == 'batch':
                train_iterations = tqdm(train_iterations)

            for x_train, y_train in train_iterations:

                self.model.to(device)

                x_train, y_train = x_train.float(), y_train.float()

                self.optimizer.zero_grad()

                batch_total = len(y_train)
                total += batch_total

                x_train = x_train.to(device)
                y_train = y_train.to(device)
                outputs = self.model(x_train,
                                     epoch).squeeze()  # <---------------------pass the epoch to model for burn in

                predicted = torch.round(outputs)

                batch_correct = (predicted == y_train).sum().item()
                correct += batch_correct

                loss = self.loss_fn(outputs, y_train,
                                    hidden_rules_parameters=self.model.get_hidden_rules_params(),
                                    rules_weights_parameters=self.model.get_rules_weights_params())

                total_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                f1 = f1_score(predicted.to('cpu').detach().numpy(),
                              y_train.to('cpu').detach().numpy(),
                              average=None,
                              zero_division=0,
                              labels=np.array([0, 1]))

                # batch log description
                if display_log == 'batch':
                    train_iterations.set_description(
                        'batch_loss: {:.4f}, batch_accuracy: {:.4f}, f1 score 0: {:.4f}, f1 score 1: {:.4f}'.format(
                            loss.item() / batch_total, batch_correct / batch_total, f1[0], f1[1]))

            avg_loss = total_loss / total
            accuracy = correct / total

            if val_data_loader is not None:
                val_loss, val_accuracy, val_f1_0, val_f1_1 = self.eval_model(val_data=val_data_loader, device=device)
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)
                self.history['val_f1_0'].append(val_f1_0)
                self.history['val_f1_1'].append(val_f1_1)
            else:
                val_loss, val_accuracy, val_f1_0, val_f1_1 = None, None, None, None

            self.history['train_loss'].append(avg_loss)
            self.history['train_accuracy'].append(accuracy)

            if store_weights_history:
                self.observer()  # <-- store weight history

            if store_weights_grad:
                self.grad_observer()  # <-- stores weights gradient

            if val_data_loader is not None:
                if val_loss < min_loss:
                    min_loss = val_loss
                    save_model = self.model.clone()
                    self.best_configuration = [save_model.to('cpu').state_dict(), val_loss, epoch]

            if display_log == 'epoch':
                epoch_range.set_description('epoch_loss: {:.4f}, epoch_accuracy: {:.4f}'.format(avg_loss, accuracy))

    def eval_model(self, val_data, device='cpu'):
        if device != 'cpu':
            gc.collect()
            torch.cuda.empty_cache()

        self.model.to(device)
        self.model.eval()

        eval_loss = 0
        total_correct = 0
        total = 0
        f1_0 = 0
        f1_1 = 0
        num_iter = 0

        with torch.no_grad():
            for val_x, val_y in tqdm(val_data):
                total += len(val_y)
                val_x = val_x.to(device)
                val_y = val_y.to(device)

                outputs = self.model(val_x).squeeze()

                batch_pred = torch.round(outputs)

                loss = self.loss_fn(outputs.float(), val_y.float(),
                                    hidden_rules_parameters=self.model.get_hidden_rules_params(),
                                    rules_weights_parameters=self.model.get_rules_weights_params())

                eval_loss += loss.item()

                batch_correct = (batch_pred == val_y).sum().item()
                total_correct += batch_correct

                f1 = f1_score(batch_pred.to('cpu').detach().numpy(),
                              val_y.to('cpu').detach().numpy(),
                              average=None,
                              zero_division=0,
                              labels=np.array([0, 1]))

                f1_0 += f1[0]
                f1_1 += f1[1]
                num_iter += 1

        eval_loss = eval_loss / total
        accuracy = total_correct / total
        f1_0 /= num_iter
        f1_1 /= num_iter

        return eval_loss, accuracy, f1_0, f1_1

    def predict(self, val_data, device, prob=True):
        if device != 'cpu':
            gc.collect()
            torch.cuda.empty_cache()

        self.model.to(device)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for val_x, val_y in tqdm(val_data):
                val_x = val_x.to(device)
                val_y = val_y.to(device)
                outputs = self.model(val_x).squeeze()
                if not prob:
                    outputs = torch.round(outputs)
                predictions.append(outputs.to('cpu'))
        predictions = torch.cat(predictions, axis=0)
        return predictions