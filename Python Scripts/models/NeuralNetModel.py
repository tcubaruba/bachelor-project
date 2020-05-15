from models.Model import Model
from sklearn.neural_network import MLPClassifier


class NeuralNetModel(Model):
    def define_model(self, solver, activation, n_nodes):
        if solver == 'default':
            solver = 'adam'
        if activation == 'default':
            activation = 'logistic'
        if n_nodes == '()':
            n_nodes = '(100,100)'
        n_nodes = eval(n_nodes)
        self.model = MLPClassifier(solver=solver, early_stopping=True, verbose=False, learning_rate='adaptive',
                                   activation=activation,
                                   hidden_layer_sizes=n_nodes)