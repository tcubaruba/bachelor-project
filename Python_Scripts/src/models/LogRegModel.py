from src.models.Model import Model
from sklearn.linear_model import LogisticRegression


class LogRegModel(Model):
    def define_model(self, solver, c):
        if solver == 'default':
            solver = 'newton-cg'
        self.model = LogisticRegression(solver=solver, class_weight='balanced', multi_class='auto',
                                        verbose=False, C=c)
        self.model_name = 'lr'
        self.plot_name = 'Logistic Regression with \n solver: ' + solver + ' and C = ' + str(c)
        self.description = 'Logistic Regression with solver: ' + solver + ' and C = ' + str(c)