from models.Model import Model
from sklearn.linear_model import LogisticRegression


class LogRegModel(Model):
    def define_model(self, solver, penalty):
        if solver == 'default':
            solver = 'newton-cg'
        if penalty == '':
            penalty = 'l2'
        self.model = LogisticRegression(solver=solver, class_weight='balanced', multi_class='auto',
                                        verbose=False, warm_start=True, C=0.5, penalty=penalty)