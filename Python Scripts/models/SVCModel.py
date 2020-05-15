from models.Model import Model
from sklearn.svm import SVC


class SVCModel(Model):
    def define_model(self, kernel):
        if kernel == 'default':
            kernel = 'rbf'

        self.model = SVC(gamma='scale', probability=True, kernel=kernel, C=0.99, class_weight='balanced')