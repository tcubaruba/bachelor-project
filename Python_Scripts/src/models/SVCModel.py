from src.models.Model import Model
from sklearn.svm import SVC


class SVCModel(Model):
    def define_model(self, kernel, c):
        if kernel == 'default':
            kernel = 'rbf'

        self.model = SVC(gamma='scale', probability=True, kernel=kernel, C=c, class_weight='balanced')
        self.model_name = 'svc'
        self.plot_name = 'Support Vector Classifier with \n kernel: ' + kernel + ' and C = ' + str(c)
        self.description = 'Support Vector Classifier with kernel: ' + kernel + ' and C = ' + str(c)