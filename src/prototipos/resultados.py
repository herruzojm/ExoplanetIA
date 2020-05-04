import matplotlib.pyplot as plt
import os

class Resultados():
    def __init__(self, model_name):
        self.model_name = os.getcwd() + "\\img\\" + model_name
        self.train_losses = [] 
        self.validation_losses = []
        self.scores = []
        self.best_score = 0
        self.fpr = None
        self.tpr = None
        self.auc = None
        
    def add_train_loss(self, train_loss):
        self.train_losses.append(train_loss)
        
    def add_validation_loss(self, validation_loss):
        self.validation_losses.append(validation_loss)
        
    def add_score(self, score):
        self.scores.append(score)
        if self.is_best_score(score):
            self.best_score = score
    
    def is_best_score(self, score):
        return score > self.best_score
    
    def set_roc(self, fpr, tpr, auc):
        self.fpr = fpr
        self.tpr = tpr
        self.auc = auc
    
    def print_resultados(self, epoch):
        if len(self.validation_losses) > 0:
            print('Epoch: {}'.format(epoch), 
                  'Train loss {}'.format(self.train_losses[-1]),
                  'Validation loss {}'.format(self.validation_losses[-1]))
        else:
            print('Epoch: {}'.format(epoch), 
                  'Train loss {}'.format(self.train_losses[-1]))
        
    def print_best_score(self):
        print('Best score {}'.format(max(self.scores)))
        
    def plot_graphics(self, save = True):
        self.plot_losses(save)
        self.plot_scores(save)
        self.plot_roc(save)
        
    def plot_losses(self, save = True):
        plt.plot(self.train_losses, label = 'Training loss')
        plt.plot(self.validation_losses, label = 'Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')        
        plt.legend(frameon = False)
        if save:
            plt.savefig(self.model_name + '_losses.png')
        plt.show()
        
    def plot_scores(self, save = True):
        plt.plot(self.scores, label = 'Score')
        plt.xlabel('Epoch')
        plt.ylabel('Score')        
        plt.legend(frameon = False)
        if save:
            plt.savefig(self.model_name + '_scores.png')
        plt.show()
        
    def plot_roc(self, save = True):
        plt.plot([0, 1], [0, 1], color = 'red', lw = 1, linestyle = '--', alpha = 0.5)
        plt.plot(self.fpr, self.tpr, color = 'blue', label = 'ROC curve (auc = %0.2f)' % self.auc)        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(frameon = False)
        if save:
            plt.savefig(self.model_name + '_roc.png')
        plt.show()