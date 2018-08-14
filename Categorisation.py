from collections import Counter

import pandas

import keras
import random
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dense, Input, LSTM, Embedding, Conv1D, K, Concatenate, Lambda, \
    MaxPooling1D, Flatten
from keras.optimizers import Adam
from keras import Model
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import L1L2
from keras.utils import to_categorical, np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from numpy import unique
from scipy.stats import pearsonr
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder, normalize

from utils.utils import map_strings_to_intarrays
import csv
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

smiles_CB1 = []
smiles_CB2 = []
Ki_CB1 = []
Ki_CB2 = []
smiles = []
Ki = []
f = open('Mydata.csv')
csv_f = csv.reader(f)

for row in csv_f:
    # Cannabinoid receptor 1(CB1) ==> Cluster 131
    # Cannabinoid receptor 2(CB2) ==> Cluster 225
    if(row[4] == "Cluster 131" and row[5] != ""):
        smiles_CB1.append(row[3])
        Ki_CB1.append(row[5])
    if(row[4] == "Cluster 225" and row[5] != ""):
        smiles_CB2.append(row[3])
        Ki_CB2.append(row[5])

bla = []
unique_smiles_CB1 = unique(smiles_CB1)
compte = Counter(smiles_CB1)
for x in unique_smiles_CB1:
    if compte[x] > 1:
        bla.append(x)

tt = []
count = 0
somme = 0
index = []
ind_list = [i for i in range(len(smiles_CB1))]
for y in bla:
    for z in ind_list:
        if smiles_CB1[z] == y:
            somme += float(Ki_CB1[z])
            count += 1
            index.append(z)
    tt.append([y,somme/count])
    somme = 0
    count = 0

ind_refine = []
for x in ind_list:
    ind_refine.append(x)

for y in ind_refine:
    if y in index:
        ind_refine.remove(y)

smiles_CB1_bis = []
Ki_CB1_bis = []
for z in ind_refine:
    smiles_CB1_bis.append(smiles_CB1[z])
    Ki_CB1_bis.append(Ki_CB1[z])

for elem in tt:
    smiles_CB1_bis.append(elem[0])
    Ki_CB1_bis.append(elem[1])

Ki_CB1_categorize = []
for inhib in Ki_CB1_bis:
    if float(inhib) <= 1.4:
        #Tres fort
        Ki_CB1_categorize.append([1,0,0,0])
    if float(inhib) > 1.4 and float(inhib) <= 2.4:
        #Fort
        Ki_CB1_categorize.append([0,1,0,0])
    if float(inhib) > 2.4 and float(inhib) <= 3.4:
        #Moyen
        Ki_CB1_categorize.append([0,0,1,0])
    if float(inhib) > 3.4:
        #Faible
        Ki_CB1_categorize.append([0,0,0,1])

bla = []
unique_smiles_CB2 = unique(smiles_CB2)
compte = Counter(smiles_CB2)
for x in unique_smiles_CB2:
    if compte[x] > 1:
        bla.append(x)

tt = []
count = 0
somme = 0
index = []
ind_list = [i for i in range(len(smiles_CB2))]
for y in bla:
    for z in ind_list:
        if smiles_CB2[z] == y:
            somme += float(Ki_CB2[z])
            count += 1
            index.append(z)
    tt.append([y,somme/count])
    somme = 0
    count = 0

ind_refine = []
for x in ind_list:
    ind_refine.append(x)

for y in ind_refine:
    if y in index:
        ind_refine.remove(y)

smiles_CB2_bis = []
Ki_CB2_bis = []
for z in ind_refine:
    smiles_CB2_bis.append(smiles_CB2[z])
    Ki_CB2_bis.append(Ki_CB2[z])

for elem in tt:
    smiles_CB2_bis.append(elem[0])
    Ki_CB2_bis.append(elem[1])

Ki_CB2_categorize = []
for inhib in Ki_CB2_bis:
    if float(inhib) <= 1.4:
        #Tres fort
        Ki_CB2_categorize.append([1,0,0,0])
    if float(inhib) > 1.4 and float(inhib) <= 2.4:
        #Fort
        Ki_CB2_categorize.append([0,1,0,0])
    if float(inhib) > 2.4 and float(inhib) <= 3.4:
        #Moyen
        Ki_CB2_categorize.append([0,0,1,0])
    if float(inhib) > 3.4:
        #Faible
        Ki_CB2_categorize.append([0,0,0,1])

periodic_elements = ['H', 'Li', 'B', 'C', 'N', 'O', 'F', 'Na', 'Al', 'Si', 'P', 'S', 'Cl',
                       'K', 'Ca', 'V', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br',
                       'Nb', 'Mo', 'Tc', 'Ru', 'Pd', 'Ag', 'Sn', 'Sb', 'Te', 'I', 'Gd', 'W',
                       'Re', 'Os', 'Pt', 'Au', 'Hg', 'Bi']

smilesAlphabet = list('#%)(+*-/.1032547698:=@[]\\conslr') + periodic_elements + ['se']

def smiles_to_arrays(smiles_list, L=None):
    L = L if L else max([len(x) for x in smiles_list])
    smiles_dict = dict(zip(smilesAlphabet, 1 + np.arange(len(smilesAlphabet))))
    return pad_sequences(map_strings_to_intarrays(smiles_list, smiles_dict), L, padding='post')

class Bioactivity_net(object):
    ENCODING_NAME = 'encoding_layer'

    def __init__(self, max_length=None):
        self.max_length = max_length
        self.n_epochs_patience = 10
        self.n_epochs = 1000
        self.verbose = 1
        self.model = None

    def network_initializer(self, max_timesteps=None):
        raise NotImplementedError("Abstract")

    def fit(self, x, y, x_valid=None, y_valid=None, validation_split=None, batch_size=32):
        if x_valid is None or y_valid is None:
            validation_split = validation_split if validation_split else 1/5.
        x_new = smiles_to_arrays(x, self.max_length)
        K.set_learning_phase(1)

        early_stopping = EarlyStopping(monitor='val_loss', patience=self.n_epochs_patience, verbose=0, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='loss', patience=self.n_epochs_patience/2, epsilon=1e-3)


        callbacks = [early_stopping, reduce_lr]

        if x_valid is not None and y_valid is not None:
            x_valid_new = smiles_to_arrays(x_valid, self.max_length)
            self.history = self.model.fit(x_new, y,
                                          epochs=self.n_epochs,
                                          batch_size=batch_size,
                                          validation_data=(x_valid_new, y_valid),
                                          callbacks=callbacks,
                                          verbose=1)
        else:
            self.history = self.model.fit(x_new, y,
                                          epochs=self.n_epochs,
                                          batch_size=batch_size,
                                          validation_split=validation_split,
                                          callbacks=callbacks,
                                          verbose=1)
        self.val_loss = early_stopping.best
        K.set_learning_phase(0)
        return self

    def predict(self, x):
        x_new = smiles_to_arrays(x, self.max_length)
        return self.model.predict(x_new)

class BioactivityLSTM(Bioactivity_net):
    def __init__(self, vocab_size=76, embedding_size=20, layer_sizes=[128],
                 lr=0.001, l2=0.001,dropout=0.5):
        super(BioactivityLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.layer_sizes = layer_sizes
        self.optimizer = Adam(lr)
        self.dropout = dropout
        self.l2 = l2

        self.network_initializer()

    def network_initializer(self, max_timesteps=None):
        ### build encoder
        enc_input = Input(shape=(None, ), dtype='int32', name='input')
        enc_embedding = Embedding(self.vocab_size, self.embedding_size, mask_zero=True, name='embedding')(enc_input)

        # stacking encoding LSTMs
        hidden_states = []
        enc_layer = enc_embedding
        for i, layer_size in enumerate(self.layer_sizes):
            return_sequences = (i != len(self.layer_sizes) - 1)
            enc_layer, hidden_state, cell_state = LSTM(layer_size, return_sequences=return_sequences,
                                                       return_state=True, name='lstm_%d' % (i+1))(enc_layer)
            hidden_states += [hidden_state, cell_state]

        # concatenating LSTMs' states and normalizing their norms
        enc_output = Concatenate()(hidden_states)
        enc_output = Lambda(lambda x: K.l2_normalize(x, axis=1), name=BioactivityLSTM.ENCODING_NAME)(enc_output)

        ### output layer
        out_layer = Dense(4, kernel_regularizer=L1L2(l2=self.l2), activation='softmax')(enc_output)
        self.model = Model(inputs=enc_input, outputs=out_layer)
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
        # self.model.summary()


class BioactivityCNN(Bioactivity_net):
    def __init__(self, max_length, vocab_size=20, embedding_size=20, kernel_sizes=[128], l_pooling=1,
                 lr=0.001, l2=0.001, dropout=0.5):
        super(BioactivityCNN, self).__init__(max_length)
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.kernel_sizes = kernel_sizes
        self.optimizer = Adam(lr)
        self.dropout = dropout
        self.L_pooling = l_pooling
        self.l2 = l2

        self.network_initializer()

    def network_initializer(self, max_timesteps=None):
        ### build encoder
        enc_input = Input(shape=(self.max_length, ), dtype='int32', name='input')
        enc_embedding = Embedding(self.vocab_size, self.embedding_size, name='embedding')(enc_input)

        # stacking encoding LSTMs
        convos = []
        for i, hs in enumerate(self.kernel_sizes):
            if hs > 0:
                convo_layer = Conv1D(
                    filters=hs,
                    kernel_size=i + 1,
                    padding='same')(enc_embedding)
                if self.L_pooling > 1:
                    convo_layer = MaxPooling1D(pool_size=self.L_pooling)(convo_layer)
                convo_layer = Flatten()(convo_layer)
                convos.append(convo_layer)
        enc_output = Concatenate()(convos) if len(convos) > 1 else convos[0]
        enc_output = Lambda(lambda x: K.l2_normalize(x, axis=1), name=BioactivityLSTM.ENCODING_NAME)(enc_output)

        ### output layer

        out_layer = Dense(4, kernel_regularizer=L1L2(l2=self.l2), activation='softmax')(enc_output)
        self.model = Model(inputs=enc_input, outputs=out_layer)
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
        # self.model.summary()


if __name__ == '__main__':

    np.random.seed(9)

    def plot_confusion_matrix(cm,
                              target_names,
                              title='Confusion matrix',
                              cmap=None,
                              normalize=True):
        """
        given a sklearn confusion matrix (cm), make a nice plot

        Arguments
        ---------
        cm:           confusion matrix from sklearn.metrics.confusion_matrix

        target_names: given classification classes such as [0, 1, 2]
                      the class names, for example: ['high', 'medium', 'low']

        title:        the text to display at the top of the matrix

        cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                      see http://matplotlib.org/examples/color/colormaps_reference.html
                      plt.get_cmap('jet') or plt.cm.Blues

        normalize:    If False, plot the raw numbers
                      If True, plot the proportions

        Usage
        -----
        plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                                  # sklearn.metrics.confusion_matrix
                              normalize    = True,                # show proportions
                              target_names = y_labels_vals,       # list of names of the classes
                              title        = best_estimator_name) # title of graph

        Citiation
        ---------
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        """
        import matplotlib.pyplot as plt
        import numpy as np
        import itertools

        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.show()
    """
    kf = KFold(5)
    oos_y = []
    oos_pred = []
    fold = 0

    x = np.asarray(smiles_CB1)
    y = np.asarray(Ki_CB1_categorize)
    ind_list = [i for i in range(len(Ki_CB1))]
    random.shuffle(ind_list)
    x_shuffle = x[ind_list,]
    y_shuffle = y[ind_list,]

    for train, test in kf.split(x_shuffle):
        fold+=1
        print("Fold #{}".format(fold))

        x_train = x_shuffle[train]
        y_train = y_shuffle[train]
        x_test = x_shuffle[test]
        y_test = y_shuffle[test]
        rgr = BioactivityLSTM(vocab_size=76, embedding_size=75, layer_sizes=[128], lr=0.01, l2=0.01)
        rgr.fit(x_train, y_train, x_valid=x_test,y_valid=y_test, batch_size=32)

        y_pred_prob = rgr.predict(x_test)
        y_pred_prob = np.argmax(y_pred_prob, axis=1)
        y_test2 = np.argmax(y_test, axis=1)

        oos_y.append(y_test2)
        oos_pred.append(y_pred_prob)

        score = accuracy_score(y_pred_prob,y_test2)
        print("Fold score (Accuracy): {}".format(score))

oos_y = np.concatenate(oos_y)
oos_pred = np.concatenate(oos_pred)
score = accuracy_score(oos_pred,oos_y)
print("Final, out of sample score (Accuracy): {}".format(score))

cm = confusion_matrix(oos_y, oos_pred)
plot_confusion_matrix(cm,normalize=False,
                          target_names=['Very Strong', 'Strong','Medium', 'Low'],
                          title="Confusion Matrix")

exit()

    """

    rgr = BioactivityLSTM(vocab_size=76,embedding_size = 75, layer_sizes = [128], lr = 0.01 , l2 = 0.01)
    #rgr = BioactivityCNN(vocab_size=76, max_length=300,embedding_size = 75, kernel_sizes= [128], lr = 0.001 , l2 = 0.01)

    x = np.asarray(smiles_CB2_bis)
    y = np.asarray(Ki_CB2_categorize)

    ind_list = [i for i in range(len(x))]
    random.shuffle(ind_list)
    x_shuffle = x[ind_list,]
    y_shuffle = y[ind_list,]

    x_train, x_test, y_train, y_test = train_test_split(x_shuffle, y_shuffle, test_size=0.2)
    rgr.fit(x_train, y_train,validation_split= 0.2,batch_size=8)

    y_pred_prob = rgr.predict(x_test)
    y_pred_prob = np.argmax(y_pred_prob,axis = 1)
    y_test2 = np.argmax(y_test, axis = 1)

    print(accuracy_score(y_test2, y_pred_prob))

    cm = confusion_matrix(y_test2, y_pred_prob)
    plot_confusion_matrix(cm,
                          normalize=False,
                          target_names=['Very Strong', 'Strong','Medium', 'Low'],
                          title="Confusion Matrix")


    """predictions = []
    for sm in x:
        y = rgr.predict([sm])
        y__ = np.argmax(y, axis=1)
        predictions.append(y__)
    Ki__ = []
    for i in Ki_CB2_categorize:
        if i == [1,0,0,0]:
            Ki__.append(0)
        if i == [0,1,0,0]:
            Ki__.append(1)
        if i == [0,0,1,0]:
            Ki__.append(2)
        if i == [0,0,0,1]:
            Ki__.append(3)

    with open('Cat_4_classes_predictions_CB2.tsv', 'w') as f:
        f.write("SMILES\tCatégorie_réelle\tCatégorie_prédite\n")
        lis = [smiles_CB2, Ki__, predictions]
        for x in zip(*lis):
            f.write("{0}\t{1}\t{2}\n".format(*x))"""