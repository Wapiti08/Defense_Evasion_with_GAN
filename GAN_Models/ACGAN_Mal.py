from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Maximum, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifierCV, SGDClassifier
from sklearn.neural_network import MLPClassifier
from Ensemble_Classifiers import Ensemble_Classifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
global seed

seed = 0

class MalGAN():
    def __init__(self, blackbox='RF', same_train_data=1, data_path='./dataset/Malware/ClaMP_Integrated-5184.csv'):
        self.apifeature_dims = 68
        self.z_dims = 20
        self.generator_layers = [self.apifeature_dims+self.z_dims, 256, 512, 1024 , self.apifeature_dims]
        self.substitute_detector_layers = [self.apifeature_dims, 512, 512, 512, 1]

        # self.generator_layers = [self.apifeature_dims+self.z_dims, 32, 64, 128 , self.apifeature_dims]
        # self.substitute_detector_layers = [self.apifeature_dims, 64, 64, 64, 1]
        # self.substitute_detector_layers = [self.apifeature_dims, 512, 512, 1]

        self.blackbox = blackbox       
        optimizer = Adam(lr=0.0002, beta_1=0.5)
        self.data_path = data_path

        # Build and Train blackbox_detector
        self.blackbox_detector = self.build_blackbox_detector()

        # Build and compile the substitute_detector
        self.substitute_detector = self.build_substitute_detector()
        self.substitute_detector.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes malware and noise as input and generates adversarial malware examples
        example = Input(shape=(self.apifeature_dims,))
        noise = Input(shape=(self.z_dims,))
        input = [example, noise]
        malware_examples = self.generator(input)

        # The discriminator takes generated images as input and determines validity
        validity = self.substitute_detector(malware_examples)

        # The combined model  (stacked generator and substitute_detector)
        # Trains the generator to fool the discriminator
        self.combined = Model(input, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        
        # For the combined model we will only train the generator
        self.substitute_detector.trainable = False
    

    def build_blackbox_detector(self):
        if self.blackbox in ['SVM']:
            blackbox_detector = SVC(kernel = 'linear')
        elif self.blackbox in ['MLP']:
            blackbox_detector = MLPClassifier(hidden_layer_sizes=(50,), max_iter=50, alpha=1e-4,
                                                  solver='sgd', verbose=0, tol=1e-4, random_state=seed,
                                                  learning_rate_init=.1)
        elif self.blackbox in ['Ensem']:
            blackbox_detector = Ensemble_Classifier()
        return blackbox_detector

    def build_generator(self):

        example = Input(shape=(self.apifeature_dims,))
        noise = Input(shape=(self.z_dims,))
        x = Concatenate(axis=1)([example, noise])
        for dim in self.generator_layers[1:]:
            x = Dense(dim)(x)
        x = Activation(activation='tanh')(x)
        x = Maximum()([example, x])
        generator = Model([example, noise], x, name='generator')
        generator.summary()
        return generator

    def build_substitute_detector(self):

        input = Input(shape=(self.substitute_detector_layers[0],))
        x = input
        for dim in self.substitute_detector_layers[1:]:
            x = Dense(dim)(x)
        x = Activation(activation='sigmoid')(x)
        substitute_detector = Model(input, x, name='substitute_detector')
        substitute_detector.summary()
        return substitute_detector

    def load_data(self):
        data = pd.read_csv(self.data_path)
        data.drop('packer_type', axis=1, inplace=True)
        x_ran, y_ran, x_ben, y_ben = data[data['class']==1].iloc[:,:-1], data[data['class']==1].iloc[:,-1], data[data['class']==0].iloc[:,:-1], data[data['class']==0].iloc[:,-1]

        return (x_ran, y_ran), (x_ben, y_ben)
    
    
    def train(self, epochs, batch_size=32):

        # Load and Split the dataset
        (xmal, ymal), (xben, yben) = self.load_data()
        # get the numpy array
        xmal = xmal.values
        xben = xben.values
        ymal = ymal.values
        yben = yben.values
        # build the normalizer transformer
        transformer = Normalizer().fit(xmal)
        # transform training x values
        xmal = transformer.transform(xmal)
        xben = transformer.transform(xben)

        xtrain_mal, xtest_mal, ytrain_mal, ytest_mal = train_test_split(xmal, ymal, test_size=0.50)
        xtrain_ben, xtest_ben, ytrain_ben, ytest_ben = train_test_split(xben, yben, test_size=0.50)

        bl_xtrain_mal, bl_ytrain_mal, bl_xtrain_ben, bl_ytrain_ben = xtrain_mal, ytrain_mal, xtrain_ben, ytrain_ben

        
        self.blackbox_detector.fit(np.concatenate([xmal, xben]), np.concatenate([ymal, yben]))

        ytrain_ben_blackbox = self.blackbox_detector.predict(bl_xtrain_ben)
        
        Original_Train_TPR = self.blackbox_detector.score(bl_xtrain_mal, bl_ytrain_mal)
        
        Original_Test_TPR = self.blackbox_detector.score(xtest_mal, ytest_mal)
        Train_TPR, Test_TPR = [Original_Train_TPR], [Original_Test_TPR]


        for epoch in range(epochs):

            for step in range(xtrain_mal.shape[0] // batch_size):
                # ---------------------
                #  Train substitute_detector
                # ---------------------

                # Select a random batch of malware examples
                idx_mal = np.random.randint(0, xtrain_mal.shape[0], batch_size)
  
                xmal_batch = xtrain_mal[idx_mal]
                
                noise = np.random.normal(0, 1, (batch_size, self.z_dims))
                
                idx_ben = np.random.randint(0, xmal_batch.shape[0], batch_size)
                
                xben_batch = xtrain_ben[idx_ben]
                yben_batch = ytrain_ben_blackbox[idx_ben]

                # Generate a batch of new malware examples
                gen_examples = self.generator.predict([xmal_batch, noise])
                ymal_batch = self.blackbox_detector.predict(np.ones(gen_examples.shape)*(gen_examples > 0.5))

                # Train the substitute_detector

                d_loss_real = self.substitute_detector.train_on_batch(gen_examples, ymal_batch)
                d_loss_fake = self.substitute_detector.train_on_batch(xben_batch, yben_batch)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                idx = np.random.randint(0, xtrain_mal.shape[0], batch_size)
                xmal_batch = xtrain_mal[idx]
                noise = np.random.uniform(0, 1, (batch_size, self.z_dims))

                # Train the generator
                g_loss = self.combined.train_on_batch([xmal_batch, noise], np.zeros((batch_size, 1)))

            # Compute Train TPR
            noise = np.random.uniform(0, 1, (xtrain_mal.shape[0], self.z_dims))
            gen_examples = self.generator.predict([xtrain_mal, noise])
            TPR = self.blackbox_detector.score(np.ones(gen_examples.shape) * (gen_examples > 0.5), ytrain_mal)
            Train_TPR.append(TPR)

            # Compute Test TPR
            noise = np.random.uniform(0, 1, (xtest_mal.shape[0], self.z_dims))
            gen_examples = self.generator.predict([xtest_mal, noise])
            TPR = self.blackbox_detector.score(np.ones(gen_examples.shape) * (gen_examples > 0.5), ytest_mal)
            Test_TPR.append(TPR)

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            
            if int(epoch) == int(epochs-1):
                return  d_loss[0], 100*d_loss[1], g_loss
        

if __name__ == '__main__':
    # create the dict to save the D loss, acc and G loss for different classifiers
    D_loss_dict, Acc_dict, G_loss_dict = {}, {}, {}

    # load the classifier
    for classifier in [ 'SVM', 'MLP', 'Ensem']: 
        print('[+] \nTraining the model with {} classifier\n'.format(classifier))
        malgan = MalGAN(blackbox=classifier)
        d_loss, acc, g_loss = malgan.train(epochs=50, batch_size=32)

        D_loss_dict[classifier] = d_loss
        Acc_dict[classifier] = acc 
        G_loss_dict[classifier] = g_loss


    print('=====================')
    print(D_loss_dict)
    print('=====================')
    print(Acc_dict)
    print('=====================')
    print(G_loss_dict)
    
'''



Test with different units:

Test1:
    self.generator_layers = [self.apifeature_dims+self.z_dims, 256, 512, 1024 , self.apifeature_dims]
    self.substitute_detector_layers = [self.apifeature_dims, 512, 512, 1]

    {'SVM': 0.06019804626703262, 'MLP': 0.1914462000131607, 'Ensem': 0.227969191968441}
    =====================
    {'SVM': 100.0, 'MLP': 89.0625, 'Ensem': 87.5}
    =====================
    {'SVM': 0.0018579576862975955, 'MLP': 1.2675385505644954e-06, 'Ensem': 0.0003518983139656484}

Test2:
    self.generator_layers = [self.apifeature_dims+self.z_dims, 32, 64, 128 , self.apifeature_dims]
    self.substitute_detector_layers = [self.apifeature_dims, 64, 64, 64, 1]

    {'SVM': 0.09068786725401878, 'MLP': 0.3209685608744621, 'Ensem': 0.22809869050979614}
    =====================
    {'SVM': 95.3125, 'MLP': 84.375, 'Ensem': 87.5}
    =====================
    {'SVM': 9.215701766152051e-07, 'MLP': 2.0628694983315654e-05, 'Ensem': 0.014765027910470963}

Test3:
    self.generator_layers = [self.apifeature_dims+self.z_dims, 256, 512, 1024 , self.apifeature_dims]
    self.substitute_detector_layers = [self.apifeature_dims, 512, 512, 512, 1]

    {'SVM': 0.058258047327399254, 'MLP': 0.17800971865653992, 'Ensem': 0.13576413318514824}
    =====================
    {'SVM': 98.4375, 'MLP': 93.75, 'Ensem': 95.3125}
    =====================
    {'SVM': 2.630974336170766e-07, 'MLP': 4.509088284976315e-06, 'Ensem': 2.16914554584946e-06}

'''