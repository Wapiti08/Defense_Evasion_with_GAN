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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
global seed

seed = 0

class MalGAN():
    def __init__(self, blackbox='RF', same_train_data=1, data_path='./dataset/training_data/features_ran_part.pkl'):
        self.apifeature_dims = 688
        self.z_dims = 100
        self.generator_layers = [self.apifeature_dims+self.z_dims, 256, 512, 1024 , self.apifeature_dims]
        # self.substitute_detector_layers = [self.apifeature_dims, 512, 512, 512, 1]
        # self.substitute_detector_layers = [self.apifeature_dims, 512, 512, 1]
        self.substitute_detector_layers = [self.apifeature_dims, 512, 1]
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
        elif self.blackbox in ['DT']:
            blackbox_detector = DecisionTreeClassifier(random_state=seed)
        elif self.blackbox in ['RC']:
            blackbox_detector = RidgeClassifierCV()
        elif self.blackbox in ['SGD']:
            blackbox_detector = SGDClassifier(random_state=seed)
        elif self.blackbox in ['MLP']:
            blackbox_detector = MLPClassifier(hidden_layer_sizes=(50,), max_iter=50, alpha=1e-4,
                                                  solver='sgd', verbose=0, tol=1e-4, random_state=seed,
                                                  learning_rate_init=.1)

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
        data = pd.read_pickle(self.data_path)
        x_ran, y_ran, x_ben, y_ben = data[data['label']==1].iloc[:,:-1], data[data['label']==1].iloc[:,-1], data[data['label']==0].iloc[:,:-1], data[data['label']==0].iloc[:,-1]

        return (x_ran, y_ran), (x_ben, y_ben)
    
    
    def train(self, epochs, batch_size=32):

        # Load and Split the dataset
        (xmal, ymal), (xben, yben) = self.load_data()
        xtrain_mal, xtest_mal, ytrain_mal, ytest_mal = train_test_split(xmal.values, ymal.values, test_size=0.50)
        xtrain_ben, xtest_ben, ytrain_ben, ytest_ben = train_test_split(xben.values, yben.values, test_size=0.50)

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
    for classifier in ['MLP', 'SGD', 'RC', 'DT', 'SVM']: 
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
Test on layers:

    Test1:
        SVM: [D loss: 0.000110, acc.: 100.00%] [G loss: 0.000005]
        self.generator_layers = [self.apifeature_dims+self.z_dims, 256, 512, 1024 , self.apifeature_dims]
        self.substitute_detector_layers = [self.apifeature_dims, 512, 512, 512, 1]


    Test2:
        SVM: [D loss: 0.000189, acc.: 100.00%] [G loss: 0.000018]
        self.generator_layers = [self.apifeature_dims+self.z_dims, 256, 512, 1024 , self.apifeature_dims]
        self.substitute_detector_layers = [self.apifeature_dims, 512, 512, 1]


Test on classifiers:

Test1:
    self.generator_layers = [self.apifeature_dims+self.z_dims, 256, 512, 1024 , self.apifeature_dims]
    self.substitute_detector_layers = [self.apifeature_dims, 512, 512, 1]

    {'MLP': 4.650643700188084e-05, 'SGD': 0.00024402368228493287, 'RC': 0.0003104711322521325, 'DT': 0.0003598919756768737, 'SVM': 0.00019829026769002667}
    =====================
    {'MLP': 100.0, 'SGD': 100.0, 'RC': 100.0, 'DT': 100.0, 'SVM': 100.0}
    =====================
    {'MLP': 4.621968855644809e-06, 'SGD': 1.6674750895617763e-06, 'RC': 2.969875276903622e-05, 'DT': 3.61568781954702e-05, 'SVM': 9.004337698570453e-06}

Test2:
    self.generator_layers = [self.apifeature_dims+self.z_dims, 256, 512, 1024 , self.apifeature_dims]
    self.substitute_detector_layers = [self.apifeature_dims, 512, 1]

    {'MLP': 0.002002428260311717, 'SGD': 0.0006275568621276761, 'RC': 0.0008978054956969572, 'DT': 0.0006781554411645629, 'SVM': 0.0026881536255132232}
    =====================
    {'MLP': 100.0, 'SGD': 100.0, 'RC': 100.0, 'DT': 100.0, 'SVM': 100.0}
    =====================
    {'MLP': 8.395116310566664e-06, 'SGD': 5.485580459207995e-06, 'RC': 1.450031777494587e-05, 'DT': 5.976425200060476e-06, 'SVM': 1.1830474250018597e-05}

'''