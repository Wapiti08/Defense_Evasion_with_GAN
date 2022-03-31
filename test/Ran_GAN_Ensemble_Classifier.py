from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Maximum, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
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
        self.substitute_detector_layers = [self.apifeature_dims, 512, 512, 1]
        # self.substitute_detector_layers = [self.apifeature_dims, 512, 1]
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
        if self.blackbox in ['Bag']:
            blackbox_detector = BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=seed)
        elif self.blackbox in ['RF']:
            blackbox_detector = RandomForestClassifier(max_depth=2, random_state=seed)
        elif self.blackbox in ['AdaBoost']:
            blackbox_detector = AdaBoostClassifier(n_estimators=100, random_state=seed)
        elif self.blackbox in ['GB']:
            blackbox_detector = GradientBoostingClassifier(n_estimators=100, random_state=seed)

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
        xtrain_mal, xtest_mal, ytrain_mal, ytest_mal = train_test_split(xmal.values, ymal.values, test_size=0.20)
        xtrain_ben, xtest_ben, ytrain_ben, ytest_ben = train_test_split(xben.values, yben.values, test_size=0.20)

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
    for classifier in ['Bag', 'RF', 'AdaBoost', 'GB']: 
        print('\n[+] Training the model with {} classifier\n'.format(classifier))
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
Test on ensemble classifiers:

Test1:
    =====================
    {'Bag': 0.0009752021333042649, 'RF': 0.0005696318921764032, 'AdaBoost': 0.00045098867940396303, 'GB': 0.0005312355951900827}
    =====================
    {'Bag': 100.0, 'RF': 100.0, 'AdaBoost': 100.0, 'GB': 100.0}
    =====================
    {'Bag': 6.450652108469512e-06, 'RF': 1.4073579222895205e-05, 'AdaBoost': 1.1061413715651724e-05, 'GB': 2.3088181478669867e-05}
    
    self.generator_layers = [self.apifeature_dims+self.z_dims, 256, 512, 1024 , self.apifeature_dims]
    self.substitute_detector_layers = [self.apifeature_dims, 512, 1]


Test2:
    =====================
    {'Bag': 8.22592260192323e-05, 'RF': 0.00233113380636496, 'AdaBoost': 0.0004080970820723451, 'GB': 0.0006035762362444075}
    =====================
    {'Bag': 100.0, 'RF': 100.0, 'AdaBoost': 100.0, 'GB': 100.0}
    =====================
    {'Bag': 3.5508924156602006e-06, 'RF': 8.965355846157763e-06, 'AdaBoost': 9.755195605976041e-06, 'GB': 1.797146614990197e-05}
    
    self.generator_layers = [self.apifeature_dims+self.z_dims, 256, 512, 1024 , self.apifeature_dims]
    self.substitute_detector_layers = [self.apifeature_dims, 512, 512, 1]

'''