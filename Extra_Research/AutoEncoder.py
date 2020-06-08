'''
for AE, this is the process to compress input layer by layer and decode the compressed data
layer by layer
'''



'''
Critical part of autoencoder
'''
class AutoEncoder():
    def __init__():

        #define the encoder
        self.encoder=nn.Sequential(
            #from input to first layer
            nn.Linear(28*28,128),
            #the incentive function
            nn.Tanh(),
            nn.Linear(128,64),
            nn.Tanh(),
            nn.Linear(64,12),
            nn.Tanh(),
            nn.Linear(12,3),
        )
        self.decoder=nn.Sequential(
            nn.Linear(3,12),
            nn.Tanh(),
            nn.Linear(12,64),
            nn.Tanh(),
            nn.Linear(64,128),
            nn.Tanh(),
            nn.Linear(128,28*28),
            #compress the output into the value between 0~1
            nn.Sigmoid(),
        )
    #combine the encoder and decoder
    def forward(self,x):
        encoded=self.encoder(x)
        decoded=self.decoder(encoded)
        return decoded, encoded
    
#optimize the model
autoencoder=AutoEncoder()
optimizer=torch.optim.Adam(autoencoder.parameters(),lr=LR)
loss_func=nn.MSELoss()
