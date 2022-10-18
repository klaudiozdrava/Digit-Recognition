
import torch
import numpy
import torchaudio
import random
from torch import nn
from torch.utils.data import DataLoader,Dataset
from os.path import exists

INPUT_SIZE=1504 #Dataset row length
FIRST_LAYER=512
SECOND_LAYER=64
THIRD_LAYER=64
NUM_CLASSES=10


class AudioDataset(Dataset):

    def __init__(self):
        file = numpy.loadtxt('dataset.csv',delimiter=",",dtype=numpy.float32,skiprows=0)
        self.X=torch.from_numpy(file[:,1:])
        self.y=torch.from_numpy(file[:,0])
        self.samples=file.shape[0]

    def __getitem__(self, item):
        return self.X[item],self.y[item]

    def __len__(self):
        return self.samples



#Simple neural network
class FeedForwordNetwork(nn.Module):

    def __init__(self,input_size,first_layer,second_layer,num_classes):
        super(FeedForwordNetwork,self).__init__()

        self.dense_layers=nn.Sequential(
            nn.Linear(input_size, first_layer),
            nn.ReLU6(),
            nn.Linear(first_layer, second_layer),
            nn.ReLU6(),
            nn.Linear(second_layer,num_classes)
        )


    def forward(self,signal):
        out=self.dense_layers(signal)
        return out


#train model
def train_model(model,data,loss_fn,optimiser,devise,epochs):

    print(f"Devise that is used is {devise}")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        for i, (input_data, target) in enumerate(data):

            target = target.type(torch.LongTensor)
            input_data, target = input_data.to(devise), target.to(devise)

            prediction = model(input_data)#predict the imput
            loss = loss_fn(prediction, target)#calculate the loss and use derivative to optimise loss

            optimiser.zero_grad()  # Shoudl not store and sum up gradients for every changed weight
            loss.backward()
            optimiser.step()

        print(f"Loss {loss.item()}")


def createModel():
    EPOCHS = 340
    LEARNING_RATE = 0.001

    devise = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = AudioDataset()

    # DataLoader
    dataLoader = DataLoader(dataset=dataset, batch_size=100, shuffle=True)

    # build model
    neural_network = FeedForwordNetwork(INPUT_SIZE,FIRST_LAYER,SECOND_LAYER,NUM_CLASSES).to(devise)

    # Initialize loss function and optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(neural_network.parameters(),
                                 lr=LEARNING_RATE)

    train_model(neural_network, dataLoader, loss_fn, optimiser, devise, EPOCHS)

    torch.save(neural_network.state_dict(), "feedforward.pth")
    print("Model trained successfully")


if __name__ == '__main__':
    file_exists = exists('feedforward.pth')
    if file_exists:
        print('Model is already trained')
    else:
        print('Model must be trained.You need to wait..')
        createModel()


