import torch
import torchvision.transforms as transforms
from syn import FourLayerFNN
from data_loader import MyCustomDataset
from generate_data import painting
import  matplotlib.pyplot as plt

model = FourLayerFNN()
model.load_state_dict(torch.load('syn_model.pkl'))
model.eval()


# Ploting predictions
test_loader = torch.utils.data.DataLoader(
    MyCustomDataset('./data/dataset.csv',
                    transform=transforms.Compose([
                        transforms.ToTensor()])),
    batch_size=1, shuffle=False)

print(test_loader.dataset)
images  = test_loader.dataset.images

images = images.view(-1, 2)
outputs = model(images)
print(outputs)
_, prediction = torch.max(outputs.data, 1)
np_prediction = prediction.detach().numpy()
images = images.detach().numpy()

xc, yc, xr, yr = [],[],[],[]
for i in range(len(images)):
    if np_prediction[i] == 0:
        xc.append(images[i][0])
        yc.append(images[i][1])
    else:
        xr.append(images[i][0])
        yr.append(images[i][1])
#print(xc, yc)
print(len(xc))
#plt.scatter(xc,yc, color='red', label='Positive Point')
#plt.show()
#print(len(xyc), len(xyr))
painting(xc,yc, xr,yr,'Prediction')
