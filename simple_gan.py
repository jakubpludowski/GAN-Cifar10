
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchsummary import summary
import numpy as np
from torch.utils.tensorboard import SummaryWriter

nr_disc_fil = 64
nr_gene_fil = 64
z_dim = 100 # noise dimension
device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_dim = 3
batch_size = 32
num_epochs = 100
real_label = 1
fake_label = 0

#weight initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Discriminator(nn.Module):
    def __init__(self, in_features=3):
        super().__init__()
        self.disc = nn.Sequential(
            #input 3 x 64x 64
            nn.Conv2d(in_features, nr_disc_fil, 4, 2, 1,bias = False),
            nn.LeakyReLU(0.2),
            # size 64 x 32 x 32
            nn.Conv2d(nr_disc_fil, nr_disc_fil*2, 4, 2, 1,bias = False),
            nn.BatchNorm2d(nr_disc_fil*2),
            nn.LeakyReLU(0.2),
            # size 128 x 16 x 16
            nn.Conv2d(nr_disc_fil*2, nr_disc_fil*4, 4, 2, 1,bias = False),
            nn.BatchNorm2d(nr_disc_fil*4),
            nn.LeakyReLU(0.2),
            # size 256 x 8 x 8 
            nn.Conv2d(nr_disc_fil*4, nr_disc_fil*8, 4, 2, 1,bias = False),
            nn.BatchNorm2d(nr_disc_fil*8),
            nn.LeakyReLU(0.2),
            # size 512 x 4 x 4
            nn.Conv2d(nr_disc_fil*8,1,4,1,0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x).view(-1, 1).squeeze(1)
    

class Generator(nn.Module):
    def __init__(self, img_dim = 3, z_dim = 100):
        super().__init__()
        self.gen = nn.Sequential(
            #input dim = 100
            nn.ConvTranspose2d(z_dim, nr_gene_fil*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nr_gene_fil*8),
            nn.ReLU(),
            #size 512 x 4 x 4
            nn.ConvTranspose2d(nr_gene_fil*8, nr_gene_fil*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nr_gene_fil*4),
            nn.ReLU(),
            #size 256 x 8 x 8
            nn.ConvTranspose2d(nr_gene_fil*4, nr_gene_fil*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nr_gene_fil*2),
            nn.ReLU(),
            #size 128 x 16 x 16
            nn.ConvTranspose2d(nr_gene_fil*2, nr_gene_fil, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nr_gene_fil),
            nn.ReLU(),
            #size 64 x 32 x 32
            nn.ConvTranspose2d(nr_gene_fil, img_dim, 4, 2, 1, bias=False),
            nn.Tanh()
            #output size 3 x 64 x 64
        )

    def forward(self, x):
        return self.gen(x)



#Creating 2 models
disc = Discriminator(image_dim).to(device)
disc.apply(weights_init)
gen = Generator(image_dim, z_dim).to(device)
gen.apply(weights_init)

opt_disc = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_gen =  optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()

#Creating noise to put in generator
fixed_noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)


#Getting data Cifar10
transforms = transforms.Compose([
                               transforms.Resize(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])


#Get data but only with one type of image ex. cat
dataset = datasets.CIFAR10(root="dataset/", transform=transforms, download=True)
dataset.targets = torch.tensor(dataset.targets)
idx = (dataset.targets == 3)
dataset.targets = dataset.targets[idx]
dataset.data = dataset.data[idx]

loader = DataLoader(dataset, batch_size, shuffle=True)

writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")

step = 0
g_loss = []
d_loss = []

for epoch in range(num_epochs):
    for i, data in enumerate(loader, 0):
        

        ### Train Discriminator: max dX + log(1 - dgZ)
        #real
        disc.zero_grad()
        real = data[0].to(device)
        batch_size = real.size(0)
        label = torch.full((batch_size,), real_label, device=device)
        label = label.to(torch.float32)
        output = disc(real)
        lossD_real = criterion(output, label)
        lossD_real.backward()
        dX = output.mean().item()

        #fake
        noise = torch.randn(batch_size, z_dim,1,1).to(device)
        fake = gen(noise)
        label.fill_(fake_label)
        output = disc(fake.detach())

        lossD_fake = criterion(output, label)
        lossD_fake.backward()
        dgZ1 = output.mean().item()

        lossD = lossD_real + lossD_fake
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        gen.zero_grad()
        label.fill_(real_label)
        output = disc(fake)
        lossG = criterion(output, label)
        lossG.backward()
        dgZ2 = output.mean().item()
        opt_gen.step()
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (epoch, num_epochs, i, len(loader), lossD.item(), lossG.item(), dX, dgZ1, dgZ2))
        

        if i == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 3, 64, 64)
                data = real.reshape(-1, 3, 64, 64)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Cifar10 Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Cifar10 Real Images", img_grid_real, global_step=step
                )
                step += 1
    torch.save(gen.state_dict(), 'weights/gen_epoch_%d.pth' % (epoch))
    torch.save(disc.state_dict(), 'weights/disc_epoch_%d.pth' % (epoch))