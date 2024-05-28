import os
from itertools import cycle
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
import torch
from torch import nn
import torch.optim as optim
import torchvision.utils as vutils

class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Input: Latent vector (latent_dim)
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # Upsample to (512, 8, 8)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # Upsample to (256, 16, 16)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # Upsample to (128, 32, 32)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Output: Image (img_channels, 64, 64)
            nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # Normalize the output to [-1, 1]
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, img_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Input: Image (img_channels, 64, 64)
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Downsample to (64, 32, 32)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Downsample to (128, 16, 16)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Downsample to (256, 8, 8)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Flatten and output decision
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1),
            nn.Sigmoid()  # Output in [0, 1], representing fake vs. real
        )

    def forward(self, x):
        return self.model(x)


def save_generator_output(generator, fixed_noise, epoch, batch_size, output_dir='output_generator', img_size=(64, 64)):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Tell PyTorch we are not accumulating gradients
    with torch.no_grad():
        # Generate batch of images
        fake_images = generator(fixed_noise).detach().cpu()
    
    # Save the generated images
    filename = os.path.join(output_dir, f'epoch_{epoch}.png')
    vutils.save_image(fake_images, filename, normalize=True, nrow=int(batch_size**0.5))
    
    print(f'Saved generator output at: {filename}')




transform = Compose([
    Resize((256, 256)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



monet_dataset = datasets.ImageFolder('./training-data/monet_jpg', transform=transform)
#regular_image_dataset = MonetDataset(images_dir='training-data/photo_jpg', transform=transform)

dataloader_monet = DataLoader(monet_dataset, batch_size=4, shuffle=True)
#dataloader_normal = DataLoader(regular_image_dataset, batch_size=4, shuffle=True)


# Hyperparameters
lr = 0.0002
beta1 = 0.5  # Hyperparameter for Adam optimizers
epochs = 10
latent_dim = 100
img_channels = 3  # Assuming RGB images

# Initialize models
generator = Generator(latent_dim, img_channels)
discriminator = Discriminator(img_channels)

# Loss function
criterion = nn.BCELoss()

# Optimizers
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

for epoch in range(epochs):
    for i, data in enumerate(dataloader_monet, 0):

        # ===== Train Discriminator =====
        discriminator.zero_grad()
        
        # Train with real images
        real_images = data[0]
        label = torch.full((len(real_images),), 1, dtype=torch.float)  # Real labels = 1
        print(real_images)
        
        output = discriminator(real_images).view(-1)

        lossD_real = criterion(output, label)
        lossD_real.backward()
        
        # Train with fake images
        noise = torch.randn(len(real_images), latent_dim, 1, 1)
        fake_images = generator(noise)
        label.fill_(0)  # Fake labels = 0
        output = discriminator(fake_images.detach()).view(-1)
        lossD_fake = criterion(output, label)
        lossD_fake.backward()
        
        # Update discriminator
        optimizerD.step()

        # ===== Train Generator =====
        generator.zero_grad()
        label.fill_(1)  # Generator wants discriminator to think the fake images are real
        output = discriminator(fake_images).view(-1)
        lossG = criterion(output, label)
        lossG.backward()
        
        # Update generator
        optimizerG.step()
        
        # Logging (optional, to keep track of progress)
        if i % 50 == 0:
            print(f'Epoch [{epoch}/{epochs}] Batch {i}/{len(dataloader_monet)} Loss D: {lossD_real + lossD_fake}, Loss G: {lossG}')
            save_generator_output(generator, noise, epoch, batch_size=64)

print("finished training")


generator_save_path = 'models/generator_state_dict.pth'
discriminator_save_path = 'models/discriminator_state_dict.pth'

torch.save(generator.state_dict(), generator_save_path)
torch.save(discriminator.state_dict(), discriminator_save_path)

print("saved model")