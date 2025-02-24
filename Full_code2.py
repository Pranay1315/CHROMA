import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

# Paths (These paths are specific to your local machine)
path_train = "./Dataset/train_black/"
path_target = "./Dataset/train_color/"  
checkpoint_folder = "./Dataset/checkpoints/"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
NUM_EPOCHS = 20

# Ensure checkpoint directory exists
os.makedirs(checkpoint_folder, exist_ok=True)

# Importing the dataset
class ImageColorizationDataset(Dataset):
    def __init__(self, train_dir, target_dir, transform=None):
        self.train_dir = train_dir
        self.target_dir = target_dir
        self.transform = transform
        self.train_images = os.listdir(train_dir)

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.train_dir, self.train_images[idx])
        target_name = os.path.join(self.target_dir, self.train_images[idx])
        input_img = cv2.imread(img_name)
        target_img = cv2.imread(target_name)

        if input_img is None or target_img is None:
            print(f"Error loading image: {img_name} or {target_name}")
            return None, None

        if self.transform:
            input_img, target_img = self.transform(input_img, target_img)

        # Ensure images are not None before transposing
        if input_img is not None and target_img is not None:
            return input_img.astype(np.float32).transpose(2, 0, 1), target_img.astype(np.float32).transpose(2, 0, 1)
        else:
            return None, None

# Data augmentation
def random_jitter(input_img, tar_img):
    # Resize to a fixed size BEFORE cropping
    input_img = cv2.resize(input_img, (286, 286))
    tar_img = cv2.resize(tar_img, (286, 286))

    # Generate random coordinates for cropping
    crop_x = np.random.randint(0, 286 - 256)
    crop_y = np.random.randint(0, 286 - 256)

    # Crop both images using the same coordinates
    input_img = input_img[crop_y:crop_y + 256, crop_x:crop_x + 256]
    tar_img = tar_img[crop_y:crop_y + 256, crop_x:crop_x + 256]

    if np.random.rand() > 0.5:
        input_img = cv2.flip(input_img, 1)
        tar_img = cv2.flip(tar_img, 1)

    return input_img, tar_img

def load_image(filename):
    input_img = cv2.imread(os.path.join(path_train, filename))
    tar_img = cv2.imread(os.path.join(path_target, filename))
    return input_img, tar_img

class Transforms:
    def __call__(self, inp, tar):
        inp, tar = random_jitter(inp, tar)
        inp = (inp / 255.0) * 2 - 1
        tar = (tar / 255.0) * 2 - 1
        return inp, tar

# Define the Generator model
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x

class UNetSkipConnection(nn.Module):
    def __init__(self, in_channels, out_channels, submodule, down=True, use_dropout=False):
        super(UNetSkipConnection, self).__init__()
        self.down = Block(in_channels, out_channels, down=down, act="leaky", use_dropout=use_dropout)
        self.up = Block(out_channels*2, in_channels, down=False, act="relu", use_dropout=use_dropout)  # Adjusted in_channels for upsampling
        self.submodule = submodule
        self.use_dropout = use_dropout

    def forward(self, x):
        down_x = self.down(x)
        submodule_x = self.submodule(down_x)
        up_x = self.up(torch.cat([down_x, submodule_x], dim=1))
        return up_x

class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(Generator, self).__init__()
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(features, features * 2, down=True, act="leaky")
        self.down2 = Block(features * 2, features * 4, down=True, act="leaky")
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky")
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky")
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky")
        self.down6 = Block(features * 8, features * 8, down=True, act="leaky")
        self.bottleneck = Block(features * 8, features * 8, down=True, act="leaky")

        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(features * 16, features * 8, down=False, act="relu", use_dropout=True)
        self.up3 = Block(features * 16, features * 8, down=False, act="relu", use_dropout=True)
        self.up4 = Block(features * 16, features * 8, down=False, act="relu")
        self.up5 = Block(features * 16, features * 4, down=False, act="relu")
        self.up6 = Block(features * 8, features * 2, down=False, act="relu")
        self.up7 = Block(features * 4, features, down=False, act="relu")
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.first(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], dim=1))
        up3 = self.up3(torch.cat([up2, d6], dim=1))
        up4 = self.up4(torch.cat([up3, d5], dim=1))
        up5 = self.up5(torch.cat([up4, d4], dim=1))
        up6 = self.up6(torch.cat([up5, d3], dim=1))
        up7 = self.up7(torch.cat([up6, d2], dim=1))
        return self.final_up(torch.cat([up7, d1], dim=1))

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(in_channels * 2, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            Block(features, features * 2, down=True, act="leaky"),
            Block(features * 2, features * 4, down=True, act="leaky"),
            Block(features * 4, features * 8, down=True, act="leaky"),
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.disc(x)

# Initialize models and optimizers
generator = Generator().to(device)
discriminator = Discriminator().to(device)
opt_gen = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
loss_L1 = nn.L1Loss()
loss_GAN = nn.BCEWithLogitsLoss()

# Load dataset
transform = Transforms()
dataset = ImageColorizationDataset(train_dir=path_train, target_dir=path_target, transform=transform)

# Check if the dataset loaded correctly
if not dataset:
    raise ValueError("Failed to load the dataset. Check the paths.")

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Training function
def train_step(input_image, target):
    # Train Discriminator
    with torch.cuda.amp.autocast():
        output_image = generator(input_image)
        disc_real = discriminator(input_image, target)
        disc_fake = discriminator(input_image, output_image.detach())
        loss_disc_real = loss_GAN(disc_real, torch.ones_like(disc_real))
        loss_disc_fake = loss_GAN(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2

    opt_disc.zero_grad()
    loss_disc.backward()
    opt_disc.step()

    # Train Generator
    with torch.cuda.amp.autocast():
        disc_fake = discriminator(input_image, output_image)
        loss_gan = loss_GAN(disc_fake, torch.ones_like(disc_fake))
        loss_l1 = loss_L1(output_image, target)
        loss_gen = loss_gan + (L1_LAMBDA * loss_l1)

    opt_gen.zero_grad()
    loss_gen.backward()
    opt_gen.step()

    return loss_disc.item(), loss_gen.item()

# Training loop
for epoch in range(NUM_EPOCHS):
    for idx, (input_image, target) in enumerate(dataloader):
        # Check if the images are valid
        if input_image is None or target is None:
            print(f"Skipping batch {idx} due to invalid images.")
            continue

        input_image, target = input_image.to(device), target.to(device)
        loss_disc, loss_gen = train_step(input_image, target)

        print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {idx}/{len(dataloader)} \
              Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")

    # Save checkpoints
    if epoch % 5 == 0:
        torch.save(generator.state_dict(), os.path.join(checkpoint_folder, f"generator_epoch_{epoch}.pth"))
        torch.save(discriminator.state_dict(), os.path.join(checkpoint_folder, f"discriminator_epoch_{epoch}.pth"))

print("Training complete!")
