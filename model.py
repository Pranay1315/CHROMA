import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Paths (These paths are specific to your local machine)
path_train = "./Dataset/train_black/"
path_target = "./Dataset/train_color/"
checkpoint_folder = "./Dataset/checkpoints/"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 50
NUM_EPOCHS = 200

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
        
        # Using PIL to open the images
        input_img = Image.open(img_name).convert("RGB")
        target_img = Image.open(target_name).convert("RGB")

        if self.transform:
            input_img, target_img = self.transform(input_img, target_img)

        # Ensure images are not None before transposing
        if input_img is not None and target_img is not None:
            return np.array(input_img).astype(np.float32).transpose(2, 0, 1), np.array(target_img).astype(np.float32).transpose(2, 0, 1)
        else:
            return None, None

# Data augmentation using PIL
def random_jitter(input_img, tar_img):
    # Resize to a fixed size BEFORE cropping using PIL
    input_img = input_img.resize((286, 286))
    tar_img = tar_img.resize((286, 286))

    # Generate random coordinates for cropping
    crop_x = np.random.randint(0, 286 - 256)
    crop_y = np.random.randint(0, 286 - 256)

    # Crop both images using the same coordinates
    input_img = input_img.crop((crop_x, crop_y, crop_x + 256, crop_y + 256))
    tar_img = tar_img.crop((crop_x, crop_y, crop_x + 256, crop_y + 256))

    if np.random.rand() > 0.5:
        input_img = input_img.transpose(Image.FLIP_LEFT_RIGHT)
        tar_img = tar_img.transpose(Image.FLIP_LEFT_RIGHT)

    return input_img, tar_img

class Transforms:
    def __call__(self, inp, tar):
        inp, tar = random_jitter(inp, tar)
        inp = (np.array(inp) / 255.0) * 2 - 1  # Convert to numpy array for torch compatibility
        tar = (np.array(tar) / 255.0) * 2 - 1  # Convert to numpy array for torch compatibility
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

         # Apply label smoothing
        real_labels = torch.ones_like(disc_real) * 0.9  # Smooth real labels
        fake_labels = torch.zeros_like(disc_fake) + 0.1  # Smooth fake labels

        loss_disc_real = loss_GAN(disc_real, real_labels)
        loss_disc_fake = loss_GAN(disc_fake, fake_labels)
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

# Function to save sample images
def save_sample_images(epoch, generator, dataloader, device, save_dir="./samples"):
    os.makedirs(save_dir, exist_ok=True)

    generator.eval()  # Set to evaluation mode
    with torch.no_grad():
        for idx, (input_image, _) in enumerate(dataloader):
            input_image = input_image.to(device)
            colorized_image = generator(input_image)

            # Convert from [-1, 1] to [0, 1] range
            colorized_image = (colorized_image + 1) / 2

            # Convert tensor to PIL image
            colorized_image = colorized_image[0].cpu().detach().numpy().transpose(1, 2, 0)  # Convert to HWC
            colorized_image = np.uint8(colorized_image * 255)

            pil_image = Image.fromarray(colorized_image)
            pil_image.save(os.path.join(save_dir, f"epoch_{epoch}sample{idx}.png"))
            
            # Display using matplotlib
            # plt.figure(figsize=(8, 8))
            # plt.axis("off")
            # plt.title(f"Generated Image - Epoch {epoch}")
            # plt.imshow(colorized_image)
            # plt.show()

            if idx == 0:  # Show only the first batch
                break

    generator.train()

def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from {checkpoint_path} at epoch {epoch}")
        return epoch
    return 0

# Training loop
def train():
    # start_epoch = load_checkpoint(generator, opt_gen, os.path.join(checkpoint_folder, "generator_latest.pth"))
    # load_checkpoint(discriminator, opt_disc, os.path.join(checkpoint_folder, "discriminator_latest.pth"))
    start_epoch = 0
    for epoch in range(start_epoch, NUM_EPOCHS):
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': opt_gen.state_dict(),
            }, os.path.join(checkpoint_folder, "generator_latest.pth"))
            torch.save({
                'epoch': epoch,
                'model_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': opt_disc.state_dict(),
            }, os.path.join(checkpoint_folder, "discriminator_latest.pth"))
        
        # Save sample images every 5 epochs
        if epoch % 5 == 0:
            save_sample_images(epoch, generator, dataloader, device)

    print("Training complete!")

if __name__ == "__main__":
    train()
