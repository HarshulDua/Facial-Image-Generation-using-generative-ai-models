import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import csv
import kagglehub
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. Dataset & Dataloader
# ==========================================
class GlassesDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = []
        self.labels = []
        print("Scanning Kaggle download for images and CSV files...")
        
        csv_path = None
        img_dir = None
        for root, _, files in os.walk(root_dir):
            if 'train.csv' in files:
                csv_path = os.path.join(root, 'train.csv')
            if any(f.endswith('.png') or f.endswith('.jpg') for f in files):
                img_dir = root
                
        if csv_path and img_dir:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)
                label_idx = headers.index('glasses') if 'glasses' in headers else -1
                
                for row in reader:
                    if not row: 
                        continue
                    img_id = row[0]
                    label = int(row[label_idx])
                    
                    path1 = os.path.join(img_dir, f"face-{img_id}.png")
                    path2 = os.path.join(img_dir, f"{img_id}.png")
                    
                    if os.path.exists(path1):
                        self.image_paths.append(path1)
                        self.labels.append(label)
                    elif os.path.exists(path2):
                        self.image_paths.append(path2)
                        self.labels.append(label)
        
        print(f"Successfully loaded {len(self.image_paths)} images into memory.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img), torch.tensor(self.labels[idx])

# ==========================================
# 2. VAE Architecture
# ==========================================
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.enc1 = nn.Conv2d(3, 32, 4, 2, 1)
        self.enc2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.enc3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)
        
        self.fc_dec = nn.Linear(latent_dim + 1, 128 * 8 * 8)
        self.dec1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.dec3 = nn.ConvTranspose2d(32, 3, 4, 2, 1)

    def encode(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z, labels):
        z_cond = torch.cat([z, labels.view(-1, 1).float()], dim=1)
        x = F.relu(self.fc_dec(z_cond)).view(-1, 128, 8, 8)
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        return torch.sigmoid(self.dec3(x))

    def forward(self, x, labels):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar

def vae_loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# ==========================================
# 3. GAN Architecture
# ==========================================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(100 + 1, 256 * 8 * 8)
        self.conv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.conv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.conv3 = nn.ConvTranspose2d(64, 3, 4, 2, 1)

    def forward(self, z, labels):
        z_cond = torch.cat([z, labels.view(-1, 1).float()], dim=1)
        x = F.relu(self.fc(z_cond)).view(-1, 256, 8, 8)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return torch.sigmoid(self.conv3(x))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.fc = nn.Linear(256 * 8 * 8, 1)

    def forward(self, img, labels):
        lbl_ch = labels.view(-1, 1, 1, 1).expand(-1, 1, 64, 64).float()
        x = F.leaky_relu(self.conv1(torch.cat([img, lbl_ch], dim=1)), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        return torch.sigmoid(self.fc(x.view(x.size(0), -1)))

# ==========================================
# 4. Diffusion Architecture (DDPM & U-Net)
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SimpleUNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=2):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU()
        )
        self.class_emb = nn.Embedding(num_classes, time_dim)

        self.inc = nn.Conv2d(c_in, 64, 3, padding=1)
        self.down1 = nn.Conv2d(64, 128, 4, 2, 1)
        self.down2 = nn.Conv2d(128, 256, 4, 2, 1)
        
        self.emb1 = nn.Linear(time_dim, 128)
        self.emb2 = nn.Linear(time_dim, 256)

        self.bot1 = nn.Conv2d(256, 256, 3, padding=1)
        self.bot2 = nn.Conv2d(256, 256, 3, padding=1)

        self.up1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(128 * 2, 64, 4, 2, 1)
        
        self.outc = nn.Conv2d(64 * 2, c_out, 1)

    def forward(self, x, t, labels):
        t_emb = self.time_mlp(t)
        c_emb = self.class_emb(labels)
        emb = t_emb + c_emb 

        x1 = F.relu(self.inc(x))
        x2 = F.relu(self.down1(x1))
        x2 = x2 + self.emb1(emb)[:, :, None, None].expand(-1, -1, x2.shape[2], x2.shape[3])
        x3 = F.relu(self.down2(x2))
        x3 = x3 + self.emb2(emb)[:, :, None, None].expand(-1, -1, x3.shape[2], x3.shape[3])

        x3 = F.relu(self.bot1(x3))
        x3 = F.relu(self.bot2(x3))

        x = F.relu(self.up1(x3))
        x = torch.cat([x, x2], dim=1) 
        x = F.relu(self.up2(x))
        x = torch.cat([x, x1], dim=1)
        
        return self.outc(x)

class DDPM:
    def __init__(self, model, num_steps=1000):
        self.model = model
        self.num_steps = num_steps
        self.betas = torch.linspace(1e-4, 0.02, num_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def forward_diffusion(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod[t])[:, None, None, None]
        return sqrt_alphas_cumprod * x_0 + sqrt_one_minus_alphas_cumprod * noise, noise

    @torch.no_grad()
    def sample(self, labels):
        self.model.eval()
        n_samples = len(labels)
        x = torch.randn((n_samples, 3, 64, 64)).to(device)
        
        for i in tqdm(reversed(range(self.num_steps)), desc="Sampling Diffusion", total=self.num_steps):
            t = (torch.ones(n_samples) * i).long().to(device)
            predicted_noise = self.model(x, t, labels)
            
            alpha = self.alphas[t][:, None, None, None]
            alpha_cumprod = self.alphas_cumprod[t][:, None, None, None]
            beta = self.betas[t][:, None, None, None]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * predicted_noise) + torch.sqrt(beta) * noise
            
        self.model.train()
        return (x.clamp(-1, 1) + 1) / 2 

# ==========================================
# 5. Image Saving Utility
# ==========================================
def save_generated_images(model_type, samples, labels):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for i, ax in enumerate(axes.flatten()):
        img = np.transpose(samples[i], (1, 2, 0))
        ax.imshow(img)
        ax.set_title("Glasses" if labels[i] == 1 else "No Glasses")
        ax.axis('off')
    
    filename = f"{model_type}_results.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

# ==========================================
# 6. Main Execution Block
# ==========================================
if __name__ == "__main__":
    path = kagglehub.dataset_download("jeffheaton/glasses-or-no-glasses")
    dataset = GlassesDataset(root_dir=path)
    
    if len(dataset) == 0:
        print("Dataset failed to load. Aborting.")
        exit()
        
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
    target_labels = torch.tensor([0, 0, 0, 1, 1, 1]).to(device)
    
    # Requirement: Specific Epoch counts for each model
    vae_epochs = 75
    gan_epochs = 100
    diff_epochs = 120

    # ---------------------------------------------------------
    # PART A: Train VAE
    # ---------------------------------------------------------
    print("\n--- Starting VAE Training ---")
    vae = VAE().to(device)
    opt_vae = optim.Adam(vae.parameters(), lr=1e-3)
    
    for epoch in range(vae_epochs):
        pbar = tqdm(dataloader, desc=f"VAE Epoch {epoch+1}/{vae_epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            opt_vae.zero_grad()
            recon, mu, logvar = vae(imgs, labels)
            loss = vae_loss_fn(recon, imgs, mu, logvar)
            loss.backward()
            opt_vae.step()
            pbar.set_postfix(loss=loss.item()/len(imgs))
            
    vae.eval()
    with torch.no_grad():
        z = torch.randn(6, 128).to(device)
        vae_samples = vae.decode(z, target_labels).cpu().numpy()
    save_generated_images('VAE', vae_samples, target_labels.cpu().numpy())

    # ---------------------------------------------------------
    # PART B: Train GAN
    # ---------------------------------------------------------
    print("\n--- Starting GAN Training ---")
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    optG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    
    for epoch in range(gan_epochs):
        pbar = tqdm(dataloader, desc=f"GAN Epoch {epoch+1}/{gan_epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            b_size = imgs.size(0)
            
            # Update Discriminator
            netD.zero_grad()
            errD_real = criterion(netD(imgs, labels), torch.ones(b_size, 1).to(device))
            fake_imgs = netG(torch.randn(b_size, 100).to(device), labels)
            errD_fake = criterion(netD(fake_imgs.detach(), labels), torch.zeros(b_size, 1).to(device))
            errD = errD_real + errD_fake
            errD.backward()
            optD.step()
            
            # Update Generator
            netG.zero_grad()
            errG = criterion(netD(fake_imgs, labels), torch.ones(b_size, 1).to(device))
            errG.backward()
            optG.step()
            
            pbar.set_postfix(D_loss=errD.item(), G_loss=errG.item())
            
    netG.eval()
    with torch.no_grad():
        z = torch.randn(6, 100).to(device)
        gan_samples = netG(z, target_labels).cpu().numpy()
    save_generated_images('GAN', gan_samples, target_labels.cpu().numpy())

    # ---------------------------------------------------------
    # PART C: Train Diffusion Model
    # ---------------------------------------------------------
    print("\n--- Starting Diffusion Training ---")
    unet = SimpleUNet().to(device)
    opt_diff = optim.Adam(unet.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()
    diffusion = DDPM(unet, num_steps=1000)
    
    for epoch in range(diff_epochs):
        pbar = tqdm(dataloader, desc=f"Diffusion Epoch {epoch+1}/{diff_epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Diffusion models require inputs scaled to [-1, 1]
            imgs_diff = imgs * 2.0 - 1.0
            
            t = torch.randint(0, diffusion.num_steps, (imgs.shape[0],)).to(device)
            x_t, true_noise = diffusion.forward_diffusion(imgs_diff, t)
            
            opt_diff.zero_grad()
            predicted_noise = unet(x_t, t, labels)
            loss = mse_loss(predicted_noise, true_noise)
            loss.backward()
            opt_diff.step()
            
            pbar.set_postfix(MSE=loss.item())

    print("\nStarting Reverse Diffusion Sampling...")
    diff_samples = diffusion.sample(labels=target_labels).cpu().numpy()
    save_generated_images('Diffusion', diff_samples, target_labels.cpu().numpy())
    
    print("\nExecution complete. Generated images have been saved to the current directory.")