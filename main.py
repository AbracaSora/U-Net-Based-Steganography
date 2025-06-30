from torchvision.utils import save_image
from tqdm import tqdm

from UNet.unet_model import UNet
from Extractor import Extractor
from Dataset import ImageDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hiding_model = UNet(n_channels=6, n_classes=3, bilinear=True)
extractor = Extractor(in_channels=3, out_channels=3)
hiding_model = hiding_model.to(device)
extractor = extractor.to(device)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop((256, 256)),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
dataset = ImageDataset(cover_path='/media/新加卷/images/MIRFlickR/images/0/',
                          secret_path='/media/新加卷/images/MIRFlickR/images/1/',
                            transform=transform)
print(f"Dataset size: {len(dataset)}")

EPOCHS = 10
BATCH_SIZE = 4
alpha = 1 # Weight for the loss function, can be adjusted based on the task
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
optimizer = torch.optim.Adam(list(hiding_model.parameters()) + list(extractor.parameters()), lr=0.001)
loss_fn = torch.nn.MSELoss()
for epoch in range(EPOCHS):
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}', unit='batch')
    for cover, secret in pbar:
        cover = cover.to(device)
        secret = secret.to(device)

        optimizer.zero_grad()

        # Concatenate cover and secret images along the channel dimension
        input_images = torch.cat((cover, secret), dim=1)

        # Forward pass through the hiding model
        hidden_images = hiding_model(input_images)

        # Forward pass through the extractor
        extracted_images = extractor(hidden_images)

        loss = alpha * loss_fn(extracted_images, secret) + loss_fn(hidden_images, cover)

        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())
        if pbar.n % 10 == 0:
            save_image(hidden_images, f'Output/hidden_epoch{epoch+1}_batch{pbar.n}.png')
            save_image(extracted_images, f'Output/extracted_epoch{epoch+1}_batch{pbar.n}.png')
            save_image(cover, f'Output/cover_epoch{epoch+1}_batch{pbar.n}.png')
            save_image(secret, f'Output/secret_epoch{epoch+1}_batch{pbar.n}.png')
