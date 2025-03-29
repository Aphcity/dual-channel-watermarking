import torch
import model
import hashlib
import numpy as np
from PIL import Image
from torchvision import transforms
import phash
def load_pretrained_models(encoder_path, encoder1_path, decoder_path, decoder1_path, decoder2_path, device):
    encoder = torch.load(encoder_path, map_location=device)
    encoder1 = torch.load(encoder1_path, map_location=device)
    decoder = torch.load(decoder_path, map_location=device)
    decoder1 = torch.load(decoder1_path, map_location=device)
    decoder2 = torch.load(decoder2_path, map_location=device)

    encoder.eval()
    encoder1.eval()
    decoder.eval()
    decoder1.eval()
    decoder2.eval()

    return encoder, encoder1, decoder, decoder1, decoder2

def compute_sha3_hash(latent, device):
    latent_np = latent.detach().cpu().numpy()
    sha3_hash = hashlib.sha3_256(latent_np.tobytes()).digest()  
    sha3_hash_tensor = torch.tensor(list(sha3_hash), dtype=torch.float32).to(device)  
    return sha3_hash_tensor

def process_single_image(image_path, secret, encoder, encoder1, decoder, decoder1, decoder2, autoencoder, device):
    transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image_input = transform(image).unsqueeze(0).to(device)  

    secret_input = torch.tensor(secret, dtype=torch.float32).unsqueeze(0).to(device) 
    encoder_output = encoder((secret_input, image_input))  
    encoded_image = encoder_output

    latent = autoencoder.encoder(encoded_image)  

    sha3_hash_tensor = compute_sha3_hash(latent, device).unsqueeze(0) 

    encoder1_output = encoder1((sha3_hash_tensor, encoded_image)) 
    final_encoded_image = encoded_image + encoder1_output 

    extracted_secret = decoder(final_encoded_image) 

    reconstructed_residual = decoder1(final_encoded_image) 

    reconstructed_image = final_encoded_image - reconstructed_residual 

    extracted_hash = decoder2(reconstructed_image) 

    return final_encoded_image, extracted_secret, reconstructed_image, extracted_hash

if __name__ == "__main__":
    encoder_path = "path/to/encoder.pth"
    encoder1_path = "path/to/encoder1.pth"
    decoder_path = "path/to/decoder.pth"
    decoder1_path = "path/to/decoder1.pth"
    decoder2_path = "path/to/decoder2.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder, encoder1, decoder, decoder1, decoder2 = load_pretrained_models(
        encoder_path, encoder1_path, decoder_path, decoder1_path, decoder2_path, device
    )

    autoencoder = model.Autoencoder(latent_dim=2048).to(device)
    autoencoder.eval()

    image_path = "path/to/image.jpg"  
    secret_hash = phash.calculate_phash(image_path, hash_size=int(64**0.5))
    secret = np.array([int(bit) for bit in bin(int(secret_hash, 16))[2:].zfill(64)])
    secret = torch.from_numpy(secret).float()

    
    final_encoded_image, extracted_secret, reconstructed_image, extracted_hash = process_single_image(
        image_path, secret, encoder, encoder1, decoder, decoder1, decoder2, autoencoder, device
    )


    print(f"Final encoded image shape: {final_encoded_image.shape}")  
    print(f"Extracted secret shape: {extracted_secret.shape}")  
    print(f"Reconstructed image shape: {reconstructed_image.shape}")  
    print(f"Extracted hash shape: {extracted_hash.shape}") 