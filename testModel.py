import torch
import model
import hashlib
import numpy as np
from PIL import Image
from torchvision import transforms
import phash
import json
import os

def load_pretrained_models(encoder_path, encoder1_path, decoder_path, decoder1_path, decoder2_path, device):
    encoder = torch.load(encoder_path, map_location=device, weights_only=False)
    encoder1 = torch.load(encoder1_path, map_location=device, weights_only=False)
    decoder = torch.load(decoder_path, map_location=device, weights_only=False)
    decoder1 = torch.load(decoder1_path, map_location=device, weights_only=False)
    decoder2 = torch.load(decoder2_path, map_location=device, weights_only=False)

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

def encode_image(image_path, author_info, encoder, encoder1, decoder, decoder1, decoder2, autoencoder, device):
    transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image_input = transform(image).unsqueeze(0).to(device)

    # Convert author info JSON to secret tensor
    secret = json.dumps(author_info)
    secret_hash = phash.calculate_phash(image_path, hash_size=int(64**0.5))
    secret_array = np.array([int(bit) for bit in bin(int(secret_hash, 16))[2:].zfill(64)])
    secret_input = torch.tensor(secret_array, dtype=torch.float32).unsqueeze(0).to(device)

    encoder_output = encoder((secret_input, image_input))
    encoded_image = encoder_output
    latent = autoencoder.encoder(encoded_image)
    sha3_hash_tensor = compute_sha3_hash(latent, device).unsqueeze(0)
    encoder1_output = encoder1((sha3_hash_tensor, encoded_image))
    final_encoded_image = encoded_image + encoder1_output

    # Save the encoded image
    os.makedirs('out', exist_ok=True)
    encoded_image_path = os.path.join('out', 'encoded_image.png')
    transforms.ToPILImage()(final_encoded_image.squeeze().cpu()).save(encoded_image_path)

    return encoded_image_path

def decode_image(encoded_image_path, encoder, encoder1, decoder, decoder1, decoder2, autoencoder, device):
    transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
    ])
    encoded_image = Image.open(encoded_image_path).convert("RGB")
    encoded_image_input = transform(encoded_image).unsqueeze(0).to(device)

    latent = autoencoder.encoder(encoded_image_input)
    sha3_hash_tensor = compute_sha3_hash(latent, device).unsqueeze(0)
    encoder1_output = encoder1((sha3_hash_tensor, encoded_image_input))
    final_encoded_image = encoded_image_input + encoder1_output
    extracted_secret = decoder(final_encoded_image)
    reconstructed_residual = decoder1(final_encoded_image)
    reconstructed_image = final_encoded_image - reconstructed_residual
    extracted_hash = decoder2(reconstructed_image)

    # Save the reconstructed image
    os.makedirs('out', exist_ok=True)
    decoded_image_path = os.path.join('out', 'decoded_image.png')
    transforms.ToPILImage()(reconstructed_image.squeeze().cpu()).save(decoded_image_path)
    print(f"Decoded image saved to: {decoded_image_path}")

    # Print the extracted JSON format info
    extracted_secret_np = extracted_secret.detach().cpu().numpy().flatten()
    extracted_info = ''.join([str(int(bit)) for bit in extracted_secret_np])
    extracted_info_json = json.loads(extracted_info)
    print(json.dumps(extracted_info_json, indent=4))

    return extracted_info_json, decoded_image_path

if __name__ == "__main__":
    encoder_path = "saved_models/encoder.pth"
    encoder1_path = "saved_models/encoder1.pth"
    decoder_path = "saved_models/decoder.pth"
    decoder1_path = "saved_models/decoder1.pth"
    decoder2_path = "saved_models/decoder2.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder, encoder1, decoder, decoder1, decoder2 = load_pretrained_models(
        encoder_path, encoder1_path, decoder_path, decoder1_path, decoder2_path, device
    )

    autoencoder = model.Autoencoder(latent_dim=2048).to(device)
    autoencoder.eval()

    image_path = "imgs/iphone-vs-android-global-market-share.png"
    author_info = {
        "author": "Your Name",
        "email": "your.email@example.com",
        "date": "2025-04-03"
    }

    encoded_image_path = encode_image(image_path, author_info, encoder, encoder1, decoder, decoder1, decoder2, autoencoder, device)
    print(f"Encoded image saved to: {encoded_image_path}")

    decoded_info, decoded_image_path = decode_image(encoded_image_path, encoder, encoder1, decoder, decoder1, decoder2, autoencoder, device)
    print("Decoded JSON info:")
    print(json.dumps(decoded_info, indent=4))