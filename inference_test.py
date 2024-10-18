from transformers import pipeline

# Create inference pipeline for our model.
model_id = "itsx-tom/estates-exterier-interier-classifier"
pipe = pipeline("image-classification", model=model_id)

# Run some tests.
print(f"parrot: {pipe('https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png')}")
print(f"house exterier: {pipe('https://img.flatzone.cz/thumbnails/350x200/dcd4d8d2b04e401e.webp')}")
print(f"house exterier: {pipe('https://img.flatzone.cz/thumbnails/350x200/65291922a07c90ac.webp')}")
print(f"house exterier: {pipe('https://img.flatzone.cz/thumbnails/350x200/c9882b74e1a7dbf2.webp')}")
print(f"house exterier: {pipe('https://img.flatzone.cz/thumbnails/350x200/4ab6dc1ff32f3e6d.webp')}")
print(f"house exterier: {pipe('https://img.flatzone.cz/thumbnails/350x200/498a5be374390542.webp')}")
print(f"house exterier: {pipe('https://img.flatzone.cz/thumbnails/350x200/0f55ba9452a6f784.webp')}")
print(f"house interier: {pipe('https://img.flatzone.cz/thumbnails/350x200/2a73f552beea8da8.webp')}")
print(f"house interier: {pipe('https://img.flatzone.cz/thumbnails/350x200/88eebca3149a8dfe.webp')}")


# Create inference pipeline for example model.

model_id = "andupets/real-estate-image-classification"
pipe = pipeline("image-classification", model=model_id)

# Run some tests.
print(f"parrot: {pipe('https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png')}")
print(f"house facade: {pipe('https://img.flatzone.cz/thumbnails/350x200/dcd4d8d2b04e401e.webp')}")

