# saliency_with_histogram_equalization

Step 1: Load the Model
```
import torch
from torchvision import models


# Load the pre-trained ResNet-50 model

model = models.resnet50(pretrained=True)
model.eval()  # Set the model to evaluation mode
```
The pre-trained ResNet-50 model is loaded and the model is set to evaluation mode. This mode ensures the behaviors specific to training are disabled and give continuous inference results.

Step 2: Preprocess the Model
```
from torchvision import transforms
from PIL import Image

# Preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load your image
img = Image.open("path_to_your_image.jpg")
img_tensor = preprocess(img)
img_tensor.unsqueeze_(0)
```
The input images is loaded and preprocessed based on the input format required by the model (resnet-50).

Step 3: Perform Forward and Backward Passes
```
# Gradients are enabled
img_tensor.requires_grad_(True)

# Forward pass
outputs = model(img_tensor)

# Get the max log-probability
score, index = torch.max(outputs, 1)

# Zero gradients
model.zero_grad()

# Backward pass to get gradients
score.backward()
```

By setting the “requires_grad ”of img_tensor to True, PyTorch tracks the operations on the tensor using PyTorch’s autograd system. Further, The forward pass is done by feeding the model with the input, which returns the predicted output. Next, The class with highest prediction score for the input image is chosen. The “model.zero_grad()” helps to prevent the accumulation of gradients by clearing the existing gradient. Then lastly, backward pass is done to calculate the gradient with respect to the predicted class.

Step 4: Generate the Saliency Map
```
import numpy as np
from scipy.ndimage import zoom
from skimage import exposure 

# Get the gradients of the input image
gradients = img_tensor.grad.data

# Process the gradients for visualization
gradients = gradients.abs().squeeze().numpy()
gradients = np.max(gradients, axis=0)

# Normalize to 0-1
gradients_raw = (gradients - gradients.min()) / (gradients.max() - gradients.min())

# Apply histogram equalization
gradients_eq = exposure.equalize_hist(gradients)

# Resize saliency maps to original image size
original_size = img.size
zoom_factors = [original_size[1] / gradients.shape[0], original_size[0] / gradients.shape[1]]
saliency_map_resized_raw = zoom(gradients_raw, zoom_factors, order=1)  
saliency_map_resized_eq = zoom(gradients_eq, zoom_factors, order=1)

```

The gradients are processed to create a Saliency Map, which is then normalized and then resized to the original image size for visualizations. This equalized gradient is also been resized to the original image size.

Step 5: Visualize the Results
```
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 4, figsize=(24, 6))

# Original image
ax[0].imshow(img)
ax[0].axis('off')
ax[0].set_title('Original Image')

# Raw saliency map
ax[1].imshow(saliency_map_resized_raw, cmap='hot')
ax[1].axis('off')
ax[1].set_title('Raw Saliency Map')

# Saliency map with histogram equalization
ax[2].imshow(saliency_map_resized_eq, cmap='hot')
ax[2].axis('off')
ax[2].set_title('Saliency Map with Histogram Equalization')

# Overlay of the saliency map with histogram equalization on the original image
ax[3].imshow(img)
ax[3].imshow(saliency_map_resized_eq, cmap='hot', alpha=0.5)  
ax[3].axis('off')
ax[3].set_title('Overlay of Saliency Map on Image')

plt.show()
```
Finally, we visualize the original image alongside with the raw Saliency Map, Saliency Map with Histogram Equalization and overlaying it on top of the image to see which areas influenced by the model’s decision.

Step 6: Result
