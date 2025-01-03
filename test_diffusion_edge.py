import coremltools as ct
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

def test_diffusion_edge_conversion():
    # Load the CoreML model
    model = ct.models.MLModel('DiffusionEdge.mlpackage')
    
    # Load and prepare test image
    original_image = Image.open('test.jpg')
    
    # Set target resolution to 1024
    TARGET_RES = 1024
    BATCH_SIZE = 4
    
    # Preserve aspect ratio while resizing to higher resolution
    width, height = original_image.size
    aspect_ratio = width / height
    if width > height:
        new_width = TARGET_RES
        new_height = int(TARGET_RES / aspect_ratio)
    else:
        new_height = TARGET_RES
        new_width = int(TARGET_RES * aspect_ratio)
    
    # Convert to grayscale and resize
    test_image = original_image.convert('L').resize((new_width, new_height), Image.LANCZOS)
    
    # Pad to square if needed
    if new_width != new_height:
        background = Image.new('L', (TARGET_RES, TARGET_RES), 255)
        offset = ((TARGET_RES - new_width) // 2, (TARGET_RES - new_height) // 2)
        background.paste(test_image, offset)
        test_image = background
    
    # Create batch input by repeating the image
    batch_input = []
    for _ in range(BATCH_SIZE):
        batch_input.append(test_image)
    
    try:
        print("Starting prediction...")
        # Process each batch item
        batch_outputs = []
        for input_image in batch_input:
            coreml_input = {'input': input_image}
            coreml_output = model.predict(coreml_input)
            batch_outputs.append(coreml_output['output'])
        
        # Combine batch outputs
        output_array = np.mean(batch_outputs, axis=0)
        print(f"Output shape: {output_array.shape}")
        print(f"Output min/max values: {output_array.min()}, {output_array.max()}")
        
        # Enhanced post-processing for edge detection
        output_normalized = output_array[0, 0]  # Remove batch and channel dimensions
        
        # Apply more sophisticated edge processing
        # Use Otsu's thresholding for better edge detection
        from skimage.filters import threshold_otsu
        thresh = threshold_otsu(output_normalized)
        output_binary = output_normalized > thresh
        output_binary = output_binary.astype(np.uint8) * 255
        
        # Create output image
        output_image = Image.fromarray(output_binary)
        
        # Crop back to original aspect ratio if needed
        if new_width != new_height:
            output_image = output_image.crop((
                (TARGET_RES - new_width) // 2,
                (TARGET_RES - new_height) // 2,
                (TARGET_RES + new_width) // 2,
                (TARGET_RES + new_height) // 2
            ))
            output_image = output_image.resize((width, height), Image.LANCZOS)
        
        # Save results
        original_image.save('original.png')
        output_image.save('edge_detection.png')
        
        print("Test complete! Check original.png and edge_detection.png")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        print(f"Model input info: {model.input_description}")
        print(f"Model output info: {model.output_description}")

if __name__ == "__main__":
    test_diffusion_edge_conversion()