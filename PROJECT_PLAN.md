# Project Context: Urban Heat Island (UHI) Prediction from Satellite Imagery

We are building a Machine Learning project to predict Land Surface Temperature (LST) based on multispectral satellite imagery (RGB + NIR). The task is treated as an Image-to-Image Regression problem. 

**Tech Stack:** 
- Python
- PyTorch (for the Deep Learning model)
- Rasterio / TIFFFile / OpenCV (for handling satellite images)
- Streamlit (for the UI)
- Scikit-learn (for data splitting and metrics)
- Matplotlib/Seaborn (for visualization)

**Data Context:**
I will manually download and provide the dataset. The data will be stored locally in the following directory structure:
/dataset/
  /input_features/ (contains .tif files for RGB + Near Infrared bands)
  /target_labels/ (contains .tif files representing actual surface temperature)

**Goal:**
I need you to help me write the code step-by-step. Please do not write all the code at once. Let's go module by module. Here is the roadmap:

### Step 1: Data Preparation & Dataloader
Write a Python script that:
1. Iterates through the manual dataset directories.
2. Crops the large large GeoTIFF files into smaller patches (e.g., 256x256 pixels) to make them manageable for the GPU.
3. Implements a PyTorch `Dataset` class (`UHIDataset`) that reads the patches.
4. Normalizes the input features (Min-Max scaling) and target temperatures.
5. Implements PyTorch `DataLoader` for Training and Validation splits.

### Step 2: Model Architecture (U-Net for Regression)
Write the PyTorch model:
1. Implement a U-Net architecture.
2. The input channels should be 4 (Red, Green, Blue, Near-Infrared).
3. The output must be 1 channel (the predicted temperature map).
4. Make sure the final activation function is suitable for regression (e.g., Identity/Linear, NOT Softmax/Sigmoid since temperatures can be outside [0,1] if not strictly scaled, or Sigmoid if target is strictly Min-Max scaled).

### Step 3: Training Loop
Write the training script:
1. Set up the optimizer (Adam).
2. Set up the Loss Function (Mean Squared Error - MSE or L1 Loss).
3. Write the training loop over N epochs, logging the training loss and validation loss.
4. Save the best model weights (`best_model.pth`) based on validation MAE (Mean Absolute Error, so we can interpret the error in degrees Celsius).

### Step 4: Streamlit User Interface (UI)
Write a `app.py` script using Streamlit that:
1. Loads the saved `best_model.pth`.
2. Provides a file uploader in the sidebar for the user to upload a sample 4-channel image (or an RGB image, fallback if needed).
3. Runs the model inference on the uploaded image.
4. Displays the results side-by-side using `st.columns`:
   - Left: The original optical image (RGB representation).
   - Right: The predicted Thermal Heatmap overlaid with a standard heatmap colormap (e.g., 'jet' or 'inferno' from Matplotlib) and a colorbar indicating temperatures.

**Agent Instructions:**
Acknowledge this plan. Once you confirm you understand, wait for me to say "Start Step 1" and we will begin with the Data Preparation module.