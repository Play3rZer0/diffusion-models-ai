import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def add_gaussian_noise(image_array, variance_schedule_beta, current_step):
    """
    Adds Gaussian noise to an image array based on a variance schedule.
    This simulates one step of the forward diffusion process.

    Args:
        image_array (np.ndarray): The input image as a NumPy array (e.g., grayscale or RGB).
        variance_schedule_beta (list or np.ndarray): A list/array of beta values
                                                    defining the noise level at each step.
        current_step (int): The current step in the diffusion process (0-indexed).

    Returns:
        np.ndarray: The image array with added noise.
        np.ndarray: The noise that was added.
    """
    if not (0 <= current_step < len(variance_schedule_beta)):
        raise ValueError("current_step must be within the bounds of variance_schedule_beta")

    beta_t = variance_schedule_beta[current_step]

    # Calculate alpha_t and alpha_bar_t for the current step
    # alpha_t = 1 - beta_t
    # alpha_bar_t = product of (1 - beta_s) from s=1 to t
    # For simplicity here, we're just adding noise based on beta_t directly
    # as a conceptual demonstration of "adding noise".
    # In a full DDPM, the noise addition is more complex, involving alpha_bar_t
    # to allow direct sampling of x_t from x_0.

    # Generate Gaussian noise with mean 0 and variance beta_t
    noise = np.random.normal(0, np.sqrt(beta_t), image_array.shape)

    # Add noise to the image. We scale the image to keep it in a reasonable range
    # for visualization, though in actual diffusion models, scaling and normalization
    # are handled carefully.
    noisy_image_array = image_array + noise * 50 # Multiply noise for better visibility

    # Clip values to ensure they stay within valid image range (e.g., 0-255 for uint8)
    noisy_image_array = np.clip(noisy_image_array, 0, 255)

    return noisy_image_array, noise

def main():
    # 1. Create a simple grayscale image (e.g., a gradient)
    image_size = 64
    original_image_array = np.linspace(0, 255, image_size * image_size).reshape(image_size, image_size)
    original_image_array = original_image_array.astype(np.float32) # Use float for calculations

    print(f"Original image shape: {original_image_array.shape}")
    print(f"Original image min/max: {original_image_array.min()}/{original_image_array.max()}")

    # Define a simple variance schedule (beta values)
    # These values control how much noise is added at each step.
    # A linear schedule is common, where beta increases over time.
    num_diffusion_steps = 10
    variance_schedule_beta = np.linspace(0.0001, 0.02, num_diffusion_steps)

    # Prepare for plotting
    fig, axes = plt.subplots(1, num_diffusion_steps + 1, figsize=(15, 3))
    axes[0].imshow(original_image_array, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    current_noisy_image = original_image_array.copy()

    # 2. Simulate the forward diffusion process
    print("\nSimulating forward diffusion:")
    for i in range(num_diffusion_steps):
        print(f"Step {i+1}/{num_diffusion_steps}, Beta: {variance_schedule_beta[i]:.4f}")
        current_noisy_image, _ = add_gaussian_noise(current_noisy_image, variance_schedule_beta, i)

        # Display the noisy image at each step
        axes[i+1].imshow(current_noisy_image, cmap='gray', vmin=0, vmax=255)
        axes[i+1].set_title(f"Step {i+1}")
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.show()

    print("\nForward diffusion simulation complete.")
    print("The images above show the original image gradually becoming more noisy.")

if __name__ == "__main__":
    main()