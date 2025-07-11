import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def add_gaussian_noise_for_forward_sim(image_array, variance_schedule_beta, current_step):
    """
    Helper function to generate a noisy image for the starting point of reverse diffusion.
    This is a simplified version of the forward process for demonstration purposes.
    In a real DDPM, x_t can be directly sampled from x_0.
    """
    beta_t = variance_schedule_beta[current_step]
    noise = np.random.normal(0, np.sqrt(beta_t), image_array.shape)
    noisy_image_array = image_array + noise * 50 # Scale noise for visibility
    noisy_image_array = np.clip(noisy_image_array, 0, 255)
    return noisy_image_array, noise

def simulate_noise_predictor(noisy_image_array, timestep, variance_schedule_beta, original_image_array):
    """
    SIMULATED noise prediction function.
    In a real diffusion model, this would be a trained neural network
    that predicts the noise (epsilon) given the noisy_image_array and timestep.

    For this example, we're cheating by using the original_image_array to
    calculate the *actual* noise that was added to get to noisy_image_array.
    This allows us to demonstrate denoising without a trained model.

    Args:
        noisy_image_array (np.ndarray): The current noisy image.
        timestep (int): The current timestep in the reverse process.
        variance_schedule_beta (list or np.ndarray): The beta schedule.
        original_image_array (np.ndarray): The clean original image (used for simulation only).

    Returns:
        np.ndarray: The predicted noise.
    """
    # This is a highly simplified and "cheating" way to get the noise.
    # In reality, the neural network would learn to estimate this.
    # We're calculating the noise that would transform original_image_array
    # to noisy_image_array at this specific timestep, scaled by 50 as in forward.
    # A more accurate simulation for this step would involve understanding the
    # exact forward formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    # and then solving for epsilon.
    # For a simple visual demo, we'll just use the difference.

    # Reconstruct the noise based on the original image and the current noisy image
    # and the scaling factor used in the forward process.
    # This is a simplification and not how a real noise predictor works.
    # A real predictor learns to map (noisy_image, t) -> noise_to_remove.
    simulated_noise = (noisy_image_array - original_image_array) / 50
    return simulated_noise


def reverse_diffusion(initial_noisy_image, num_diffusion_steps, variance_schedule_beta, original_image_for_sim=None):
    """
    Simulates the reverse diffusion (denoising) process to construct an image from noise.

    Args:
        initial_noisy_image (np.ndarray): The starting pure noise image (or highly noisy image).
        num_diffusion_steps (int): The total number of steps in the reverse process.
        variance_schedule_beta (list or np.ndarray): The beta values used in the forward process.
        original_image_for_sim (np.ndarray, optional): The original clean image.
                                                        REQUIRED for this *simulated* example
                                                        to allow the 'noise predictor' to work.

    Returns:
        list: A list of image arrays at different denoising steps.
    """
    if original_image_for_sim is None:
        raise ValueError("original_image_for_sim is required for this simulated reverse diffusion.")

    denoised_images = [initial_noisy_image.copy()]
    current_image = initial_noisy_image.copy()

    print("\nSimulating reverse diffusion (denoising):")
    # Iterate backwards through the timesteps
    for i in range(num_diffusion_steps - 1, -1, -1):
        print(f"Denoising step {i+1}/{num_diffusion_steps}")
        timestep = i # Current timestep for the noise predictor

        # 1. Predict the noise that was added at this step
        # In a real model, this is where the trained neural network comes in.
        predicted_noise = simulate_noise_predictor(current_image, timestep, variance_schedule_beta, original_image_for_sim)

        # 2. Remove the predicted noise to get a slightly cleaner image
        # This is a simplified denoising step. In actual DDPMs, the denoising
        # equation is derived from the forward process and involves alpha_t,
        # alpha_bar_t, and the predicted noise.
        denoised_step_image = current_image - predicted_noise * 50 # Reverse the scaling

        # Clip values to ensure they stay within valid image range
        denoised_step_image = np.clip(denoised_step_image, 0, 255)

        current_image = denoised_step_image
        denoised_images.append(current_image.copy())

    return denoised_images


def main():
    image_size = 64
    original_image_array = np.linspace(0, 255, image_size * image_size).reshape(image_size, image_size)
    original_image_array = original_image_array.astype(np.float32)

    num_diffusion_steps = 10
    variance_schedule_beta = np.linspace(0.0001, 0.02, num_diffusion_steps)

    # --- PART 1: Simulate Forward Diffusion to get a very noisy image ---
    # In a real scenario, you'd start with pure random noise, but for this
    # conceptual demo, we'll generate a noisy version of our original image
    # to make the "denoising" visually clear.
    initial_noisy_image_for_reverse = original_image_array.copy()
    for i in range(num_diffusion_steps):
        initial_noisy_image_for_reverse, _ = add_gaussian_noise_for_forward_sim(
            initial_noisy_image_for_reverse, variance_schedule_beta, i
        )
    print("Generated a highly noisy image to start reverse diffusion from.")

    # --- PART 2: Simulate Reverse Diffusion ---
    denoised_images = reverse_diffusion(
        initial_noisy_image_for_reverse,
        num_diffusion_steps,
        variance_schedule_beta,
        original_image_for_sim=original_image_array # Used by the *simulated* noise predictor
    )

    # --- Plotting Results ---
    fig, axes = plt.subplots(1, num_diffusion_steps + 1, figsize=(15, 3))

    # Display the initial noisy image
    axes[0].imshow(denoised_images[0], cmap='gray', vmin=0, vmax=255)
    axes[0].set_title("Initial Noisy Image")
    axes[0].axis('off')

    # Display the denoising steps
    for i in range(1, num_diffusion_steps + 1):
        axes[i].imshow(denoised_images[i], cmap='gray', vmin=0, vmax=255)
        axes[i].set_title(f"Denoised Step {i}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    print("\nReverse diffusion simulation complete.")
    print("The images above show the highly noisy image gradually becoming clearer.")
    print("The final image should resemble the original gradient.")

if __name__ == "__main__":
    main()