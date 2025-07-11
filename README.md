# Diffusion Models In AI Imaging
This project simulates the forward and reverse diffusion process in generative AI imaging. There are two Python programs, which demonstrates the <b>"forward diffusion" (noising)</b> process and <b>"reverse diffusion" (denoising)</b> process.

-----------

## Forward Diffusion
<b>Filename: <i>forward_diffusion.py</i></b>

Creates a Simple Image: It generates a 64x64 pixel grayscale image with a gradient, which serves as our "clean" starting point.

Defines a Variance Schedule: A <i>variance_schedule_beta</i> array is created. These values determine the amount of Gaussian noise added at each step. In a real diffusion model, this schedule is carefully designed.

<i>add_gaussian_noise</i> Function: This function takes an image array, the <i>variance_schedule_beta</i>, and the current step number. It generates random Gaussian noise with a variance determined by <i>beta_t</i> for the current step and adds it to the image. The noise is scaled up slightly for better visual effect.

Simulates Steps: The main function iteratively calls <i>add_gaussian_noise</i>, applying noise progressively.

Visualizes: It uses matplotlib to display the original image and the noisy image at each step, allowing you to see how the image gradually degrades into pure noise.

When you run this code, you'll see a series of images, starting with a clear gradient and becoming increasingly corrupted by noise with each subsequent image. This visually represents how the forward diffusion process systematically "destroys" the original data by adding noise.

--------------

## Reverse Diffusion
<b>Filename: <i>reverse_diffusion.py</i></b>

<b>Important Note:</b> In a real diffusion model, the predict_noise function would be a complex neural network (like a U-Net) that has been trained to accurately estimate the noise present in an image at a given timestep. For this simplified example, we're simulating that by knowing the original image and the noise schedule, allowing us to conceptually "remove" noise. This is a simplification to illustrate the iterative denoising idea without needing to train a full model.

The <i>simulate_noise_predictor</i> is the most important part to understand. In a real diffusion model, this function would be a sophisticated neural network (trained during the "training process"). It would take the <i>noisy_image_array</i> and the timestep as input and output its best prediction of the noise that needs to be removed.

We pass in the <i>original_image_array</i> (which a real model wouldn't have access to during generation) to calculate the actual noise that was added. This allows the denoising to work perfectly in this simplified example, illustrating the concept.

It takes an <i>initial_noisy_image</i> (which we generate by running the forward process to its end for a clear starting point for the demo), the <i>num_diffusion_steps</i>, the <i>variance_schedule_beta</i>, and the <i>original_image_for_sim</i> (our "cheat" for the noise predictor).

It iterates backward from the <i>num_diffusion_steps</i> - 1 down to 0.

In each step:

It calls <i>simulate_noise_predictor</i> to get the "predicted" noise for the current noisy image and timestep.

It then subtracts this predicted noise from the current_image. The exact mathematical formula for this subtraction is derived from the forward diffusion process, but here it's simplified to <i>current_image</i> - <i>predicted_noise</i> * 50 to reverse the scaling applied during noising.

The current_image is updated to this slightly cleaner version.

The image at each denoising step is stored for visualization.

The main function runs the <i>add_gaussian_noise_for_forward_sim</i> function multiple times to create a very noisy image, which serves as our starting point for the reverse diffusion. This simulates the "pure noise" image (xT) that a real diffusion model would start with.

Then, it calls <i>reverse_diffusion</i> with this highly noisy image.

Finally, it plots the series of images, showing the transformation from a noisy image back to a clear gradient.

When you run this code, you'll see the highly noisy image gradually become clearer and clearer, eventually revealing the original gradient pattern. This visually represents the iterative denoising process that is at the heart of diffusion model generation.
