# An Action-Conditioned World Model for a Soft Robot: From CAD to Neural Simulation

This project implements an action-conditioned world model for a soft robotic manipulator. It demonstrates a complete end-to-end design pipeline: moving from a raw CAD file, to high-fidelity finite-element (FEM) simulation in ANSYS, and finally into a fast neural simulator that predicts the deformation of the robot directly from its pressure inputs.

## Project Overview

* **The Robot:** A custom soft robotic manipulator built from 3 inflatable bellows arranged in a triangle, so that the 3 pressure inputs bend it like a continuum robot.
* **The Dataset:** Video data collected from ANSYS finite-element simulations. Each simulation is rendered from 4 calibrated viewpoints (3 side views 90 degrees apart and 1 top view). Pressure profiles span 0 to 100,000 Pa in 25,000 Pa steps and include ramp-up / pause / ramp-down sequences, staircases and random walks, which expose the nonlinear hysteresis of the pneumatic actuator.
* **The World Model:** A 2D neural simulator that autoregressively predicts the deformation of the soft robot conditioned on the 3D pressure action.

| Real-Life Hardware | ANSYS Simulation Environment |
| :---: | :---: |
| <img src="assets/3bellow_real.png" width="200"> | <img src="assets/3bellow_sim.png" width="200"> |

## Technical Architecture

* **Encoder / Decoder:** A lightweight convolutional encoder maps each 128x128 silhouette to a 32x32 spatial latent. A decoder with residual blocks maps the predicted latent back to a mask.
* **Autoregressive Dynamics:** A Convolutional GRU (ConvGRU) evolves the spatial latent over time. The 3D pressure action is embedded by an MLP, broadcast spatially and fed into the GRU together with the current latent. Dropout on the hidden state pushes the model to rely on the action rather than on memory alone.
* **Inverse-Dynamics Head:** An auxiliary head predicts the last 5 actions from the current and next latent. This forces the latent to encode the recent pressure history, which is what makes the hysteresis learnable from a single frame.
* **Training Scheme:** Each sequence starts with a short teacher-forced burn-in (5 frames), or a cold start from a single frame 30% of the time, followed by strict autoregression where every predicted latent is fed back into the model. The prediction horizon grows through the run (4, then 11, then 19 steps ahead) so the model learns short-term dynamics before long rollouts. Training windows are 24 frames at a temporal stride of 2.
* **Multi-Objective Loss:** BCE + Dice for sharp mask boundaries, plus an MSE inverse-action loss. Gradient clipping and a cosine learning-rate schedule keep training stable.

The whole model has roughly 0.5M parameters.

## Neural Simulation Results

Below are samples of the 2D world model predicting the bending dynamics of the soft robot over time. Every validation rollout uses a 5-frame burn-in and then predicts the full sequence blind.

### Inference Speed

A full 10-second deformation sequence takes 10 to 50 minutes of ANSYS simulation depending on the pressure input and on self-contact under extreme bending. The world model rolls out the same sequence in about 0.1 seconds on an Apple M4 Max (over 1000 predicted frames per second), a speedup of roughly four orders of magnitude.

### Validation Set (Unseen Ground Truth Comparison)

| &nbsp; | &nbsp; |
| :---: | :---: |
| **Validation Case 1**<br><video src="https://github.com/user-attachments/assets/c1271794-bbf2-4f06-86bd-ed9205048f92" autoplay loop muted playsinline width="100%"></video><br>*Unseen Validation Target 1* | **Validation Case 2**<br><video src="https://github.com/user-attachments/assets/37669af7-841a-41a1-9778-121285f71a94" autoplay loop muted playsinline width="100%"></video><br>*Unseen Validation Target 2* |
| **Validation Case 3**<br><video src="https://github.com/user-attachments/assets/f29866b0-1955-4b43-979b-a13ecaa47461" autoplay loop muted playsinline width="100%"></video><br>*Unseen Validation Target 3* | **Validation Case 4**<br><video src="https://github.com/user-attachments/assets/b5e3e15c-48aa-4f7c-aed2-a6b9f53a9ee5" autoplay loop muted playsinline width="100%"></video><br>*Unseen Validation Target 4* |
| **Validation Case 5**<br><video src="https://github.com/user-attachments/assets/ce0c52d2-1f60-4804-b397-1b367bfbef7b" autoplay loop muted playsinline width="100%"></video><br>*Unseen Validation Target 5* | **Validation Case 6**<br><video src="https://github.com/user-attachments/assets/c776879d-8a44-4862-a397-9331a11b3949" autoplay loop muted playsinline width="100%"></video><br>*Unseen Validation Target 6* |

### Custom Action Sequences (Pure Generation)
*These sequences have no ground truth reference. The model generates the physics purely from the input pressure actions.*

| &nbsp; | &nbsp; | &nbsp; |
| :---: | :---: | :---: |
| **Custom Sim 1**<br><video src="https://github.com/user-attachments/assets/a3e3e04d-79a6-4ebc-8050-7c9a23e1c209" autoplay loop muted playsinline width="100%"></video><br>*No Ground Truth Reference* | **Custom Sim 2**<br><video src="https://github.com/user-attachments/assets/ea8fab15-a49e-4677-8e87-88f186a31ec9" autoplay loop muted playsinline width="100%"></video><br>*No Ground Truth Reference* | **Custom Sim 3**<br><video src="https://github.com/user-attachments/assets/7f22e276-d2da-4df8-a454-059eb4303f72" autoplay loop muted playsinline width="100%"></video><br>*No Ground Truth Reference* |

## Usage

Run scripts as modules from the repository root so that the `src` imports resolve:

```bash
python train_2d.py
python -m src.inference.inference_2d_validation
```

## Notes

- Supports up to Python 3.12 since Open3D isn't supported by newer versions.