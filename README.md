# An Action-Conditioned World Model for a Soft Robot: From CAD to Differentiable Simulation

This project implements an action-conditioned world model for a soft robotic manipulator. It demonstrates a complete end-to-end design pipeline: moving from a raw CAD file, to high-fidelity finite element simulation (ANSYS), and finally into a fast, differentiable neural simulator.

## Project Overview

* **The Robot:** A custom soft robotic manipulator designed with a combination of 3 inflatable bellows.
* **The Dataset:** Video data collected via ANSYS finite element simulations. The environment captures 4 distinct viewpoints (3 side views and 1 top view) across a pressure control range of 1 Pa to 100,000 Pa, incremented in 25,000 Pa steps, together with random walks.
* **The World Model:** A 2D neural simulator that autoregressively predicts the physical deformation and spatiotemporal dynamics of the soft robot conditioned directly on the pressure actions. 

| Real-Life Hardware | ANSYS Simulation Environment |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/PLACEHOLDER-IMAGE-1" width="400"> | <img src="https://github.com/user-attachments/assets/PLACEHOLDER-IMAGE-2" width="400"> |

## Technical Architecture

To achieve stable, long-horizon physics predictions without collapsing, the model leverages several advanced training methodologies:
* **Autoregressive Dynamics Engine:** Utilizes a Convolutional GRU (ConvGRU) to model temporal physics and inertia across a 24-frame sequence.
* **Curriculum Learning & Scheduled Sampling:** Employs Teacher Forcing decay to gently transition the network from a single-step predictor to a fully unguided, blind autoregressive physics engine.
* **Multi-Objective Loss:** Balances visual fidelity and structural logic using a combination of BCE Loss, Dice Loss (for precise mask boundaries), and Inverse Action Loss.

## Neural Simulation Results

Below are samples of the 2D world model successfully predicting the physical bending dynamics of the soft robot over time. 

| Validation Set | Validation Set | Custom Action Sequence (Pure Generation) |
| :---: | :---: | :---: |
| <video src="https://github.com/user-attachments/assets/PLACEHOLDER-ID-1" autoplay loop muted playsinline width="100%"></video><br>*Unseen Validation Target 1* | <video src="https://github.com/user-attachments/assets/PLACEHOLDER-ID-2" autoplay loop muted playsinline width="100%"></video><br>*Unseen Validation Target 2* | <video src="https://github.com/user-attachments/assets/PLACEHOLDER-ID-3" autoplay loop muted playsinline width="100%"></video><br>*No Ground Truth Reference* |
| <video src="https://github.com/user-attachments/assets/PLACEHOLDER-ID-4" autoplay loop muted playsinline width="100%"></video><br>*Unseen Validation Target 3* | <video src="https://github.com/user-attachments/assets/PLACEHOLDER-ID-5" autoplay loop muted playsinline width="100%"></video><br>*Unseen Validation Target 4* | <video src="https://github.com/user-attachments/assets/PLACEHOLDER-ID-6" autoplay loop muted playsinline width="100%"></video><br>*No Ground Truth Reference* |
| <video src="https://github.com/user-attachments/assets/PLACEHOLDER-ID-7" autoplay loop muted playsinline width="100%"></video><br>*Unseen Validation Target 5* | <video src="https://github.com/user-attachments/assets/PLACEHOLDER-ID-8" autoplay loop muted playsinline width="100%"></video><br>*Unseen Validation Target 6* | <video src="https://github.com/user-attachments/assets/PLACEHOLDER-ID-9" autoplay loop muted playsinline width="100%"></video><br>*No Ground Truth Reference* |

## Notes

- Supports up to Python 3.12 since Open3D isn't supported by newer versions.