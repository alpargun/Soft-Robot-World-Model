import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def generate_signals(time_steps=60, max_p=100.0, min_p=1.0):
    # 1. The Ramp Signal (What you currently have)
    ramp_up = np.linspace(min_p, max_p, time_steps // 2)
    ramp_down = np.linspace(max_p, min_p, time_steps // 2)
    ramp_signal = np.concatenate([ramp_up, ramp_down])
    
    # 2. The PRBS/Random Step Signal (What your advisor wants)
    # Randomly jump to different pressure levels every 10 frames
    random_signal = np.ones(time_steps) * min_p
    for i in range(0, time_steps, 10):
        random_signal[i:i+10] = np.random.uniform(min_p, max_p)
        
    return ramp_signal, random_signal

def plot_fft_comparison():
    time_steps = 60
    fps = 30 # Assuming your 60 frames represent 2 seconds of real time
    T = 1.0 / fps
    
    ramp_signal, random_signal = generate_signals(time_steps)
    
    # Calculate FFTs
    N = time_steps
    xf = fftfreq(N, T)[:N//2]
    
    # Remove the DC offset (mean) so the 0 Hz spike doesn't ruin the plot scale
    ramp_fft = 2.0/N * np.abs(fft(ramp_signal - np.mean(ramp_signal))[0:N//2])
    random_fft = 2.0/N * np.abs(fft(random_signal - np.mean(random_signal))[0:N//2])
    
    # --- PLOTTING ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    # Time Domain Plots
    axs[0, 0].plot(ramp_signal, color='blue', linewidth=2)
    axs[0, 0].set_title('Time Domain: Current Ramp Signal')
    axs[0, 0].set_ylabel('Pressure (kPa)')
    axs[0, 0].set_xlabel('Frame')
    
    # THE FIX IS HERE
    axs[0, 1].plot(random_signal, color='red', linewidth=2, drawstyle='steps-post')
    axs[0, 1].set_title('Time Domain: Persistent Excitation (Random Steps)')
    axs[0, 1].set_xlabel('Frame')
    
    # Frequency Domain (FFT) Plots
    axs[1, 0].plot(xf, ramp_fft, color='blue', linewidth=2)
    axs[1, 0].set_title('Frequency Domain: Ramp FFT')
    axs[1, 0].set_ylabel('Magnitude')
    axs[1, 0].set_xlabel('Frequency (Hz)')
    axs[1, 0].fill_between(xf, ramp_fft, color='blue', alpha=0.2)
    
    axs[1, 1].plot(xf, random_fft, color='red', linewidth=2)
    axs[1, 1].set_title('Frequency Domain: Random Step FFT')
    axs[1, 1].set_xlabel('Frequency (Hz)')
    axs[1, 1].fill_between(xf, random_fft, color='red', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig("FFT_Pressure_Analysis.png", dpi=300)
    print("Saved FFT_Pressure_Analysis.png")
    plt.show()

if __name__ == "__main__":
    plot_fft_comparison()