# VOC Experiments Runner

## Description
Bash script for running VOC (Visual Object Classes) attribution experiments with parallel processing support. The script executes multiple instances of `main.py` with different seeds while managing system resources.

## Requirements
- Python 3.6+
- PyTorch
- CUDA (optional, for GPU acceleration)
- Bash 4.3+ (recommended for better process management)

## Setup
```bash
chmod +x main.sh
```

## Usage

Basic usage:

```bash
./main.sh
```

With specific device and parallel processes:

```bash
processes=4 device=cuda ./main.sh
```

## Parameters
### Environment Variables
+ `processes`: Number of parallel processes (default: 1)
+ `device`: : Computing device to use (default: "cpu")

### Default Parameters
+ `n_steps`: 50 (steps for methods like Integrated Gradients)
+ `seed`: Runs from 12 to 60 in steps of 12
    + `explainers`:
    + geodesic_integrated_gradients
    + input_x_gradient
    + kernel_shap
    + svi_integrated_gradients
    + guided_integrated_gradients
    + integrated_gradients
    + gradient_shap
    + augmented_occlusion
    + occlusion
    + random
    + smooth_grad
## Examples
Run with 4 parallel processes on CPU:

```bash 
processes=4 ./main.sh
```

Run on CUDA GPU with 2 processes:

```bash
processes=2 device=cuda ./main.sh
```

Run single process with specific seed range:
```bash 
./main.sh --seed_start 12 --seed_end 24
```

## Notes
+ The script automatically handles process management and resource allocation
+ Use Ctrl+C to gracefully stop all running processes
+ For systems with Bash < 4.3, the script falls back to a compatible process management method
+ Experiment results are saved in the results directory with timestamp-based organization
