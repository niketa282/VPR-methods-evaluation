import torch
import pynvml

def measure_inference_memory(model, images):
    torch.cuda.synchronize()

    # Memory already allocated before inference
    baseline_memory = torch.cuda.memory_allocated()

    # Start a fresh peak-memory measurement
    torch.cuda.reset_peak_memory_stats()

    # Run inference
    _ = model(images)

    # Wait until GPU inference has finished
    torch.cuda.synchronize()

    # Highest amount of allocated GPU memory during inference
    peak_memory = torch.cuda.max_memory_allocated()

    # Additional memory required by this forward pass
    extra_memory = peak_memory - baseline_memory

    # Convert bytes to MB
    peak_memory_mb = peak_memory / (1024 ** 2)
    extra_memory_mb = extra_memory / (1024 ** 2)

    return peak_memory_mb, extra_memory_mb

def measure_inference_latency(model, images):
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    descriptors = model(images)
    end_event.record()
    
    # Wait until the GPU has actually finished the forward pass
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    
    return descriptors, elapsed_ms


def measure_energy_consumption(model, images):
    energy_iterations = 20
    pynvml.nvmlInit()

    # Get a reference to a GPU
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    torch.cuda.synchronize()
     # Read energy BEFORE benchmark
    energy_before = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

        # Run 200 measured forward passes
    for _ in range(energy_iterations):
        
         _ = model(images)

        # Wait until all 200 have actually completed
    torch.cuda.synchronize()

        # Read energy AFTER benchmark
    energy_after = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    energy_used_mJ = energy_after - energy_before
    energy_used_J = energy_used_mJ / 1000
    energy_per_image_J = energy_used_J / (
    energy_iterations * images.size(0)
  )
    return energy_per_image_J