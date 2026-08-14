import parser
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import faiss
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm
import pynvml

import visualizations
import vpr_models
from test_dataset import TestDataset
from benchmarkingmeasurements import measure_inference_latency, measure_inference_memory, measure_energy_consumption
from calculations import rgb_energy_runs, calculate_mean_and_std

def main(args):
    start_time = datetime.now()

    logger.remove()  # Remove possibly previously existing loggers
    log_dir = Path("logs") / args.log_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S")
    logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "info.log", format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "debug.log", level="DEBUG")
    logger.info(" ".join(sys.argv))
    logger.info(f"Arguments: {args}")
    logger.info(
        f"Testing with {args.method} with a {args.backbone} backbone and descriptors dimension {args.descriptors_dimension}"
    )
    logger.info(f"The outputs are being saved in {log_dir}")

    model = vpr_models.get_model(args.method, args.backbone, args.descriptors_dimension)
    
    if args.input_mode == "grayscale_1ch":
    
        for name, module in model.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                print("First Conv2d:", name, module)
                break
            
        # old_conv = model.backbone[0]
        if hasattr(model.backbone, "model") and hasattr(model.backbone.model, "conv1"):
            print("ENTERING IF")
            old_conv = model.backbone.model.conv1
        else:
            print("ENTERING ELSE")
            old_conv = model.backbone[0]
            


        new_conv = torch.nn.Conv2d(
           in_channels=1, # Creates Conv2d(1 → 64) instead of Conv2d(3 → 64)
           out_channels=old_conv.out_channels,
           kernel_size=old_conv.kernel_size,
           stride=old_conv.stride,
           padding=old_conv.padding,
           bias=False
        )   

        with torch.no_grad():
            new_conv.weight.copy_(
                old_conv.weight.mean(dim=1, keepdim=True) # Combines R-filter weights, G-filter weights,B-filter weights into one grayscale filter
            )
    
       # model.backbone[0] = new_conv
            if hasattr(model.backbone, "model") and hasattr(model.backbone.model, "conv1"):
                print("ENTERING IF")
                model.backbone.model.conv1 = new_conv
            else:
                print("ENTERING ELSE")
                model.backbone[0] = new_conv
    
    model = model.eval().to(args.device)
    
    test_ds = TestDataset(
        args.database_folder,
        args.queries_folder,
        positive_dist_threshold=args.positive_dist_threshold,
        image_size=args.image_size,
        use_labels=args.use_labels,
        input_mode=args.input_mode,
    )
    logger.info(f"Testing on {test_ds}")

    with torch.inference_mode():
        logger.debug("Extracting database descriptors for evaluation/testing")
        num_database = (
                      args.num_database_images
                      if args.num_database_images is not None
                      else test_ds.num_database
                    )
        database_subset_ds = Subset(test_ds, list(range(num_database)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds, num_workers=args.num_workers, batch_size=args.batch_size
        )
        all_descriptors = np.empty((len(test_ds), args.descriptors_dimension), dtype="float32")
        for images, indices in tqdm(database_dataloader):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

        logger.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        num_queries = (
                       args.num_query_images
                       if args.num_query_images is not None
                       else test_ds.num_queries
                      )
        queries_subset_ds = Subset(
            test_ds, list(range(test_ds.num_database, test_ds.num_database + + num_queries))
        )
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers, batch_size=1)
        
        query_latencies_ms = []
        
        warmup_iterations = 20
        
        for batch_idx, (images, indices) in enumerate(tqdm(queries_dataloader)):
            # Move image to GPU BEFORE timing
            images = images.to(args.device)
            # Warm up the GPU using the first query image only
            if batch_idx == 0:
                for _ in range(warmup_iterations):
                    _ = model(images)
                # Make sure all warm-up operations have finished
                torch.cuda.synchronize()

            descriptors, elapsed_ms = measure_inference_latency(model, images)
            query_latencies_ms.append(elapsed_ms)

            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

        # Measure memory in a separate pass so its extra inference does not
        # affect any of the recorded latency measurements.
        query_peak_memory_mb = []
        query_extra_memory_mb = []
        for images, _ in tqdm(queries_dataloader, desc="Measuring query GPU memory"):
            images = images.to(args.device)
            peak_memory_mb, extra_memory_mb = measure_inference_memory(model, images)
            query_peak_memory_mb.append(peak_memory_mb)
            query_extra_memory_mb.append(extra_memory_mb)

    query_latencies_ms = np.array(query_latencies_ms)
    query_peak_memory_mb = np.array(query_peak_memory_mb)
    query_extra_memory_mb = np.array(query_extra_memory_mb)
    
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
  # Take one query batch only for energy measurement
    images, _ = next(iter(queries_dataloader))
    images = images.to(args.device)
    
    energy_j = measure_energy_consumption(
        model,
        images,
        handle
    )
    
    logger.info(
        f"GPU energy consumption: "
        f"{energy_j:.6f} J/image"
    )
    
    
    logger.info(
        "GPU inference memory: "
        f"mean peak={query_peak_memory_mb.mean():.3f} MB, "
        f"max peak={query_peak_memory_mb.max():.3f} MB, "
        f"mean extra forward memory={query_extra_memory_mb.mean():.3f} MB, "
        f"max extra forward memory={query_extra_memory_mb.max():.3f} MB"
    )

    # Print summary statistics
    logger.info(
       f"GPU query inference latency: "
       f"mean={query_latencies_ms.mean():.3f} ms, "
       f"median={np.median(query_latencies_ms):.3f} ms, "
       f"std={query_latencies_ms.std():.3f} ms, "
       f"p95={np.percentile(query_latencies_ms, 95):.3f} ms"
    )     
    
    queries_descriptors = all_descriptors[test_ds.num_database :]
    database_descriptors = all_descriptors[: test_ds.num_database]
    
    if args.save_descriptors:
        logger.info(f"Saving the descriptors in {log_dir}")
        np.save(log_dir / "queries_descriptors.npy", queries_descriptors)
        np.save(log_dir / "database_descriptors.npy", database_descriptors)

    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(args.descriptors_dimension)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors

    logger.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_descriptors, max(args.recall_values))

    # For each query, check if the predictions are correct
    if args.use_labels:
        positives_per_query = test_ds.get_positives()
        recalls = np.zeros(len(args.recall_values))
        for query_index, preds in enumerate(predictions):
            for i, n in enumerate(args.recall_values):
                if np.any(np.isin(preds[:n], positives_per_query[query_index])):
                    recalls[i:] += 1
                    break

        # Divide by num_queries and multiply by 100, so the recalls are in percentages
        recalls = recalls / test_ds.num_queries * 100
        recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
        logger.info(recalls_str)

    # Save visualizations of predictions
    if args.num_preds_to_save != 0:
        logger.info("Saving final predictions")
        # For each query save num_preds_to_save predictions
        visualizations.save_preds(
            predictions[:, : args.num_preds_to_save], test_ds, log_dir, args.save_only_wrong_preds, args.use_labels
        )
        

if __name__ == "__main__":
    args = parser.parse_arguments()
    main(args)