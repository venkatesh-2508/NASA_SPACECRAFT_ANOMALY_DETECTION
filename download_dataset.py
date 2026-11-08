"""
NASA Spacecraft Anomaly Detection
Script: download_dataset.py

Downloads the real NASA SMAP and MSL telemetry dataset (82 channels)
from the verified HuggingFace repository mirror of Telemanom.
"""
import argparse
import concurrent.futures
import sys
import time
from pathlib import Path
import urllib.request
import urllib.error

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("dataset_downloader")

BASE_HF_URL = "https://huggingface.co/datasets/appleparan/telemanom/resolve/main/data/data"
LABELS_URL = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"

# 82 channels in the official NASA SMAP & MSL benchmark
ALL_CHANNELS = [
    # SMAP (55 channels)
    "A-1", "A-2", "A-3", "A-4", "A-5", "A-6", "A-7", "A-8", "A-9",
    "B-1",
    "D-1", "D-2", "D-3", "D-4", "D-5", "D-6", "D-7", "D-8", "D-9", "D-11", "D-12", "D-13",
    "E-1", "E-2", "E-3", "E-4", "E-5", "E-6", "E-7", "E-8", "E-9", "E-10", "E-11", "E-12", "E-13",
    "F-1", "F-2", "F-3",
    "G-1", "G-2", "G-3", "G-4", "G-6", "G-7",
    "P-1", "P-2", "P-3", "P-4", "P-7",
    "R-1",
    "S-1",
    "T-1", "T-2", "T-3",
    # MSL (27 channels)
    "C-1", "C-2",
    "D-14", "D-15", "D-16",
    "F-4", "F-5", "F-7", "F-8",
    "M-1", "M-2", "M-3", "M-4", "M-5", "M-6", "M-7",
    "P-10", "P-11", "P-14", "P-15",
    "S-2",
    "T-4", "T-5", "T-8", "T-9", "T-12", "T-13"
]

CORE_BENCHMARK_CHANNELS = [
    "P-1", "S-1", "E-1", "E-2", "D-1", "F-1", "G-1", "A-1",  # SMAP
    "M-1", "M-2", "C-1", "T-4", "F-4", "P-10"                # MSL
]


def download_file(url: str, dest_path: Path, max_retries: int = 3) -> bool:
    """Download single file with retries."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "NASA-Spacecraft-Anomaly-Detection/1.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, "wb") as out_file:
                out_file.write(response.read())
            return True
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Failed to download {url}: {e}")
                if dest_path.exists():
                    dest_path.unlink()
                return False
            time.sleep(1)
    return False


def download_channel(channel_id: str, data_dir: Path) -> tuple[str, bool]:
    """Download both train and test files for a channel."""
    train_dest = data_dir / "train" / f"{channel_id}.npy"
    test_dest = data_dir / "test" / f"{channel_id}.npy"

    train_url = f"{BASE_HF_URL}/train/{channel_id}.npy"
    test_url = f"{BASE_HF_URL}/test/{channel_id}.npy"

    ok_train = download_file(train_url, train_dest)
    ok_test = download_file(test_url, test_dest)

    return channel_id, (ok_train and ok_test)


def download_dataset(channels: list[str], data_dir: Path, workers: int = 8) -> dict:
    """Download specified channels and labels CSV."""
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train").mkdir(parents=True, exist_ok=True)
    (data_dir / "test").mkdir(parents=True, exist_ok=True)

    # 1. Download labeled_anomalies.csv
    labels_path = data_dir / "labeled_anomalies.csv"
    logger.info("Downloading labeled_anomalies.csv...")
    download_file(LABELS_URL, labels_path)

    # 2. Download channels concurrently
    logger.info(f"Downloading {len(channels)} channels using {workers} concurrent workers...")
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_channel = {
            executor.submit(download_channel, cid, data_dir): cid
            for cid in channels
        }
        for future in concurrent.futures.as_completed(future_to_channel):
            cid, success = future.result()
            results[cid] = success
            status = "✓" if success else "✗"
            logger.info(f"  [{status}] Channel {cid}")

    successful = sum(1 for s in results.values() if s)
    logger.info(f"Download complete: {successful}/{len(channels)} channels available.")
    return results


def main():
    parser = argparse.ArgumentParser(description="Download real NASA SMAP and MSL telemetry data.")
    parser.add_argument("--all", action="store_true", help="Download all 82 NASA channels")
    parser.add_argument("--benchmark", action="store_true", help="Download 14 core benchmark channels (fast)")
    parser.add_argument("--channel", action="append", default=[], help="Specific channel ID(s) to download")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent download threads")
    parser.add_argument("--dest", type=str, default="data", help="Target data directory")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.dest

    if args.all:
        channels_to_download = ALL_CHANNELS
    elif args.channel:
        channels_to_download = args.channel
    else:
        # Default: benchmark channels for quick, rich setup
        channels_to_download = CORE_BENCHMARK_CHANNELS

    logger.info(f"Starting download of {len(channels_to_download)} channels to {data_dir}...")
    download_dataset(channels_to_download, data_dir, workers=args.workers)


if __name__ == "__main__":
    main()

