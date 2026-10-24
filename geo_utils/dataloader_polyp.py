import os
import random
import hashlib
import multiprocessing as mp
import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


SUPPORTED_AUGMENTATION_SEED_MODES = {"legacy_v1", "deterministic_v3"}
DETERMINISTIC_V3_ALBUMENTATIONS_VERSION = "2.0.8"


class DeterministicEpochRandomSampler(data.Sampler[int]):
    """A worker-independent permutation derived from ``(seed, epoch, length)``.

    Unlike a mutable ``torch.Generator`` shared with DataLoader worker seeding,
    this sampler produces the same epoch order after a restart and across the
    persistent-worker setting.
    """

    def __init__(self, data_source, seed: int):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _epoch_seed(self) -> int:
        payload = (
            f"emcad-shuffle-v3|{self.seed}|{self.epoch}|{len(self.data_source)}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.blake2s(payload, digest_size=8).digest(), "little")

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self._epoch_seed())
        return iter(torch.randperm(len(self.data_source), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.data_source)


class PolypDataset(data.Dataset):
    """
    Unified adaptive dataloader for polyp segmentation.
    Uses Albumentations and strictly handles binary mask conversion.
    """
    def __init__(
        self,
        image_root,
        gt_root,
        trainsize,
        augmentation,
        split='train',
        color_image=True,
        augmentation_seed_mode="legacy_v1",
        augmentation_seed=None,
        augmentation_library_version=None,
        multiprocessing_context=None,
    ):
        self.trainsize = trainsize
        self.color_image = color_image
        self.augmentation = augmentation
        self.split = split
        self.augmentation_seed_mode = str(augmentation_seed_mode)
        if self.augmentation_seed_mode == "deterministic_v2":
            raise ValueError("deterministic_v2 is unsupported; use deterministic_v3")
        if self.augmentation_seed_mode not in SUPPORTED_AUGMENTATION_SEED_MODES:
            raise ValueError(
                "Unsupported augmentation_seed_mode: "
                f"{self.augmentation_seed_mode}; expected one of "
                f"{sorted(SUPPORTED_AUGMENTATION_SEED_MODES)}"
            )
        if self.augmentation_seed_mode == "deterministic_v3" and augmentation_seed is None:
            raise ValueError("deterministic_v3 augmentation requires a non-null augmentation_seed")
        self.augmentation_seed = None if augmentation_seed is None else int(augmentation_seed)
        self.augmentation_library_version = (
            None if augmentation_library_version is None else str(augmentation_library_version)
        )
        if self.augmentation_seed_mode == "deterministic_v3":
            if self.augmentation_library_version != DETERMINISTIC_V3_ALBUMENTATIONS_VERSION:
                raise RuntimeError(
                    "deterministic_v3 requires train.augmentation_library_version="
                    f"{DETERMINISTIC_V3_ALBUMENTATIONS_VERSION!r}, got "
                    f"{self.augmentation_library_version!r}"
                )
            if A.__version__ != self.augmentation_library_version:
                raise RuntimeError(
                    "deterministic_v3 Albumentations version mismatch: expected "
                    f"{self.augmentation_library_version!r}, running {A.__version__!r}"
                )
        # Shared across persistent workers, avoiding dependence on worker assignment
        # or prefetch scheduling when a new epoch begins.
        epoch_context = (
            mp.get_context(multiprocessing_context)
            if isinstance(multiprocessing_context, str)
            else (multiprocessing_context or mp.get_context())
        )
        self._augmentation_epoch = epoch_context.Value("q", 0, lock=True)
        
        # Load and sort file paths
        exts = ('.jpg', '.png', '.jpeg', '.tif')
        self.images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.lower().endswith(exts)])
        self.gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.lower().endswith(exts)])
        
        self.filter_files()
        self.size = len(self.images)

        # Transformation Setup
        mean = [0.485, 0.456, 0.406] if color_image else [0.5]
        std = [0.229, 0.224, 0.225] if color_image else [0.229]

        if self.split == 'train' and self.augmentation:
            self.transform = A.Compose([
                A.Rotate(limit=90, p=0.5),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Resize(height=self.trainsize, width=self.trainsize),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=self.trainsize, width=self.trainsize),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])

        if self.augmentation_seed_mode == "deterministic_v3" and not callable(
            getattr(self.transform, "set_random_state", None)
        ):
            raise RuntimeError(
                "Albumentations Compose must provide set_random_state for deterministic_v3"
            )

    def set_epoch(self, epoch: int) -> None:
        """Set the shared epoch used by deterministic_v3 augmentation."""
        with self._augmentation_epoch.get_lock():
            self._augmentation_epoch.value = int(epoch)

    def augmentation_seed_for_index(self, index: int) -> int | None:
        """Return a stable uint64 seed for one `(base seed, epoch, sample)` tuple."""
        if self.augmentation_seed_mode != "deterministic_v3" or not self.augmentation:
            return None
        with self._augmentation_epoch.get_lock():
            epoch = int(self._augmentation_epoch.value)
        sample_id = os.path.basename(self.images[index])
        payload = f"emcad-augmentation-v3|{self.augmentation_seed}|{epoch}|{index}|{sample_id}".encode(
            "utf-8"
        )
        return int.from_bytes(hashlib.blake2s(payload, digest_size=8).digest(), "little")

    def filter_files(self):
        valid_images, valid_gts = [], []
        for img_p, gt_p in zip(self.images, self.gts):
            if os.path.exists(img_p) and os.path.exists(gt_p):
                valid_images.append(img_p)
                valid_gts.append(gt_p)
        self.images, self.gts = valid_images, valid_gts

    def __getitem__(self, index):
        # 1. Load Image
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB if self.color_image else cv2.COLOR_BGR2GRAY)
        
        # 2. Load Mask
        mask_np = cv2.imread(self.gts[index], cv2.IMREAD_GRAYSCALE)

        # 3. Seed Albumentations per sample for deterministic worker scheduling.
        sample_seed = self.augmentation_seed_for_index(index)
        if sample_seed is not None:
            self.transform.set_random_state(
                np.random.default_rng(sample_seed),
                random.Random(sample_seed),
            )
        augmented = self.transform(image=image, mask=mask_np)
        image = augmented['image']
        mask = augmented['mask']

        # 4. Normalize masks encoded as 0/255 or integer labels.
        max_val = mask.max()
        if max_val > 127.0:
            # Treats everything above 20 as foreground to catch 255 but ignore noise
            mask = (mask > 20).long()
        else:
            # Treats all integer labels (1, 2, 3...) as foreground
            mask = (mask >= 1).long()

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        # 5. Return Logic
        if self.split == 'train':
            return image, mask
        else:
            with Image.open(self.gts[index]) as img:
                original_shape = img.size
            name = os.path.basename(self.images[index])
            if name.lower().endswith('.jpg'):
                name = name.rsplit('.', 1)[0] + '.png'
            return image, mask, original_shape, name

    def __len__(self):
        return self.size

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_loader(
    image_root,
    gt_root,
    batchsize,
    trainsize,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    drop_last=False,
    augmentation=False,
    split='train',
    color_image=True,
    seed=None,
    augmentation_seed_mode="legacy_v1",
    augmentation_library_version=None,
    multiprocessing_context=None,
):
    dataset = PolypDataset(
        image_root,
        gt_root,
        trainsize,
        augmentation,
        split,
        color_image,
        augmentation_seed_mode=augmentation_seed_mode,
        augmentation_seed=seed,
        augmentation_library_version=augmentation_library_version,
        multiprocessing_context=multiprocessing_context,
    )
    worker_generator = None
    sampler = None
    worker_init_fn = None
    if seed is not None:
        base_seed = int(seed) & ((1 << 64) - 1)
        if augmentation_seed_mode == "deterministic_v3":
            # Keep the worker-base-seed stream independent from the epoch sampler.
            # Persistent workers otherwise consume a different number of worker
            # seed draws and can alter later shuffle orders.
            worker_generator = torch.Generator()
            worker_generator.manual_seed(base_seed ^ 0x9E3779B97F4A7C15)
            if shuffle:
                sampler = DeterministicEpochRandomSampler(dataset, base_seed)
        else:
            # Preserve legacy_v1's historic DataLoader generator behavior.
            worker_generator = torch.Generator()
            worker_generator.manual_seed(base_seed)
        worker_init_fn = seed_worker
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
        if multiprocessing_context is not None:
            loader_kwargs["multiprocessing_context"] = multiprocessing_context
    loader = data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=bool(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=worker_generator,
        worker_init_fn=worker_init_fn,
        **loader_kwargs,
    )
    if augmentation_seed_mode == "deterministic_v3":
        def set_epoch(epoch: int) -> None:
            dataset.set_epoch(epoch)
            if sampler is not None:
                sampler.set_epoch(epoch)

        # The training engine invokes this before constructing each epoch iterator.
        loader.set_epoch = set_epoch
    return loader
    
