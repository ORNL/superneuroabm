"""
diehl_cook_mnist -- MNIST, the Poisson input encoding and the spike-count readout of
Diehl & Cook (2015), for tutorial 04.

* ``load_mnist(split)`` downloads the four MNIST archives once into ``data/mnist/`` next
  to this file (11 MB, from torchvision's mirror) and returns ``(images, labels)``:
  uint8 ``[N, 784]`` and ``[N]``. The IDX format is read directly, no torchvision needed.
* ``poisson_input(image, intensity, rng)`` encodes one image the way Brian's
  ``PoissonGroup`` does in the authors' code: pixel ``p`` fires at ``p / 8 * intensity``
  Hz, independently on every tick of the 350 ms presentation. Returns the
  ``(pixel, tick)`` pair of every spike.
* ``assign_labels`` / ``classify`` are the authors' readout: each excitatory neuron is
  labelled with the class it responds to most on average, and a digit is classified by
  the class whose neurons respond most on average.
"""

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data" / "mnist"

# yann.lecun.com refuses unauthenticated downloads now; these are torchvision's mirrors.
MNIST_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

N_INPUT = 784
N_CLASSES = 10
DT = 0.5e-3                # the authors' time step
PRESENT_TICKS = 700        # single_example_time = 350 ms
INTENSITY_DIVISOR = 8.0    # rate = pixel / 8 * intensity Hz: 63.75 Hz at intensity 2


def _download(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    for fname in MNIST_FILES.values():
        dest = data_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            continue
        for mirror in MNIST_MIRRORS:
            try:
                print(f"downloading {fname} from {mirror}")
                urllib.request.urlretrieve(mirror + fname, dest)
                break
            except Exception as exc:  # try the next mirror
                dest.unlink(missing_ok=True)
                error = exc
        else:
            raise RuntimeError(f"could not download {fname}: {error}")


def _read_idx(path):
    with gzip.open(path, "rb") as fh:
        magic, count = struct.unpack(">II", fh.read(8))
        if magic == 2051:          # images
            rows, cols = struct.unpack(">II", fh.read(8))
            return np.frombuffer(fh.read(count * rows * cols), dtype=np.uint8).reshape(count, rows * cols)
        if magic == 2049:          # labels
            return np.frombuffer(fh.read(count), dtype=np.uint8)
        raise ValueError(f"{path}: unrecognised IDX magic {magic}")


def load_mnist(split, data_dir=DATA_DIR):
    """``(images uint8 [N, 784], labels uint8 [N])`` for ``'train'`` (60,000) or ``'test'`` (10,000)."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    _download(Path(data_dir))
    images = _read_idx(Path(data_dir) / MNIST_FILES[f"{split}_images"])
    labels = _read_idx(Path(data_dir) / MNIST_FILES[f"{split}_labels"])
    assert images.shape == ((60000 if split == "train" else 10000), N_INPUT)
    return images, labels


def poisson_input(image, intensity, rng, n_ticks=PRESENT_TICKS, dt=DT):
    """The ``(pixel, tick)`` of every input spike of one presentation.

    Pixel ``p`` fires with probability ``p / 8 * intensity * dt`` on each of the
    ``n_ticks`` ticks, at most once per tick. Ticks are presentation-relative.
    """
    p = np.clip(image.astype(np.float64) / INTENSITY_DIVISOR * intensity * dt, 0.0, 1.0)
    return np.nonzero(rng.random((N_INPUT, n_ticks)) < p[:, None])


def assign_labels(responses, labels, n_classes=N_CLASSES):
    """The class each neuron responds to most strongly *on average*.

    ``responses`` is ``[n_digits, n_neurons]`` spike counts. Averaging per class (rather
    than summing) keeps a frequent class from capturing more neurons by count alone.
    """
    responses = np.asarray(responses, dtype=np.float64)
    rates = np.zeros((n_classes, responses.shape[1]))
    for c in range(n_classes):
        if (labels == c).any():
            rates[c] = responses[labels == c].mean(axis=0)
    return np.argmax(rates, axis=0)


def classify(responses, assignments, n_classes=N_CLASSES):
    """Per digit, the class whose neurons respond most on average."""
    responses = np.asarray(responses, dtype=np.float64)
    scores = np.zeros((responses.shape[0], n_classes))
    for c in range(n_classes):
        members = np.flatnonzero(assignments == c)
        if members.size:
            scores[:, c] = responses[:, members].mean(axis=1)
    return np.argmax(scores, axis=1)
