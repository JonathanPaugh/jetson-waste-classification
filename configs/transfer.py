from typing import Tuple


FEATURE_EXTRACTOR_MAP = {
  299: 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5',
  224: 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5',
  192: 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_192/feature_vector/5',
  128: 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_128/feature_vector/5',
  96: 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/5',
}

def config(size: int) -> Tuple[str, Tuple[int, int]]:
    assert size in FEATURE_EXTRACTOR_MAP
    return FEATURE_EXTRACTOR_MAP[size], (size, size)
