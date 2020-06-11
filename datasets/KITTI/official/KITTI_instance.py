from datasets.Loader import register_dataset
from datasets.Mapillary.MapillaryLike_instance import MapillaryLikeInstanceDataset
from datasets.util.Util import username

DEFAULT_PATH = "C:/Users/Tunar Mahmudov/Desktop/TrackR-CNN/data/KITTI_MOTS"
NAME = "KITTI_instance"


@register_dataset(NAME)
class KittiInstanceDataset(MapillaryLikeInstanceDataset):
  def __init__(self, config, subset):
    super().__init__(config, subset, NAME, DEFAULT_PATH, "datasets/KITTI/official", 256,
                     cat_ids_to_use=list(range(24, 34)))
