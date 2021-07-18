from .coco import CocoDataset
from .builder import DATASETS
@DATASETS.register_module()
class ShipRSImageNet_Level1(CocoDataset):
    CLASSES = ('Other Ship', 'Warship', 'Merchant', 'Dock',)

