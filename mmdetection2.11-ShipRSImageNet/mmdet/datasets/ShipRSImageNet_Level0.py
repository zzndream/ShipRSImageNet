from .coco import CocoDataset
from .builder import DATASETS
@DATASETS.register_module()
class ShipRSImageNet_Level0(CocoDataset):
    CLASSES = ('Ship', 'Dock',)

