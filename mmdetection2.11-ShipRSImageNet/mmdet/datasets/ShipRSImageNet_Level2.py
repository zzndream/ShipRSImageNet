from .coco import CocoDataset
from .builder import DATASETS
@DATASETS.register_module()
class ShipRSImageNet_Level2(CocoDataset):
    CLASSES = ('Other Ship', 'Other Warship', 'Submarine', 'Aircraft Carrier', 'Cruiser', 'Destroyer',
                        'Frigate', 'Patrol', 'Landing', 'Commander', 'Auxiliary Ships', 'Other Merchant',
                        'Container Ship', 'RoRo', 'Cargo', 'Barge', 'Tugboat', 'Ferry', 'Yacht',
                        'Sailboat', 'Fishing Vessel', 'Oil Tanker', 'Hovercraft', 'Motorboat', 'Dock',)

