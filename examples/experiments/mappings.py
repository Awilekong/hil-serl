from .ram_insertion.config import TrainConfig as RAMInsertionTrainConfig
from .usb_pickup_insertion.config import TrainConfig as USBPickupInsertionTrainConfig
from .object_handover.config import TrainConfig as ObjectHandoverTrainConfig
from .egg_flip.config import TrainConfig as EggFlipTrainConfig

CONFIG_MAPPING = {
                "ram_insertion": RAMInsertionTrainConfig,
                "usb_pickup_insertion": USBPickupInsertionTrainConfig,
                "object_handover": ObjectHandoverTrainConfig,
                "egg_flip": EggFlipTrainConfig,
               }