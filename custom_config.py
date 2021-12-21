from mrcnn.config import Config

class CustomConfig(Config):
    def __init__(self, num_classes):
        self.NUM_CLASSES = num_classes + 1
        super().__init__()
    NAME = "object"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    STEPS_PER_EPOCH = 500
    VALIDATION_STEPS = 5
    DETECTION_MIN_CONFIDENCE = 0.9


class InferenceConfig(CustomConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1