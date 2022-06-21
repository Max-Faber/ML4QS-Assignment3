import os

ROOT_DIR: str = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
main_activity_label_classes: dict[str, int] = {
    'label:UNKNOWN': -1,
    'label:LYING_DOWN': 0,
    'label:SITTING': 1,
    'label:OR_standing': 2,
    'label:FIX_walking': 3,
    'label:FIX_running': 4,
    'label:BICYCLING': 5
}
