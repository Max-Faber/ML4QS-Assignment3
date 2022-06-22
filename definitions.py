import os

ROOT_DIR: str = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
main_activity_label_classes: dict[str, int] = {
    'label:UNKNOWN': 0,
    'label:LYING_DOWN': 1,
    'label:SITTING': 2,
    'label:OR_standing': 3,
    'label:FIX_walking': 4,
    'label:FIX_running': 5,
    'label:BICYCLING': 6
}
