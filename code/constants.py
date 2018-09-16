from pathlib import Path

MAC_DATA_DIR = Path('/Users/shleifer/rsna/')
if MAC_DATA_DIR.exists():
    DATA_DIR = MAC_DATA_DIR
else:
    raise NotImplementedError('Linux paths not implemented')


det_class_path = DATA_DIR / 'stage_1_detailed_class_info.csv'
dicom_dir = DATA_DIR / 'stage_1_train_images/'
test_dicom_dir = DATA_DIR / 'stage_1_test_images/'
bbox_path = DATA_DIR / 'stage_1_train_labels.csv'

