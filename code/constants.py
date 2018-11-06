from pathlib import Path

MAC_DATA_DIR = Path('/Users/shleifer/rsna/')
if MAC_DATA_DIR.exists():
    DATA_DIR = MAC_DATA_DIR
else:
    DATA_DIR = Path('/home/paperspace/fastai/courses/dl2/kaggle/rsna_data/')
    raise NotImplementedError('Linux paths not implemented')


det_class_path = DATA_DIR / 'stage_1_detailed_class_info.csv'
dicom_dir = DATA_DIR / 'stage_1_train_images/'
test_dicom_dir = DATA_DIR / 'stage_1_test_images/'
bbox_path = DATA_DIR / 'stage_1_train_labels.csv'


exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg',
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg'] #corrupted images


