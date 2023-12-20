from PIL import Image
import glob
import os
import pandas as pd

PATH = "./DatasetB"
TRAIN_PATH = PATH + '/train'
VALID_PATH = PATH + '/valid'
TEST_PATH = PATH + '/test'
TRAIN_DATA = glob.glob(os.path.join(TRAIN_PATH, '*.jpg'))
TRAIN_LABEL = glob.glob(os.path.join(TRAIN_PATH, '*.csv'))
VALID_DATA = glob.glob(os.path.join(VALID_PATH, '*.jpg'))
VALID_LABEL = glob.glob(os.path.join(VALID_PATH, '*.csv'))
TEST_LABEL = glob.glob(os.path.join(TEST_PATH, '*.csv'))

df_test = pd.read_csv(VALID_LABEL[0])
print(df_test.head())

# Function to crop and save images
def crop_and_save_image(file_path, width, height, xmin, ymin, xmax, ymax, class_name):
    # Open the image file
    pth = PATH + "/valid/" + file_path

    if(os.path.isfile(pth)):
        image = Image.open(pth)
        # Calculate center point
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2

        # Calculate crop box
        crop_x1 = max(0, int(center_x) - 150)  # 150 is half of 300
        crop_y1 = max(0, int(center_y) - 150)
        crop_x2 = min(width, int(center_x) + 150)
        crop_y2 = min(height, int(center_y) + 150)

        # Crop image
        # cropped_image = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        cropped_image = image

        # Resize image to 300x300
        # cropped_image = cropped_image.resize((300, 300), Image.Resampling.LANCZOS)
        cropped_image = cropped_image

        # Create folder if not exists
        class_folder = os.path.join(PATH+'/valid_good', class_name)
        os.makedirs(class_folder, exist_ok=True)
        files_no = len(os.listdir(class_folder))

        # Save cropped image
        cropped_image.save(class_folder+'/'+str(files_no+1)+'_'+class_name+'.jpg')
    else:
        print("No valid images :" + PATH+"/valid/"+file_path)


for index, row in df_test.iterrows():
    crop_and_save_image(row['filename'], row['width'], row['height'], row['xmin'], row['ymin'], row['xmax'], row['ymax'], row['class'])