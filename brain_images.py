# Data Source https://www.kaggle.com/mushfirat/brain-tumor-classification-accuracy-97/data

# https://www.kaggle.com/masoudnickparvar/brain-tumor-mri-dataset/download


import shutil
import tqdm
import os 
import cv2
import imutils

DATA_URL =' https://www.kaggle.com/masoudnickparvar/brain-tumor-mri-dataset/download'
ZIP_PATH = './brain_tumor_mri_dataset.zip'
# Download the zip file from the data url and extract it

def extract_data(zip_path, extract_path):
    import zipfile
    import shutil
    os.makedirs(extract_path, exist_ok=True)
    zip_ref = zipfile.ZipFile(zip_path, 'r')
    zip_ref.extractall(extract_path)
    zip_ref.close()

def crop_image(image, add_pixels_value=0):
    # taken from https://www.kaggle.com/ruslankl/brain-tumor-detection-v1-0-cnn-vgg-16


        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = image[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        return new_img

def organize_data(data_path, output_path):
    # Create the output directory if it doesn't exist
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path, exist_ok=True)
    # Get the list of all the files in the data directory
    #all_files = [os.path.join(path, name) for name in files for (path, subdirs, files in os.walk(data_path))] 
    data_path_train = os.path.join(data_path, 'Training')
    data_path_test = os.path.join(data_path, 'Testing')
    
    for subset_path in [data_path_train, data_path_test]:
        if subset_path==data_path_train:
            train = True
        else:
            train = False
        for path, subdirs, files in os.walk(subset_path):
            for name in files:
                print("Processing {}".format(name))
                file = os.path.join(path, name)
                print(subdirs)
                if file.endswith('.jpg') and 'notumor' not in path:
                    original_image = cv2.imread(file)
                    
                    if(train):
                        lr_path = os.path.join(output_path, 'LR_train')
                        hr_path = os.path.join(output_path, 'HR_train')
                    else:
                        lr_path = os.path.join(output_path, 'LR_test')
                        hr_path = os.path.join(output_path, 'HR_test')

                    os.makedirs(hr_path, exist_ok=True)
                    os.makedirs(lr_path, exist_ok=True)
                    


                    
                    original_image= crop_image(original_image)

                    base_size = 512
                    scales= [2,3,4]
                    save_name = name.replace('.jpg', '.png')
                    
                    # Make hires version of the image
                    print("Saving {}".format(os.path.join(hr_path, save_name)))
                    image = cv2.resize(original_image, (512, 512))
                    cv2.imwrite(os.path.join(hr_path, save_name), image)
                    
                    #Read in the hires image just incase it is different
                    hires_image = cv2.imread(os.path.join(hr_path, save_name))

                    # Make low res versions of the image
                    for scale in scales:
                        image = cv2.resize(hires_image, (int(base_size/scale), int(base_size/scale)), interpolation=cv2.INTER_CUBIC)
                        assert(image.dtype == hires_image.dtype)
                        lr_scale_path = os.path.join(lr_path, 'X' + str(scale))
                        scale_save_name = save_name.replace('.png', 'x' + str(scale) + '.png')
                        print("Saving {}".format(os.path.join(lr_scale_path, scale_save_name))) 
                        os.makedirs(lr_scale_path, exist_ok=True)
                        cv2.imwrite(os.path.join(lr_scale_path, scale_save_name), image)
                    
                
    
    
    
    
    

def main():
    extract_data(ZIP_PATH, './dataset/BrainTumorRaw')
    organize_data('./dataset/BrainTumorRaw', './dataset/BrainTumor')


if __name__ == '__main__':
    main()
        