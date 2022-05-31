from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    empty_cuda_cache,
    )
import copy
from craft_text_detector.file_utils import crop_poly, rectify_poly
from DATN_main.embedding_utils import Image_embedding
from PATH import *
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import matplotlib.pyplot as plt



class Image_utils:
    def __init__(self):
        
        self.config = Cfg.load_config_from_name('vgg_transformer')
        self.config['weights'] = WEIGHT_VIETOCR
        # config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
        self.config['cnn']['pretrained']=False
        self.config['device'] = TO_CUDA
        self.config['predictor']['beamsearch']=False
        self.detector = Predictor(self.config)



    def crop_region(self, image, poly, rectify=True):
        """
        Arguments:
            image: part image type: numpy_array
            points: bbox or poly points
            file_path: path to be exported
            rectify: rectify detected polygon by affine transform
        """
        # image = read_image(image)
        

        # deepcopy image so that original is not altered
        image = copy.deepcopy(image)
        if rectify:
            # rectify poly region
            result_rgb = rectify_poly(image, poly)
        else:
            result_rgb = crop_poly(image, poly)

        # export corpped region
        # result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        return result_rgb


    def crop_region_list(self, image, region_list, rectify=True):
        """
        Arguments:
            image: full image
            poly_list: list of poly points: regions = prediction_result["boxes"]
            rectify: rectify detected polygon by affine transform
        """
        # deepcopy image so that original is not altered
        img_crop_list = []
        image = read_image(image)

        image = copy.deepcopy(image)
        for ind, region in enumerate(region_list):
            image = copy.deepcopy(image)

            img_crop =  self.crop_region(image, poly=region , rectify=rectify)
            img_crop_list.append(img_crop)
            # note exported file path
        return img_crop_list

    def recognition_img(self,img):
        img = self.rotate_90_degree(img)
        img = Image.fromarray(img)
        s = self.detector.predict(img)
        return s

    def recognition_180_degree(self,img):
        img = copy.deepcopy(img)
        img = self.rotate_90_degree(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # img_np = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
        s = self.recognition_img(img)


        img_180 = cv2.rotate(img, cv2.ROTATE_180)
        s_180 = self.recognition_img(img_180)
        # print(s)
        return [s, s_180]


    def rotate_90_degree(self, img):
        img_shape = img.shape # (height, width, channel)
        # print("img_shape: ", img_shape)
        img_rotate = copy.deepcopy(img)
        if img_shape[0] > img_shape[1]:
            img_rotate = cv2.rotate(img_rotate, cv2.ROTATE_90_CLOCKWISE)
        return img_rotate



    def plot(self, result_rgb):
        # result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        # DO ẢNH MÀU BÌNH THƯỜNG LÀ RGB, KHI imread hoặc imwrite cv2 tự động đảo kênh màu 
        # lưu lại dưới dạng BGR 
        plt.imshow(result_rgb)
        plt.show() # qua hàm imwrite nó tự đổi từ bgr về rgb 
        # DO ẢNH MÀU BÌNH THƯỜNG LÀ RGB, KHI imread hoặc imwrite cv2 tự động đảo kênh màu 
            # lưu lại dưới dạng BGR 


if __name__ == "__main__":
        img = cv2.imread("DATN_data/outputs/1_acefalgan__(13)/image_crops/crop_14.png")
        image_utils = Image_utils()
        
        print(image_utils.recognition_180_degree(img))



