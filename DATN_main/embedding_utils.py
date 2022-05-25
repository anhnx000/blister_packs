from transformers import BertTokenizer, BertModel
import torch
from PATH import *
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from PATH import *
from sklearn.metrics.pairwise import cosine_similarity




class Word_embedding:
   
    def __init__(self):
        self.tokenizer_bert = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.model_bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.model_bert.eval()
        self.model_bert.to(TO_CUDA)

    def get_embedding(self, word):
        tokenized_text = self.tokenizer_bert(word,  return_tensors='pt')
        tokenized_text.to(TO_CUDA)
        output = self.model_bert(**tokenized_text)
        return output['pooler_output']

     



class Image_embedding():
    RESNET_OUTPUT_SIZES = {
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048,
        "resnet101": 2048,
    }

    def __init__(self, cuda  = True, model = "reset-18", layer = "default", layer_output_size = 512):
        """ Img2Vec
        :param cuda: If set to True, will run forward pass on GPU
        :param model: String name of requested model
        :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
        :param layer_output_size: Int depicting the output size of the requested layer
        """

        self.device = torch.device(TO_CUDA if cuda else "cpu")
        self.layer_output_size = layer_output_size
        self.model_name = model

        self.model, self.extraction_layer = self._get_model_and_layer(model, layer)
        self.model = self.model.to(self.device)

        self.model.eval()

        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])  
        self.to_tensor = transforms.ToTensor()


    def get_vec(self, img, tensor = False):
        """ Get vector embedding from PIL image
        :param img: PIL Image or list of PIL Images
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """
        if type(img)  == list :
            a = [self.normalize(self.to_tensor(self.scaler(im))) for im in img]
            images = torch.stack(a).to(self.device) # tao 1 list Tensor từ các tensor ba đầu
            # thu được tensor lớn hơn, mang ý nghĩa là list của các tesor ban đầu, ta thực hiện tinh toán được trên các tensor ấy

            if self.model_name in ['alexnet', 'vgg']:
                my_embedding = torch.zeros(len(img), self.layer_output_size)
            elif self.model_name == 'densenet' or 'efficientnet' in self.model_name:
                my_embedding = torch.zeros(len(img), self.layer_output_size, 7, 7)
            else:
                my_embedding = torch.zeros(len(img), self.layer_output_size, 1, 1)

            def copy_data(m, i, o):
                my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)

            with torch.no_grad():
                h_x  = self.model(images)
            h.remove()

            if tensor:
                return my_embedding
            else:
                if self.model_name in ['alexnet', 'vgg']:
                    return my_embedding.numpy()[:, :]
                
                elif self.model_name == "densenet" or 'efficientnet' in self.model_name:
                    return torch.mean(my_embedding, (2,3), True).numpy()[:, :,0,0]
                else:
                    return my_embedding.numpy()[:, :, 0, 0]
        else:
            image = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0).to(self.device)

            if self.model_name in ['alexnet', 'vgg']:
                my_embedding = torch.zeros(1, self.layer_output_size)

            elif self.model_name == 'densenet' or 'efficientnet' in self.model_name:
                my_embedding = torch.zeros(1, self.layer_output_size, 7, 7)
            else:
                my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)

            def copy_data(m, i, o):
                my_embedding.copy_(o.data)
            
            h = self.extraction_layer.register_forward_hook(copy_data)

            with torch.no_grad():
                h_x = self.model(image)
            h.remove()

            if tensor:
                return my_embedding
            
            else:
                if self.model_name in ['alexnet','vgg']:
                    return my_embedding.numpy()[0, :]
                elif self.model_name == 'densenet':
                    return torch.mean(my_embedding, (2,3), True).numpy()[0, :,0,0]

                else:

                    return my_embedding.numpy()[0, :, 0, 0]
    
    def _get_model_and_layer(self, model_name, layer):
        '''
        Internal method for getting layer from model
        :param model_name: String name of requested model
        :param layer: layer as a string for resnet-18 or int for alexnet
        :return model: Pytorch model
        :return layer: Pytorch layer
        '''
        if model_name.startswith('resnet') and not model_name.startswith('resnet-'):
            model = getattr(models, model_name)(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = self.RESNET_OUTPUT_SIZES[model_name]
            else:
                layer = model._modules.get(layer)
            return model, layer
        elif model_name == 'resnet-18':
            model = models.resnet18(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 512
            else:
                layer = model._modules.get(layer)

            return model, layer

        else:
            raise KeyError('Model %s was not found - Lỗi ở phần embedding ảnh ' % model_name)


    def get_layer_output_size(self):
        return self.layer_output_size
    
if __name__ == '__main__':
    img2vec = Image_embedding(model='resnet18')
    path_image_1 = "A_REPOSITORY/img2vec/example/test_images/cat.jpg"
    path_image_2 = "A_REPOSITORY/img2vec/example/test_images/cat2.jpg"
    path_image_3 = "A_REPOSITORY/img2vec/example/test_images/catdog.jpg"
    img_1 = Image.open(path_image_1).convert('RGB')
    img_2 = Image.open(path_image_2).convert('RGB')
    img_3 = Image.open(path_image_3).convert('RGB')
    vec_img_1 = img2vec.get_vec(img_1)
    vec_img_2 = img2vec.get_vec(img_2)
    vec_img_3 = img2vec.get_vec(img_3)
    print(vec_img_1.shape)
    print(vec_img_2.shape)
    print(vec_img_3.shape)

    print("cat vs cat2:", cosine_similarity(vec_img_1.reshape((1, -1)), vec_img_2.reshape((1, -1)))[0][0])
    print("cat vs catdog:", cosine_similarity(vec_img_1.reshape((1, -1)), vec_img_3.reshape((1, -1)))[0][0])
