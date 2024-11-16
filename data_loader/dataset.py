# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:54
# @Author  : zhoujun
import math
import pathlib
import os
import cv2
import numpy as np
import scipy.io as sio
from tqdm.auto import tqdm
import json
import sys
sys.path.append(".")
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

from base import BaseDataSet
from utils import order_points_clockwise, order_points_clockwise_list, get_datalist, load,expand_polygon


class ICDAR2015Dataset(BaseDataSet):
    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags, transform=None, **kwargs):
        super().__init__(data_path, img_mode, pre_processes, filter_keys, ignore_tags, transform)
        self.dataRoot=os.path.dirname(os.path.abspath(data_path[0])) 
        dict_path=os.path.join(self.dataRoot,"dict.txt")
        self.label2char=['']
        self.char2label={'':0}
        with open(dict_path,encoding='utf-8',mode='r') as f:
            for i,line in enumerate(f.readlines()):
                word=line.strip()
                self.label2char.append(word)
                self.char2label[word]=i+1
        self.label2char.append('unknown')
        self.char2label['unknown']=len(self.label2char)-1
        if "lexicon" in kwargs:
            self.lexicon_type = kwargs['lexicon']
            self.lexicon_path = kwargs["lexicon_path"]
        else:
            self.lexicon_type=None
        
        if self.lexicon_type in ["Weak", "Generic"]:
            self.lexicon=[]
            with open(self.lexicon_path) as f:
                for line in f:
                    self.lexicon.append(line.strip())
            
            

    def load_data(self, data_path: str) -> list:
        data_list = get_datalist(data_path)
        t_data_list = []
        for img_path, label_path in data_list:
            data = self._get_annotation(label_path)
            if len(data['text_polys']) > 0:
                item = {'img_path': img_path, 'img_name': pathlib.Path(img_path).stem}
                item.update(data)
                t_data_list.append(item)
            else:
                print('there is no suit bbox in {}'.format(label_path))
        return t_data_list

    def _get_annotation(self, label_path: str) -> dict:
        boxes = []
        texts = []
        ignores = []
        with open(label_path, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                try:
                    box = order_points_clockwise(np.array(list(map(float, params[:8]))).reshape(-1, 2))
                    if cv2.contourArea(box) > 0:
                        boxes.append(box)
                        label = params[8]
                        texts.append(label)
                        ignores.append(label in self.ignore_tags)
                except Exception as e:
                    print('load label failed on {}'.format(label_path),e)
        # print(np.array(boxes))
        data = {
            'text_polys': np.array(boxes),
            'texts': texts,
            'ignore_tags': ignores,
        }
        return data

class ICDAR2019artDataset(BaseDataSet):
    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags, transform=None, **kwargs):
        super().__init__(data_path, img_mode, pre_processes, filter_keys, ignore_tags, transform)
        self.dataRoot=os.path.dirname(os.path.abspath(data_path[0])) 
        dict_path=os.path.join(self.dataRoot,"dict.txt")
        self.label2char=['']
        self.char2label={'':0}
        with open(dict_path,encoding='utf-8',mode='r') as f:
            for i,line in enumerate(f.readlines()):
                word=line.strip()
                self.label2char.append(word)
                self.char2label[word]=i+1
        self.label2char.append('unknown')
        self.char2label['unknown']=len(self.label2char)-1
        if "lexicon" in kwargs:
            self.lexicon_type = kwargs['lexicon']
            self.lexicon_path = kwargs["lexicon_path"]
        else:
            self.lexicon_type=None
        
        if self.lexicon_type in ["Weak", "Generic"]:
            self.lexicon=[]
            with open(self.lexicon_path) as f:
                for line in f:
                    self.lexicon.append(line.strip())
            
            

    def load_data(self, data_path: str) -> list:
        # print(data_path)
        t_data_list = []
        with open('./datasets/datasets/icdar2019-art/train_labels.json', 'r', encoding='utf-8') as file:
            json_data = json.load(file)
            with open(data_path[0], encoding='utf-8', mode='r') as f:
                for line in f.readlines():
                # print(data)
                    # for i in json_data:
                    # line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                    img_path = line.strip().strip('\ufeff').strip('\xef\xbb\xbf')
                    key_json = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split('.')[1].split('/')[-1]
                    img_mes = json_data[key_json]
                    boxes = []
                    texts = []
                    ignores = []
                    for t in img_mes:
                        boxes.append(np.array(t['points']).astype(np.float32))
                        # p = np.array(t['points'])
                        # maxv = np.amax(p, axis=0)
                        # minv = np.amin(p, axis=0)
                        # if len(t['points'])>=4:
                        #     boxes.append(np.array([[minv[0],minv[1]],[minv[0],maxv[1]],[maxv[0],minv[1]],[maxv[0],maxv[1]]]).astype(np.float32))
                        # elif len(t['points'])>4:
                        #     boxes.append(np.array([t['points'][0],t['points'][1],t['points'][2],t['points'][3]]).astype(np.float32))
                        #     print(boxes)
                        texts.append(t['transcription'])
                        ignores.append(t['transcription'] in self.ignore_tags)
                    # print(np.array(boxes))

                    data = {
                            'text_polys': np.array(boxes),
                            'texts': texts,
                            'ignore_tags': ignores,
                        }
                    # img_path = './datasets/datasets/icdar2019-art/train_images/'+i+'.jpg'
                    if len(data['text_polys']) > 0:
                        item = {'img_path': img_path, 'img_name': pathlib.Path(img_path).stem}
                        item.update(data)
                        t_data_list.append(item)
                    else:
                        print('there is no suit bbox in {}'.format(label_path))
                    # t_data_list.append(data)        
        
        print(len(t_data_list))
        
        return t_data_list



class TotalTextDataset(BaseDataSet):
    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags, transform=None, **kwargs):
        super().__init__(data_path, img_mode, pre_processes, filter_keys, ignore_tags, transform)

    def load_data(self, data_path: str) -> list:
        data_list = get_datalist(data_path)
        t_data_list = []
        for img_path, label_path in data_list:
            data = self._get_annotation(label_path)
            if len(data['text_polys']) > 0:
                item = {'img_path': img_path, 'img_name': pathlib.Path(img_path).stem}
                item.update(data)
                t_data_list.append(item)
            else:
                print('there is no suit bbox in {}'.format(label_path))
        return t_data_list

    def _get_annotation(self, label_path: str) -> dict:
        boxes = []
        texts = []
        ignores = []
        with open(label_path, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                try:
                    box = (np.array(list(map(float, params[:-1]))).reshape(-1, 2)).astype(np.float32)
                    if cv2.contourArea(box) > 0:
                        boxes.append(box)
                        label = params[-1]
                        texts.append(label)
                        ignores.append(label in self.ignore_tags)
                except Exception as e:
                    print('load label failed on {}'.format(label_path),e)
        data = {
            'text_polys': np.array(boxes),
            'texts': texts,
            'ignore_tags': ignores,
        }
        return data

class CTW1500Dataset(BaseDataSet):
    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags, transform=None, **kwargs):
        super().__init__(data_path, img_mode, pre_processes, filter_keys, ignore_tags, transform)
        # 字符集读取
        self.dataRoot=os.path.dirname(os.path.abspath(data_path[0])) 
        dict_path=os.path.join(self.dataRoot,"dict.txt")
        self.label2char=['']
        self.char2label={'':0}
        with open(dict_path,encoding='utf-8',mode='r') as f:
            for i,line in enumerate(f.readlines()):
                word=line.strip()
                self.label2char.append(word)
                self.char2label[word]=i+1
        self.label2char.append('unknown')
        self.char2label['unknown']=len(self.label2char)-1
        if "lexicon" in kwargs and not kwargs["lexicon"]=="None":
            self.lexicon_type = kwargs['lexicon']
            self.lexicon_path = kwargs["lexicon_path"]
        else:
            self.lexicon_type=None
        
        if self.lexicon_type in ["Weak", "Generic"]:
            self.lexicon=[]
            with open(self.lexicon_path) as f:
                for line in f:
                    self.lexicon.append(line.strip())

    def load_data(self, data_path: str) -> list:
        data_list = get_datalist(data_path)
        t_data_list = []
        for img_path, label_path in data_list:
            data = self._get_annotation(label_path)
            if len(data['text_polys']) > 0:
                item = {'img_path': img_path, 'img_name': pathlib.Path(img_path).stem}
                item.update(data)
                t_data_list.append(item)
            else:
                print('there is no suit bbox in {}'.format(label_path))
        return t_data_list

    def _get_annotation(self, label_path: str) -> dict:
        boxes = []
        texts = []
        ignores = []
        with open(label_path, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                try:
                    box = (np.array(list(map(float, params[:28]))).reshape(-1, 2)).astype(np.float32)
                    if cv2.contourArea(box) > 0:
                        boxes.append(box)
                        label = ",".join(params[28:])
                        texts.append(label)
                        ignores.append(label in self.ignore_tags)
                except Exception as e:
                    print('load label failed on {}'.format(label_path),e)
        data = {
            'text_polys': np.array(boxes),
            'texts': texts,
            'ignore_tags': ignores,
        }
        return data

class CTW1500TrainDataset(BaseDataSet):
    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags, transform=None, **kwargs):
        super().__init__(data_path, img_mode, pre_processes, filter_keys, ignore_tags, transform)
        # 字符集读取
        self.dataRoot=os.path.dirname(os.path.abspath(data_path[0])) 
        dict_path=os.path.join(self.dataRoot,"dict.txt")
        self.label2char=['']
        self.char2label={'':0}
        with open(dict_path,encoding='utf-8',mode='r') as f:
            for i,line in enumerate(f.readlines()):
                word=line.strip()
                self.label2char.append(word)
                self.char2label[word]=i+1
        self.label2char.append('unknown')
        self.char2label['unknown']=len(self.label2char)-1
        if "lexicon" in kwargs and not kwargs["lexicon"]=="None":
            self.lexicon_type = kwargs['lexicon']
            self.lexicon_path = kwargs["lexicon_path"]
        else:
            self.lexicon_type=None
        
        if self.lexicon_type in ["Weak", "Generic"]:
            self.lexicon=[]
            with open(self.lexicon_path) as f:
                for line in f:
                    self.lexicon.append(line.strip())

    def load_data(self, data_path: str) -> list:
        data_list = get_datalist(data_path)
        t_data_list = []
        for img_path, label_path in data_list:
            data = self._get_annotation(label_path)
            if len(data['text_polys']) > 0:
                item = {'img_path': img_path, 'img_name': pathlib.Path(img_path).stem}
                item.update(data)
                t_data_list.append(item)
            else:
                print('there is no suit bbox in {}'.format(label_path))
        return t_data_list

    def _get_annotation(self, label_path: str) -> dict:
        boxes = []
        texts = []
        ignores = []
        import xml.dom.minidom
        dom = xml.dom.minidom.parse(label_path)
        dom_boxes=dom.documentElement.getElementsByTagName('box')
        for dom_box in dom_boxes:
            line=dom_box.getElementsByTagName("segs")[0].firstChild.data
            params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
            try:
                box = (np.array(list(map(float, params))).reshape(-1, 2)).astype(np.float32)
                if cv2.contourArea(box) > 0:
                    boxes.append(box)
                    label = dom_box.getElementsByTagName("label")[0].firstChild.data
                    texts.append(label)
                    ignores.append(label in self.ignore_tags)
            except Exception as e:
                print('load label failed on {}'.format(label_path),e)
        data = {
            'text_polys': np.array(boxes),
            'texts': texts,
            'ignore_tags': ignores,
        }
        return data

class TD500Dataset(BaseDataSet):
    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags, transform=None, **kwargs):
        super().__init__(data_path, img_mode, pre_processes, filter_keys, ignore_tags, transform)

    def load_data(self, data_path: str) -> list:
        data_list = get_datalist(data_path)
        t_data_list = []
        for img_path, label_path in data_list:
            data = self._get_annotation(label_path)
            if len(data['text_polys']) > 0:
                item = {'img_path': img_path, 'img_name': pathlib.Path(img_path).stem}
                item.update(data)
                t_data_list.append(item)
            else:
                print('there is no suit bbox in {}'.format(label_path))
        return t_data_list

    def _get_annotation(self, label_path: str) -> dict:
        boxes = []
        texts = []
        ignores = []
        with open(label_path, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(' ')
                try:
                    box = (np.array(list(map(float, params[2:]))))
                    box = np.array(self.get_box_img(*box),dtype=np.float32).reshape(-1,2)
                    if cv2.contourArea(box) > 0:
                        boxes.append(box)
                        difficulty = params[1]
                        texts.append("")
                        ignores.append(difficulty in self.ignore_tags)
                except Exception as e:
                    print('load label failed on {}'.format(label_path),e)
        data = {
            'text_polys': np.array(boxes),
            'texts': texts,
            'ignore_tags': ignores,
        }
        return data
    #处理TD500数据
    def get_box_img(self, x, y, w, h, angle):
        # 矩形框中点(x0,y0)
        x0 = x + w/2
        y0 = y + h/2
        l = math.sqrt(pow(w/2, 2) + pow(h/2, 2))  # 即对角线的一半
        # angle小于0，逆时针转
        if angle < 0:
            a1 = -angle + math.atan(h / float(w))  # 旋转角度-对角线与底线所成的角度
            a2 = -angle - math.atan(h / float(w)) # 旋转角度+对角线与底线所成的角度
            pt1 = (x0 - l * math.cos(a2), y0 + l * math.sin(a2))
            pt2 = (x0 + l * math.cos(a1), y0 - l * math.sin(a1))
            pt3 = (x0 + l * math.cos(a2), y0 - l * math.sin(a2))  # x0+左下点旋转后在水平线上的投影, y0-左下点在垂直线上的投影，显然逆时针转时，左下点上一和左移了。
            pt4 = (x0 - l * math.cos(a1), y0 + l * math.sin(a1))
        else:
            a1 = angle + math.atan(h / float(w))
            a2 = angle - math.atan(h / float(w))
            pt1 = (x0 - l * math.cos(a1), y0 - l * math.sin(a1))
            pt2 = (x0 + l * math.cos(a2), y0 + l * math.sin(a2))
            pt3 = (x0 + l * math.cos(a1), y0 + l * math.sin(a1))
            pt4 = (x0 - l * math.cos(a2), y0 - l * math.sin(a2))
        return [pt1[0], pt1[1], pt2[0], pt2[1], pt3[0], pt3[1], pt4[0], pt4[1]]

class DetDataset(BaseDataSet):
    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags, transform=None, **kwargs):
        self.load_char_annotation = kwargs['load_char_annotation']
        self.expand_one_char = kwargs['expand_one_char']
        super().__init__(data_path, img_mode, pre_processes, filter_keys, ignore_tags, transform)

    def load_data(self, data_path: str) -> list:
        """
        从json文件中读取出 文本行的坐标和gt，字符的坐标和gt
        :param data_path:
        :return:
        """
        data_list = []
        for path in data_path:
            content = load(path)
            for gt in tqdm(content['data_list'], desc='read file {}'.format(path)):
                img_path = os.path.join(content['data_root'], gt['img_name'])
                polygons = []
                texts = []
                illegibility_list = []
                language_list = []
                for annotation in gt['annotations']:
                    if len(annotation['polygon']) == 0 or len(annotation['text']) == 0:
                        continue
                    if len(annotation['text']) > 1 and self.expand_one_char:
                        annotation['polygon'] = expand_polygon(annotation['polygon'])
                    polygons.append(annotation['polygon'])
                    texts.append(annotation['text'])
                    illegibility_list.append(annotation['illegibility'])
                    language_list.append(annotation['language'])
                    if self.load_char_annotation:
                        for char_annotation in annotation['chars']:
                            if len(char_annotation['polygon']) == 0 or len(char_annotation['char']) == 0:
                                continue
                            polygons.append(char_annotation['polygon'])
                            texts.append(char_annotation['char'])
                            illegibility_list.append(char_annotation['illegibility'])
                            language_list.append(char_annotation['language'])
                data_list.append({'img_path': img_path, 'img_name': gt['img_name'], 'text_polys': np.array(polygons),
                                  'texts': texts, 'ignore_tags': illegibility_list})
        return data_list


class SynthTextDataset(BaseDataSet):
    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, transform=None, **kwargs):    
        self.transform = transform
        self.dataRoot = pathlib.Path(data_path[0])
        if not self.dataRoot.exists():
            raise FileNotFoundError('Dataset folder is not exist.')

        self.targetFilePath = self.dataRoot / 'gt.mat'
        if not self.targetFilePath.exists():
            raise FileExistsError('Target file is not exist.')
        targets = {}
        sio.loadmat(self.targetFilePath, targets, squeeze_me=True, struct_as_record=False,
                    variable_names=['imnames', 'wordBB', 'txt'])

        self.imageNames = targets['imnames']
        self.wordBBoxes = targets['wordBB']
        self.transcripts = targets['txt']
        
        # 读取dict
        dict_path=os.path.join(self.dataRoot,"dict.txt")
        self.label2char=['']
        self.char2label={'':0}
        with open(dict_path,encoding='utf-8',mode='r') as f:
            for i,line in enumerate(f.readlines()):
                word=line.strip()
                self.label2char.append(word)
                self.char2label[word]=i+1
        self.label2char.append('unknown')
        self.char2label['unknown']=len(self.label2char)-1
        
        super().__init__(data_path, img_mode, pre_processes, filter_keys,[], transform)

    def load_data(self, data_path: str) -> list:
        t_data_list = []
        count=0
        for imageName, wordBBoxes, texts in zip(self.imageNames, self.wordBBoxes, self.transcripts):
            count=(count+1)%5
            if count !=0:#给synth做一个特殊处理，只导入10%
                continue
            item = {}
            wordBBoxes = np.expand_dims(wordBBoxes, axis=2) if (wordBBoxes.ndim == 2) else wordBBoxes
            _, _, numOfWords = wordBBoxes.shape
            text_polys = wordBBoxes.reshape([8, numOfWords], order='F').T  # num_words * 8
            
            text_polys = text_polys.reshape(numOfWords, 4, 2)  # num_of_words * 4 * 2
            transcripts = [word for line in texts for word in line.split()]
            if numOfWords != len(transcripts):
                continue
            item['img_path'] = str(self.dataRoot / imageName)
            item['img_name'] = (self.dataRoot / imageName).stem
            item['text_polys'] = text_polys
            item['texts'] = transcripts
            item['ignore_tags'] = [x in self.ignore_tags for x in transcripts]
            t_data_list.append(item)
        return t_data_list
    
    
class WildReceiptDataSet(BaseDataSet):
    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, transform=None, **kwargs):  
        self.dataRoot=os.path.dirname(os.path.abspath(data_path[0])) 
        dict_path=os.path.join(self.dataRoot,"dict.txt")
        self.label2char=['']
        self.char2label={'':0}
        with open(dict_path,encoding='utf-8',mode='r') as f:
            for i,line in enumerate(f.readlines()):
                word=line.strip()
                self.label2char.append(word)
                self.char2label[word]=i+1
        self.label2char.append('unknown')
        self.char2label['unknown']=len(self.label2char)-1
        super().__init__(data_path, img_mode, pre_processes, filter_keys,[], transform)

    def load_data(self, data_pathes: str) -> list:
        t_data_list = []
        for data_path in data_pathes:
            with open(data_path, encoding='utf-8', mode='r') as f:
                for line in f.readlines():
                    item={}
                    infos=eval(line)
                    file_name=infos['file_name']
                    annotations=infos['annotations']
                    if infos['height']>2000 or infos['width']>2000:#不要太大的
                        continue
                    item['img_path'] = os.path.join(self.dataRoot,file_name)
                    item['img_name'] = os.path.basename(file_name)
                    item['text_polys'], item['texts'], item['ignore_tags'] =[],[],[]
                    for box in annotations:
                        item['text_polys'].append(np.array(box['box']).reshape(-1, 2).astype(np.float32))
                        item['texts'].append(box['text'])
                        item['ignore_tags'].append(box['text'] in self.ignore_tags)
                    item['text_polys']=np.array(item['text_polys'])
                    t_data_list.append(item)
        return t_data_list

    
class ICDAR2019RECDataset(BaseDataSet):
    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, transform=None, **kwargs):  
        self.dataRoot=os.path.dirname(os.path.abspath(data_path[0])) 
        dict_path=os.path.join(self.dataRoot,"dict.txt")
        self.label2char=['']
        self.char2label={'':0}
        with open(dict_path,encoding='utf-8',mode='r') as f:
            for i,line in enumerate(f.readlines()):
                word=line.strip()
                self.label2char.append(word)
                self.char2label[word]=i+1
        self.label2char.append('unknown')
        self.char2label['unknown']=len(self.label2char)-1
        super().__init__(data_path, img_mode, pre_processes, filter_keys,[], transform)

    def load_data(self, data_pathes: str) -> list:
        t_data_list = []
        for data_path in data_pathes:
            with open(data_path, encoding='utf-8', mode='r') as f:
                for line in f.readlines():
                    item={}
                    width,height,file_name,text=line.split('\t')
                    width,height=int(width),int(height)
                    item['img_path'] = os.path.join(self.dataRoot,"train_images",file_name)
                    item['img_name'] = os.path.basename(file_name)
                    item['text_polys'] = np.array([[[0,0],[width,0],[width,height],[0,height]]]).astype(np.float32)
                    item['texts'] = [text.strip()]
                    item['ignore_tags'] = [False]
                    t_data_list.append(item)
        return t_data_list

class MLT2017Dataset(BaseDataSet):
    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags, transform=None, **kwargs):
        super().__init__(data_path, img_mode, pre_processes, filter_keys, ignore_tags, transform)

    def load_data(self, data_path: str) -> list:
        data_list = get_datalist(data_path)

        t_data_list = []
        for img_path, label_path in data_list:
            data = self._get_annotation(label_path)
            if len(data['text_polys']) > 0:
                item = {'img_path': img_path, 'img_name': pathlib.Path(img_path).stem}
                item.update(data)
                t_data_list.append(item)
            else:
                print('there is no suit bbox in {}'.format(label_path))
        return t_data_list

    def _get_annotation(self, label_path: str) -> dict:
        boxes = []
        texts = []
        ignores = []
        with open(label_path, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                try:
                    box = order_points_clockwise(np.array(list(map(float, params[:8]))).reshape(-1, 2))
                    if cv2.contourArea(box) > 0:
                        boxes.append(box)
                        label = params[9]
                        texts.append(label)
                        ignores.append(label in self.ignore_tags)
                except Exception as e:
                    print('load label failed on {}'.format(label_path),e)
        # print(np.array(boxes))
        data = {
            'text_polys': np.array(boxes),
            'texts': texts,
            'ignore_tags': ignores,
        }
        return data
    

if __name__ == '__main__':
    import torch
    import anyconfig
    from torch.utils.data import DataLoader
    from torchvision import transforms

    from utils import parse_config, show_img, plt, draw_bbox

    config = anyconfig.load('config/ctw1500_resnet18_FPN_DB_CT_head_polyLR.yaml')
    read_type='train'
    show_type='det_train'
    # dir_name="output/datasets/td500/validate"
    dir_name="fig"
    config = parse_config(config)
    dataset_args = config['dataset'][read_type]['dataset']['args']
    dataset_type = config['dataset'][read_type]['dataset']['type']
    config['dataset'][read_type]['loader']['shuffle']=False
    # config['dataset']['train']['dataset']['args']['filter_keys'].remove('text_polys')
    if 'img_name' in dataset_args['filter_keys']: 
        dataset_args['filter_keys'].remove('img_name')
    # dataset_args.pop('data_path')
    # data_list = [(r'E:/zj/dataset/icdar2015/train/img/img_15.jpg', 'E:/zj/dataset/icdar2015/train/gt/gt_img_15.txt')]
    train_data = eval(dataset_type)(data_path=dataset_args.pop('data_path'), transform=transforms.ToTensor(),
                                  **dataset_args)
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=0)
    import matplotlib
    matplotlib.use("agg")
    # matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
    import matplotlib.pyplot as plt 
    import matplotlib.patches as patches 
    # plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
    # plt.rcParams['axes.unicode_minus']=False


    os.makedirs(dir_name,exist_ok=True)
    
    if show_type=='det_train':
        from dataset_visualize import save_train_image
        save_image=save_train_image
    elif show_type=='det_validate':
        from dataset_visualize import save_test_image
        save_image=save_test_image
    elif show_type=='rec':
        from dataset_visualize import save_rec_image
        save_image=save_rec_image
    elif show_type=='distance_map':
        from dataset_visualize import save_distance_image
        save_image=save_distance_image
    
    for i, data in enumerate(tqdm(train_loader)):
        save_image(data['img_name'][0],data,output_dir=dir_name)       
        pass
