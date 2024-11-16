import cv2
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
import pyclipper
from shapely.geometry import Polygon
import Polygon as plg

class MakeDistanceMap():
    def __init__(self, kernel_scale, **args):
        self.kernel_scale=kernel_scale
        pass
    
    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        img = data['img']
        bboxes = data['text_polys']
        ignore_tags = data['ignore_tags']


        gt_instance = np.zeros(img.shape[0:2], dtype='uint8')
        training_mask = np.ones(img.shape[0:2], dtype='uint8')
        training_mask_distance = np.ones(img.shape[0:2], dtype='uint8')
        if len(bboxes) > 0:
            # bboxes=bboxes.astype('int32')
            for i in range(len(bboxes)):
                bbox=bboxes[i].astype('int32')
                cv2.drawContours(gt_instance, [bbox], -1, i + 1, -1)
                cv2.drawContours(training_mask, [bbox], -1, 0, -1)
                if ignore_tags[i]:
                    cv2.drawContours(training_mask_distance, [bbox], -1, 0, -1)


        gt_kernel_instance = np.zeros(img.shape[0:2], dtype='uint8')
        kernel_bboxes = self.shrink(bboxes, self.kernel_scale)
        for i in range(len(bboxes)):
            cv2.drawContours(gt_kernel_instance, [kernel_bboxes[i]], -1, i + 1, -1)
            if ignore_tags[i]:
                cv2.drawContours(training_mask, [kernel_bboxes[i]], -1, 1, -1)
        gt_kernel = gt_kernel_instance.copy()
        gt_kernel[gt_kernel > 0] = 1
        
        tmp1 = gt_instance.copy()
        dilate_kernel = np.ones((3, 3), np.uint8)
        tmp1 = cv2.dilate(tmp1, dilate_kernel, iterations=1)
        tmp2 = tmp1.copy()
        tmp2 = cv2.dilate(tmp2, dilate_kernel, iterations=1)
        gt_kernel_inner = tmp2 - tmp1

        max_instance = np.max(gt_instance)
        gt_distance = np.zeros((2, *img.shape[0:2]), dtype=np.float32)
        for i in range(1, max_instance + 1):
            ind = (gt_kernel_inner == i)
            if np.sum(ind) == 0:
                training_mask[gt_instance == i] = 0
                training_mask_distance[gt_instance == i] = 0
                continue
            kpoints = np.array(np.where(ind)).transpose((1, 0))[:, ::-1].astype('float32')

            ind = (gt_instance == i) * (gt_kernel_instance == 0)
            if np.sum(ind) == 0:
                continue
            pixels = np.where(ind)
            points = np.array(pixels).transpose((1, 0))[:, ::-1].astype('float32')

            bbox_ind = self.jaccard(points, kpoints)
            offset_gt = points - kpoints[bbox_ind] 
            gt_distance[:, pixels[0], pixels[1]] = offset_gt.T * 0.1

        data['gt_instances'] = gt_instance
        data['training_mask_distances'] = training_mask_distance
        data['gt_distances'] = gt_distance
        data['gt_kernel_instances'] = gt_kernel_instance
        
        return data


    def dist(self, a, b):
        return np.linalg.norm((a - b), ord=2, axis=0)

    def perimeter(self, bbox):
        peri = 0.0
        for i in range(bbox.shape[0]):
            peri += self.dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
        return peri

    def shrink(self, bboxes, rate, max_shr=20):
        rate = rate * rate
        shrinked_bboxes = []
        for bbox in bboxes:
            area = plg.Polygon(bbox).area()
            peri = self.perimeter(bbox)

            try:
                pco = pyclipper.PyclipperOffset()
                pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                offset = min(int(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)

                shrinked_bbox = pco.Execute(-offset)
                if len(shrinked_bbox) == 0:
                    shrinked_bboxes.append(bbox)
                    continue

                shrinked_bbox = np.array(shrinked_bbox[0])
                if shrinked_bbox.shape[0] <= 2:
                    shrinked_bboxes.append(bbox)
                    continue

                shrinked_bboxes.append(shrinked_bbox)
            except Exception as e:
                print(type(shrinked_bbox), shrinked_bbox)
                print('area:', area, 'peri:', peri)
                shrinked_bboxes.append(bbox)

        return shrinked_bboxes

    def jaccard(self, As, Bs):

        A = As.shape[0]
        B = Bs.shape[0]

        dis = np.sqrt(np.sum((As[:, np.newaxis, :].repeat(B, axis=1)
                            - Bs[np.newaxis, :, :].repeat(A, axis=0)) ** 2, axis=-1))

        ind = np.argmin(dis, axis=-1)

        return ind
