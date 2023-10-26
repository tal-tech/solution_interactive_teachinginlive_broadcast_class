from __future__ import print_function, absolute_import

import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import requests
import traceback
from PIL import Image
from config import Config
from multiprocessing.pool import ThreadPool
from util.utility import get_url_image, get_b64_image, make_error_response, \
    g_logger, make_response, make_com_response, make_all_response, alert_msg

from tools.note_utils import get_sim

pool = ThreadPool(10)


class Note_Scoring(object):

    def __init__(self, weights):
        self.model = models.resnet18(pretrained=False)
        fc_features = self.model.fc.in_features
        # 替换最后的全连接层， 改为训练2类
        self.model.fc = nn.Linear(fc_features, 3)
        self.checkpoint = os.path.join(os.path.dirname(__file__), 'models/')
        regression_trained_model = torch.load(os.path.join(self.checkpoint, 'medium_' + str(weights[2]),
                                                           'model_best_loss_9_20_regression_0.3.pth.tar'))
        self.model.load_state_dict(regression_trained_model['state_dict'])

        self.model.cuda()
        self.model.eval()

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.local_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
        self.weight_pt = torch.from_numpy(np.array(weights)).cuda()

    def calculate_propotion(self, image, bboxes):
        # 由于是二值图 计算全局占比的时候直接加和即可
        text_part = np.sum(image / 255.0)
        background_part = image.shape[0] * image.shape[1]
        op = text_part / background_part
        # relative propotion
        # 找到所有文本块的边界
        if bboxes.shape[0] > 0:
            top_left = np.min(bboxes[:, :2], axis=0)
            bottom_right = np.max(bboxes[:, :2] + bboxes[:, 2:4], axis=0)
            relative_bg_part = (
                bottom_right[0] - top_left[0]) * (bottom_right[1] - top_left[1])
            rp = text_part / relative_bg_part
        else:
            top_left = [0.0, 0.0]
            bottom_right = [0.0, 0.0]
            rp = 0.0
        # 相对占比＋绝对占比＋裁剪框
        return op, rp, (top_left, bottom_right)

    # 腐蚀膨胀 行串联
    def dilated_process(self, image):
        element1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 2))
        dilated = cv2.dilate(image, element1)
        return dilated

    def det_layout_and_cut(self, image):
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 图像二值化 cv::threshold
        ret, bin_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)
        tmp_result = self.dilated_process(bin_img)

        # 获取联通区域
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(
            tmp_result)
        # 去除最大连通域
        bboxes = stats[1::]
        # bboxes = stats
        if bboxes.shape[0] > 0:
            top_left = np.min(bboxes[:, :2], axis=0)
            bottom_right = np.max(bboxes[:, :2] + bboxes[:, 2:4], axis=0)
            new_image = np.array(image[max(0,
                                           top_left[1] - 10):min(image.shape[0],
                                                                 bottom_right[1] + 10),
                                       max(0,
                                           top_left[0] - 10):min(image.shape[1],
                                                                 bottom_right[0] + 10),
                                       :])
            # cv2.imwrite('tmp2.jpg',new_image)
            return new_image
        else:
            return None

    def save_results(self, img_path, image, pred, label, save_dir):
        img_name = os.path.basename(img_path)
        pred_label = self.classify(
            pred, bad_threshold=0.25, medium_threshold=0.62)
        if label == 0:
            cv2.putText(image, str(round(pred * 100, 2)), (50, 150),
                        cv2.FONT_HERSHEY_COMPLEX, 5, (0, 255, 255), 12)
            cv2.putText(image, str(pred_label), (50, 400),
                        cv2.FONT_HERSHEY_COMPLEX, 5, (0, 255, 255), 12)
            if not os.path.exists(os.path.join(save_dir, 'bad')):
                os.makedirs(os.path.join(save_dir, 'bad'))
            cv2.imwrite(
                os.path.join(
                    save_dir, 'bad', str(
                        round(
                            pred * 100, 2)) + '_' + img_name), image)
            # shutil.copy(img,os.path.join(save_dir,'bad',str(round(pred*100,2))+'_'+img_name))
        elif label == 1:
            cv2.putText(image, str(round(pred * 100, 2)), (50, 150),
                        cv2.FONT_HERSHEY_COMPLEX, 5, (0, 255, 255), 12)
            cv2.putText(image, str(pred_label), (50, 400),
                        cv2.FONT_HERSHEY_COMPLEX, 5, (0, 255, 255), 12)
            if not os.path.exists(os.path.join(save_dir, 'good')):
                os.makedirs(os.path.join(save_dir, 'good'))
            cv2.imwrite(
                os.path.join(
                    save_dir, 'good', str(
                        round(
                            pred * 100, 2)) + '_' + img_name), image)
            # shutil.copy(img, os.path.join(save_dir, 'good', str(round(pred * 100, 2)) + '_' + img_name))
        elif label == 2:
            cv2.putText(image, str(round(pred * 100, 2)), (50, 150),
                        cv2.FONT_HERSHEY_COMPLEX, 5, (0, 255, 255), 12)
            cv2.putText(image, str(pred_label), (50, 400),
                        cv2.FONT_HERSHEY_COMPLEX, 5, (0, 255, 255), 12)
            if not os.path.exists(os.path.join(save_dir, 'medium')):
                os.makedirs(os.path.join(save_dir, 'medium'))
            cv2.imwrite(
                os.path.join(
                    save_dir, 'medium', str(
                        round(
                            pred * 100, 2)) + '_' + img_name), image)

            # shutil.copy(img, os.path.join(save_dir, 'medium', str(round(pred * 100, 2)) + '_' + img_name))

    def classify(self, net_out, bad_threshold, medium_threshold):

        if net_out <= bad_threshold:
            pred = 0
        elif bad_threshold < net_out <= medium_threshold:
            pred = 2
        elif medium_threshold < net_out <= 1.0:
            pred = 1
        return pred

    def com_regression_label(self, per_label_count, labels_lens):

        label = 0.
        for i in range(len(per_label_count)):
            label += float(per_label_count[i] / labels_lens) * weights[i]
        return label

    def inference(self, image):

        image = self.det_layout_and_cut(image)  # 预处理，裁剪图像
        if image is not None:
            image = Image.fromarray(image)
            input_image = self.local_transforms(image)
            outputs = self.model(input_image.expand([1, 3, 224, 224]).cuda())
            outputs = F.softmax(outputs)
            outputs = outputs.float() * self.weight_pt.float()
            pred = outputs.sum(dim=1).detach().cpu().numpy()[0]  # 输出的值在0-1之间
            # pred = round(pred * 100, 2) #转换到0-100之间
            # pred = self.classify(pred,bad_threshold,medium_threshold)
            # #按照给定的阈值进行分类
        else:  # 异常处理，如果输入是空白图片,返回0分
            pred = 0.
        return pred

    def get_note_integrity_score(self, recognize_result, reference):
        """
            笔迹工整度评分

        Arguments:
            recognize_result {[json dict]} -- [识别结果]
            reference {[String]} -- [参考的标准笔记]

        Returns:
            [float] -- [识别结果与参考的标准笔记的相似度]
        """
        return get_sim(recognize_result, reference)


weights = [0., 1., 0.3]
note_score = Note_Scoring(weights)


def check_picture(req_id: str, base64=None, url=None):
    # 前置校验
    if base64:
        fn = get_b64_image(base64, req_id)
        if fn is None:
            return make_error_response("illegal base64", req_id)
    else:
        fn = get_url_image(url, req_id)
        if fn is None:
            return make_error_response("download error", req_id)
    if os.path.getsize(fn) > 4 * 1024 * 1024:
        return make_error_response("illegal size", req_id)
    # 图片限制校验
    image = cv2.imread(fn)
    os.remove(fn)
    if image is None:
        return make_error_response('illegal image type', req_id)
    if max(
            image.shape[1],
            image.shape[0]) > 4096 or min(
            image.shape[1],
            image.shape[0]) > 2160:
        return make_error_response("illegal resolution", req_id)
    return image


def get_model_result(req_id: str, image):
    try:
        st2 = time.time()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pred = int(float(note_score.inference(image)) * 100)
        st3 = time.time()
        g_logger.debug("{} - Integrity Time: {}".format(req_id, st3 - st2))
        return make_response("success", pred, req_id)
    except Exception as e:
        g_logger.error("{} - unknown error: {}".format(req_id, traceback.format_exception(type(e), e, e.__traceback__)))
        return make_error_response('internal error', req_id)


def get_ocr_result(req_id: str, text: str, base64=None, url=None):
    ret = None
    st4 = time.time()
    for _ in range(3):
        try:

            d = {"qtype": 1, 'apiName': 'note_score', 'reqId': req_id}
            if base64:
                d.update({"picture": base64})
            elif url:
                d.update({"url": url})
            else:
                return make_error_response('ocr error', req_id)
            ret = requests.post(Config.OCR_URL, json=d, timeout=20).json()
            g_logger.debug("{} - ocr post json: {}".format(req_id, ret))
            if str(ret['code']) in ('5000000', '5000001'):
                continue
            break
        except Exception as e:
            g_logger.error(Config.ALERM_OCR + " - {} - ocr请求错误: {}".
                           format(req_id, traceback.format_exception(type(e), e, e.__traceback__)))
    # g_logger.info(ret)
    # g_logger.info(ret.get('msg'))
    # if ret.get('msg') is None:
    #     g_logger.info(ret.get('msg'))
    #     return make_error_response('ocr error', req_id)
    st5 = time.time()
    try:
        ratio = note_score.get_note_integrity_score(ret['data'], text)
        st6 = time.time()
        g_logger.debug('{} - ocr time:{}'.format(req_id, st5 - st4))
        g_logger.debug('{} - carefully time: {}'.format(req_id, st6 - st5))
    except Exception as e:
        g_logger.error("{} - integrity score error: {}".format(req_id,
                                                               traceback.format_exception(type(e), e, e.__traceback__)))
        return make_error_response('model error', req_id)
    return make_com_response("success", ratio, req_id)


def get_all_result(req_id, image, text, base64=None, url=None):
    ocr = pool.apply_async(get_ocr_result, args=(req_id, text, base64, url))
    score = get_model_result(req_id, image)
    score_json = score.get_json()
    if score_json['code'] == 20000:
        try:
            ocr_ret = ocr.get(timeout=10)
            g_logger.debug("ocr_ret:{}".format(ocr_ret))
            ocr_json = ocr_ret.get_json()
            # g_logger.debug("ocr:{} score:{}".format(ocr_json, score_json))
            if ocr_json['code'] == 20000:
                return make_all_response(
                    "success",
                    ocr_json['data']['complete_ratio'],
                    score_json['data']['score'],
                    req_id)
        except Exception as e:
            g_logger.debug("{} - ocr timeout, error:{}".format(req_id,
                                                               traceback.format_exception(type(e), e, e.__traceback__)))
    return make_all_response("success", -1, score_json['data']['score'], req_id)


if __name__ == '__main__':
    def main():
        for fn in os.listdir('test'):
            img = os.path.join('test', fn)
        print(
            note_score.get_note_integrity_score(
                {
                    'ans': [], 'text': [
                        {
                            'content': '(四', 'timestamp': '0000000000', 'location': [
                                2866, 312, 3036, 461], 'probability': '[0.40198386, 0.35735083]'}, {
                            'content': '月', 'timestamp': '0000000000', 'location': [
                                2889, 495, 3023, 690], 'probability': '[0.92446667]'}, {
                                    'content': '你', 'timestamp': '0000000000', 'location': [
                                        3152, 686, 3355, 886], 'probability': '[0.9997832]'}], 'char_count': 4},
                '这是参考文本是空的情况的返回值是吗？'))
        # print(note_score.get_note_integrity_score(d, t))
        # print(os.path.join(os.path.dirname(__file__), 'models/'))
    
    main()

# if __name__ == '__main__':
#     print(Config.OCR_URL + '?requestId=xxx')
