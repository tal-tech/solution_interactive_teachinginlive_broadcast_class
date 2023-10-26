import re
from tools.process_data import union_words
import json
import os
import time

# 纯净覆盖度计算


def calc_pure_integrity(target_text, source_text):
    n1 = len(source_text)
    n2 = len(target_text)
    if target_text == source_text:
        return 1
    elif n1 == 0 or n2 == 0:
        return 0
    target_text, source_text = target_text.strip(), source_text.strip()

    v0 = [None] * (n2 + 1)
    v1 = [None] * (n2 + 1)
    for i in range(len(v0)):
        v0[i] = i
    for i in range(n1):
        v1[0] = 0
        for j in range(n2):
            cost = 0 if target_text[j] == source_text[i] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1], v0[j] + cost)

        for j in range(len(v0)):
            v0[j] = v1[j]
        # print(v1)

    return (n2 - v1[n2]) / n2

# 模糊覆盖度计算


def filter_empty(process_list):
    return [elem for elem in process_list if elem.strip()]


def get_fuzzy_integrity(pred_text, ref_text):
    clear_ref_text = union_words(ref_text)
    clear_ref_text = re.sub("[ \t]", "", clear_ref_text)
    clear_pred_text = union_words(pred_text)
    clear_pred_text = re.sub("[ \t]", "", clear_pred_text)
    ref_text_list, pred_text_list = clear_ref_text.split(
        "\n"), clear_pred_text.split("\n")
    ref_text_list, pred_text_list = filter_empty(
        ref_text_list), filter_empty(pred_text_list)

    sorted_pred_text_list = sorted(
        pred_text_list,
        key=lambda x: len(x),
        reverse=True)
    match_num = 0
    for pred_txt in sorted_pred_text_list:
        sort_integrity_list = sorted([(ref_txt,
                                       calc_pure_integrity(pred_txt,
                                                           ref_txt)) for ref_txt in ref_text_list],
                                     key=lambda x: x[1],
                                     reverse=True)
        ref_txt, integrity = sort_integrity_list[0]
        accept_theshold = 0.25
        if integrity >= accept_theshold:
            match_num += min(len(pred_txt), len(ref_txt))
    return match_num / len(clear_ref_text)


def get_sim(recognize_result, reference):

    src_text = recognize_result["text"]
    recog_text = ""
    for text_line in src_text:
        tmp_text = text_line["content"]
        recog_text += tmp_text
        recog_text += '\n'

    return get_fuzzy_integrity(recog_text, reference)


def get_all_files(path):
    filelist = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith('.json'):
                filelist.append(os.path.join(root, name))
    print('There are %d files' % (len(filelist)))
    return filelist


def test_interface(recognize_json, ref_file):
    with open(recognize_json, 'r', encoding='utf-8') as f:
        recognize_result = json.load(f)

    with open(ref_file, 'r', encoding='utf-8') as f2:
        reference = f2.read()

    return get_sim(recognize_result, reference)


if __name__ == "__main__":

    # 数学
    math_files = get_all_files(
        '/workspace/houqi/GodEye/OCR/note_integrity/抄写_result/数学')
    refer_math = '/workspace/houqi/GodEye/OCR/note_integrity/笔记demo文本txt/数学笔记demo-5年级-LaTeX.txt.bak'

    for i in math_files:
        img_name = os.path.basename(i).replace('.json', '')
        st = time.time()
        sim = test_interface(i, refer_math)
        end = time.time()
        print('note:{} \t sim:{} \t time:{}'.format(
            os.path.basename(i), sim, end - st))

    # 语文
    math_files = get_all_files(
        '/workspace/houqi/GodEye/OCR/note_integrity/抄写_result/语文')
    refer_math = '/workspace/houqi/GodEye/OCR/note_integrity/笔记demo文本txt/语文笔记demo-5年级.txt.bak'

    for i in math_files:
        img_name = os.path.basename(i).replace('.json', '')
        st = time.time()
        sim = test_interface(i, refer_math)
        end = time.time()
        print('note:{} \t sim:{} \t time:{}'.format(
            os.path.basename(i), sim, end - st))
