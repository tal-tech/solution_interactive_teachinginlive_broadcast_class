# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     note_utils
   Author :        houqi
   date：          2019/10/9
   Description :

"""
import os, sys
import numpy as np

import jieba
import jieba.analyse
import json
import gensim
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=FutureWarning, module='gensim')
from gensim import corpora
from gensim.similarities import Similarity
import time
import cv2


def completeness_scoring(recognize_result, reference):
    reference_text = reference.split("\n")
    tmp_key_reference_words = jieba.analyse.extract_tags(reference, 30)

    src_text = recognize_result["text"]
    recog_text = ""
    for text_line in src_text:
        tmp_text = text_line["content"]
        recog_text += tmp_text
        # tmp_text_list = list(jieba.cut(tmp_text))
        # print(tmp_text_list)
    print(recog_text)
    tmp_key_recognize_words = jieba.analyse.extract_tags(recog_text, 30)
    print(tmp_key_recognize_words)
    print(tmp_key_reference_words)
    print(list(jieba.cut(recog_text)))
    print(list(jieba.cut(reference)))


def get_match_dict(string):
    arr = []
    arr_index = []
    pos_map = {}
    for index, char in enumerate(string):
        if char == '{':
            arr.append(char)
            arr_index.append(index)

        elif char == '}':
            if arr and arr[-1] == '{':
                pos_map[arr_index[-1]] = index
                arr.pop()
                arr_index.pop()
            else:
                assert 'string {} is not match'.format(string)
    return pos_map


def replace_any(strip_raw_text, replace_op):
    op_in = strip_raw_text.find(replace_op)
    while op_in >= 0:
        match_dict = get_match_dict(strip_raw_text)
        # 　print(op_in, match_dict)
        bra_st = op_in + len(replace_op)
        if bra_st in match_dict:
            strip_raw_text = strip_raw_text.replace(strip_raw_text[op_in:match_dict[bra_st] + 1],
                                                    strip_raw_text[bra_st + 1:match_dict[bra_st]])
        else:
            strip_raw_text = strip_raw_text.replace(strip_raw_text[op_in:bra_st + 1], "")

        op_in = strip_raw_text.find(replace_op)

    return strip_raw_text


def union_symbol(strip_raw_text):
    raw_list = ["\\right", "\\left", "\\rm", "\\leqslant", "\\mathsf", "\\underbrace",
                "\\geqslant", "\\bigstar", "\\quad", "\\hline", "\\dfrac", "\\triangle", "\\Delta",
                "\\Rightarrow", "\\rightarrow", "\\alpha", "\\beta", "\\rho", "\\mu", "\\theta",
                "\\times", "\\div", "\\pi", "\\angle", "{}^\\circ", "^\\circ", "^{\\circ}", "\\cdots", "\\cdot",
                "\\ldots", "\\dots", "\\pm", "\\because", "\\therefore", "\\neq", "\\geq", "\\leq", "\\equiv",
                "\\approx", "\\Square", "\\square", "\\max", "\\min", "\\cos", "\\sin", "\\tan",
                "\\%", "\\_", "\\downarrow", "\\uparrow", "\\ast", "\\oplus", "\\sim", "\\bmod",
                "\\longrightarrow", "\\Downarrow", "\\Uparrow", "\\Leftrightarrow", "\\lambda",
                "arrow", "\\gamma", "\\begin{cases}", "\\end{cases}", "\\begin{aligned}", "\\end{aligned}", '\\ne'
                ]
    replace_list = ["", "", "", "", "", "",
                    "", "", "", "", "\\frac", "△", "△",
                    "→", "→", "α", "β", "ρ", "μ", "θ",
                    "×", "÷", "π", "∠", "°", "°", "°", "···", "·",
                    "···", "···", "±", "∵", "∴", "≠", "≥", "≤", "≡",
                    "≈", "□", "□", "max", "min", "cos", "sin", "tan",
                    "%", "_", "↓", "↑", "*", "⊕", "~", "mod",
                    "→", "↓", "↑", "↔", "λ",
                    "→", "γ", "", "", "", "", '≠'
                    ]

    for tmp_sym, tar_sym in zip(raw_list, replace_list):
        if tmp_sym in strip_raw_text:
            strip_raw_text = strip_raw_text.replace(tmp_sym, tar_sym)

    # replace operatorname
    if "\\operatorname{" in strip_raw_text:
        strip_raw_text = replace_any(strip_raw_text, '\\operatorname')

    # replace text
    if "\\text{" in strip_raw_text:
        strip_raw_text = replace_any(strip_raw_text, '\\text')

    # replace mathrm
    if "\\mathrm{" in strip_raw_text:
        strip_raw_text = replace_any(strip_raw_text, '\\mathrm')

    if "\\mathbf{" in strip_raw_text:
        strip_raw_text = replace_any(strip_raw_text, '\\mathbf{')

    return strip_raw_text


def union_words(strip_raw_text):
    # print(strip_raw_text)

    strip_raw_text = union_symbol(strip_raw_text)

    SYMBOL_BLACK_LIST = ["₁", "₂", "²", "³", "º", "˚", "ī", "；", "＜",
                         "＝", "＞", "？", "ｘ", "﹤", "！", "％",
                         "＆", "，", "４", "：", "￡", "｝", "｛",
                         "）", "（", "＋", "－", "≧", "…", "∣", "∶",
                         "`", "∆", "⇒", "•", "】", "㎡", "﹙", "﹚",
                         "﹛", "﹜", "﹢", "﹣", "﹥", "／", "Ａ", "Ｂ", "Ｃ",
                         "Ｄ", "Ｅ", "Ｆ", "Ｓ", "Ｘ", "［", "］", "｜", "～", "Δ",
                         "≦", "“", "”", "‘", "’", "．", "【", "？", "−", "ˉ", "‹",
                         "⇓"]
    SYMBOL_WHITE_LIST = ["_{1}", "_{2}", "^{2}", "^{3}", "°", "°", "i", ";", "<",
                         "=", ">", "?", "×", "<", "!", "%",
                         "&", ",", "4", ":", "£", "}", "{",
                         ")", "(", "+", "-", "≥", "...", "|", ":",
                         "'", "△", "→", "·", "]", "m^{2}", "(", ")",
                         "{", "}", "+", "-", ">", "/", "A", "B", "C",
                         "D", "E", "F", "S", "X", "[", "]", "|", "~", "△",
                         "≤", "\"", "\"", "\'", "\'", '.', "[", "?", "-", "¯", ":",
                         '↓']
    for tmp_sym, tar_sym in zip(SYMBOL_BLACK_LIST, SYMBOL_WHITE_LIST):
        if tmp_sym in strip_raw_text:
            strip_raw_text = strip_raw_text.replace(tmp_sym, tar_sym)

    order_symbols = ["⑴", "⑵", "⑶", "⑷", "⑸", "⑹", "⑺", "⑽", "⒀", "⑻", "⑾", "⑿", "⒁", "⒂", "⒃", "⒄", "⒅", "⒆", "⒇"]
    target_order_symbols = ["(1)", "(2)", "(3)", "(4)", "(5)", "(6)", "(7)", "(10)", "(13)", "(8)", "(11)", "(12)",
                            "(14)", "(15)", "(16)", "(17)", "(18)", "(19)", "(20)"]
    for tmp_sym, tar_sym in zip(order_symbols, target_order_symbols):
        if tmp_sym in strip_raw_text:
            strip_raw_text = strip_raw_text.replace(tmp_sym, tar_sym)

    # print(strip_raw_text)

    return strip_raw_text


def remove_stopkey(text_list):
    stop_words = ['.', '。', '，', ',', ';', '—', '、', '·', '$',
                  " ", "\n", "\t", u"\u3000", u"\xa0", u"\u200b"]

    new_list = []
    for word in text_list:
        if word not in stop_words:
            new_list.append(word)

    return new_list


def doc_bow(recognize_result, reference):
    '''

    :param recognize_result:  Student 's note , String
    :param reference:  Teacher 's note , String
    :return:
    '''

    # union
    reference = union_words(reference)

    src_text = recognize_result["text"]
    recog_text = ""
    for text_line in src_text:
        tmp_text = text_line["content"]
        recog_text += tmp_text
        recog_text += '\n'

    recog_text = union_words(recog_text)

    ## reference process
    corpora_documents = jieba.lcut(reference)
    corpora_documents = remove_stopkey(corpora_documents)

    corpora_documents = [corpora_documents]
    # print(corpora_documents)

    # 生成字典和向量语料
    dictionary = corpora.Dictionary(corpora_documents)
    corpus = [dictionary.doc2bow(text) for text in corpora_documents]

    # num_features代表生成的向量的维数（根据词袋的大小来定）
    similarity = Similarity('Similarity-index', corpus, num_features=4000, num_best=5)

    ## recognize process
    test_cut_raw_1 = jieba.lcut(recog_text)
    test_cut_raw_1 = remove_stopkey(test_cut_raw_1)
    # print('\n',test_cut_raw_1)

    test_corpus_1 = dictionary.doc2bow(test_cut_raw_1)

    if len(test_corpus_1) == 0:
        return 0
    # print(similarity[test_corpus_1])
    # print(similarity[test_corpus_1][0][1])
    return similarity[test_corpus_1][0][1]


def get_sim(recognize_json, ref_file):
    with open(recognize_json, 'r', encoding='utf-8') as f:
        recognize_result = json.load(f)
        # print(recognize_result)

    with open(ref_file, 'r', encoding='utf-8') as f2:
        reference = f2.read()

    sim = doc_bow(recognize_result, reference)

    return sim


def get_all_files(path):
    filelist = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith('.json'):
                filelist.append(os.path.join(root, name))
    print('There are %d files' % (len(filelist)))
    return filelist


def plot_sim(img_path, sim):
    img = cv2.imread(img_path)
    # h,w,_=img.shape
    p1 = (200, 200)
    cv2.putText(img, '%s: %.3f' % ('sim', sim), p1, cv2.FONT_HERSHEY_DUPLEX, 3.0, (255, 0, 255), 2)

    return img


if __name__ == '__main__':
    # get_sim('/workspace/houqi/GodEye/OCR/note_integrity/抄写_result/数学/tmp_val_5.jpg.json',
    #         '/workspace/houqi/GodEye/OCR/note_integrity/笔记demo文本txt/数学笔记demo-5年级-LaTeX.txt.bak')

    img_dir = '/workspace/houqi/GodEye/OCR/note_integrity/抄写'
    out_dir = '/workspace/houqi/GodEye/OCR/note_integrity/demo_result'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 数学
    math_files = get_all_files('/workspace/houqi/GodEye/OCR/note_integrity/抄写_result/数学')
    refer_math = '/workspace/houqi/GodEye/OCR/note_integrity/笔记demo文本txt/数学笔记demo-5年级-LaTeX.txt.bak'

    for i in math_files:
        img_name = os.path.basename(i).replace('.json', '')
        img_path = os.path.join(img_dir, img_name.replace('tmp_val_', ''))
        st = time.time()
        sim = get_sim(i, refer_math)
        sim_img = plot_sim(img_path, sim)
        end = time.time()
        cv2.imwrite(os.path.join(out_dir, img_name), sim_img)
        print('note:{} \t sim:{} \t time:{}'.format(os.path.basename(i), sim, end - st))

    # 语文
    math_files = get_all_files('/workspace/houqi/GodEye/OCR/note_integrity/抄写_result/语文')
    refer_math = '/workspace/houqi/GodEye/OCR/note_integrity/笔记demo文本txt/语文笔记demo-5年级.txt.bak'

    for i in math_files:
        img_name = os.path.basename(i).replace('.json', '')
        img_path = os.path.join(img_dir, img_name.replace('tmp_val_', ''))
        st = time.time()
        sim = get_sim(i, refer_math)
        sim_img = plot_sim(img_path, sim)
        end = time.time()
        cv2.imwrite(os.path.join(out_dir, img_name), sim_img)
        print('note:{} \t sim:{} \t time:{}'.format(os.path.basename(i), sim, end - st))
