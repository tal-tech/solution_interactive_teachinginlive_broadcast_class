#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-10-17
'''

import re


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

    order_symbols = ["⑴", "⑵", "⑶", "⑷", "⑸", "⑹", "⑺", "⑽",
                     "⒀", "⑻", "⑾", "⑿", "⒁", "⒂", "⒃", "⒄", "⒅", "⒆", "⒇"]
    target_order_symbols = [
        "(1)",
        "(2)",
        "(3)",
        "(4)",
        "(5)",
        "(6)",
        "(7)",
        "(10)",
        "(13)",
        "(8)",
        "(11)",
        "(12)",
        "(14)",
        "(15)",
        "(16)",
        "(17)",
        "(18)",
        "(19)",
        "(20)"]
    for tmp_sym, tar_sym in zip(order_symbols, target_order_symbols):
        if tmp_sym in strip_raw_text:
            strip_raw_text = strip_raw_text.replace(tmp_sym, tar_sym)

    strip_raw_text = filter_mythtype(strip_raw_text)
    strip_raw_text = filter_special_token(strip_raw_text)
    return strip_raw_text


''' Filter special token. '''


def filter_special_token(strip_raw_text):
    # backup symbol: +-×÷≠=
    strip_raw_text = re.sub(
        r"[·①②③④⑤⑥⑦⑧⑨、。，,——`~!@#$^&*()|{}':;',\[\].<>/？?~！\\\&*%]",
        "",
        strip_raw_text)
    return strip_raw_text


def filter_mythtype(strip_raw_text):
    strip_raw_text = re.sub("%.*\n", "", strip_raw_text)
    strip_raw_text = re.sub("frac", "", strip_raw_text)
    return strip_raw_text


def union_symbol(strip_raw_text):

    raw_list = [
        "\\right",
        "\\left",
        "\\rm",
        "\\leqslant",
        "\\mathsf",
        "\\underbrace",
        "\\geqslant",
        "\\bigstar",
        "\\quad",
        "\\hline",
        "\\dfrac",
        "\\triangle",
        "\\Delta",
        "\\Rightarrow",
        "\\rightarrow",
        "\\alpha",
        "\\beta",
        "\\rho",
        "\\mu",
        "\\theta",
        "\\times",
        "\\div",
        "\\pi",
        "\\angle",
        "{}^\\circ",
        "^\\circ",
        "^{\\circ}",
        "\\cdots",
        "\\cdot",
        "\\ldots",
        "\\dots",
        "\\pm",
        "\\because",
        "\\therefore",
        "\\neq",
        "\\geq",
        "\\leq",
        "\\equiv",
        "\\approx",
        "\\Square",
        "\\square",
        "\\max",
        "\\min",
        "\\cos",
        "\\sin",
        "\\tan",
        "\\%",
        "\\_",
        "\\downarrow",
        "\\uparrow",
        "\\ast",
        "\\oplus",
        "\\sim",
        "\\bmod",
        "\\longrightarrow",
        "\\Downarrow",
        "\\Uparrow",
        "\\Leftrightarrow",
        "\\lambda",
        "arrow",
        "\\gamma",
        "\\begin{cases}",
        "\\end{cases}",
        "\\begin{aligned}",
        "\\end{aligned}",
        '\\ne']
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


def replace_any(strip_raw_text, replace_op):

    op_in = strip_raw_text.find(replace_op)
    while op_in >= 0:
        match_dict = get_match_dict(strip_raw_text)
        # print(op_in, match_dict)
        bra_st = op_in + len(replace_op)
        if bra_st in match_dict:
            strip_raw_text = strip_raw_text.replace(strip_raw_text[op_in:match_dict[bra_st] + 1],
                                                    strip_raw_text[bra_st + 1:match_dict[bra_st]])
        else:
            strip_raw_text = strip_raw_text.replace(
                strip_raw_text[op_in:bra_st + 1], "")

        op_in = strip_raw_text.find(replace_op)

    return strip_raw_text


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
