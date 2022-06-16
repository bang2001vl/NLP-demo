# -*- coding: utf-8 -*-
from __future__ import print_function
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from pyvi import ViTokenizer, ViPosTagger
from joblib import dump
import re
import string
import codecs
import json

# Từ điển tích cực, tiêu cực, phủ định
path_pos = "../input/customdata/pos.txt"
path_nag = "../input/customdata/nag.txt"
path_not = "../input/customdata/phu_dinh.txt"
path_replace_list = "../input/customdata/replace-list.json"
# File dataset
path_native_train = "../input/customdata/train.txt"
path_add_train1 = "../input/customdata/train_nor_811-format.txt"
path_add_train2 = "../input/customdata/test_nor_811-format.txt"

nag_list = []
with codecs.open(path_nag, "r", encoding="UTF-8") as f:
    for n in f.readlines():
        nag_list.append(n.replace("\n", "").replace(" ", "_"))

pos_list = []
with codecs.open(path_pos, "r", encoding="UTF-8") as f:
    for n in f.readlines():
        pos_list.append(n.replace("\n", "").replace(" ", "_"))

not_list = []
with codecs.open(path_not, "r", encoding="UTF-8") as f:
    for n in f.readlines():
        not_list.append(n.replace("\n", "").replace(" ", "_"))

with codecs.open(path_replace_list, "r", encoding="UTF-8") as f:
    replace_json = json.load(f)

# Hàm chuyển thành câu không dấu
def no_marks(s):
    VN_CHARS_LOWER = "ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđð"
    VN_CHARS_UPPER = "ẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸÐĐ"
    VN_CHARS = VN_CHARS_LOWER + VN_CHARS_UPPER
    __INTAB = [ch for ch in VN_CHARS]
    __OUTTAB = "a" * 17 + "o" * 17 + "e" * 11 + "u" * 11 + "i" * 5 + "y" * 5 + "d" * 2
    __OUTTAB += "A" * 17 + "O" * 17 + "E" * 11 + "U" * 11 + "I" * 5 + "Y" * 5 + "D" * 2
    __r = re.compile("|".join(__INTAB))
    __replaces_dict = dict(zip(__INTAB, __OUTTAB))
    result = __r.sub(lambda m: __replaces_dict[m.group(0)], s)
    return result

# Hàm tiền xử lí text
def normalize_text(text):
    # Remove các ký tự kéo dài: vd: đẹppppppp -> đẹp
    text = re.sub(
        r"([A-Z])\1+", lambda m: m.group(1).upper(), text, flags=re.IGNORECASE
    )
    
    # Remove các khoảng trắng lặp lại: vd: "a   a" => "a a"
    text = re.sub(' +', ' ', text)
    text = text.strip()

    # Chuyển thành chữ thường
    text = text.lower()

    # Thay một số từ lóng, từ viết tắt, icons, tiếng anh thành từ có nghĩa
    for key in replace_json:
        text = text.replace(key, replace_json[key])

    # Chuyển dấu câu thành space vd: a,b.c -> a b c
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    text = text.translate(translator)

    # Chia câu thành các token vd: "Trường đại học bách khoa hà nội" -> ['Trường', 'đại_học', 'bách_khoa', 'hà_nội']
    tokens, tags = ViPosTagger.postagging(ViTokenizer.tokenize(text))

    # Gom câu lại
    text = " ".join(tokens)
    
    # remove nốt những ký tự thừa thãi
    text = text.replace('"', " ")
    text = text.replace("️", "")
    text = text.replace("🏻", "")

    return text

# Lấy dữ liệu từ file train
def load_data_from_file(filename, is_train=True):
    lst = []
    sample = []

    with open(filename, "r") as file:
        for line in file:
            if (line == "\n") and (len(sample) > 2) and (sample[-1].isnumeric()):
                lst.append([sample[0], "  ".join(sample[1:-1]), sample[-1]])
                sample = []
            else:
                sample.append(line.replace("\n", ""))
    return lst

# Dup
def prepare_data_set(x_set, y_set):
    X, y = [], []
    for document, topic in zip(list(x_set), list(y_set)):
        # Thêm dữ liệu đã chuẩn hóa
        document = normalize_text(document)
        X.append(document.strip())
        y.append(topic)
        # Thêm dữ liệu đã bỏ dấu
        X.append(no_marks(document))
        y.append(topic)
    return X, y

# Luồng chạy chính
steps = []
steps.append(
    (
        "CountVectorizer",
        CountVectorizer(ngram_range=(1, 5), stop_words=("rằng", "thì", "là", "mà"), analyzer=str.split),
    )
)
steps.append(
    (
        "tfidf",
        TfidfTransformer(use_idf=False, sublinear_tf=True),
    )
)
steps.append(("classifier", LogisticRegression(max_iter=500)))
# steps.append(("classifier", LinearSVC(max_iter=7000)))
clf = Pipeline(steps)

for epo in range(3):
    data_columns = list(["id", "comment", "label"])

    # Dữ liệu native
    train_data = pd.DataFrame(load_data_from_file(path_native_train), columns=data_columns)
    print(train_data)

    # Dữ liệu comment 1
    train_data = pd.concat(
        [
            train_data,
            pd.DataFrame(load_data_from_file(path_add_train1), columns=data_columns),
        ]
    ).reset_index(drop=True)
    print(train_data)

    # Dữ liệu comment 2
    train_data = pd.concat(
        [
            train_data,
            pd.DataFrame(load_data_from_file(path_add_train2), columns=data_columns),
        ]
    ).reset_index(drop=True)
    print(train_data)

    # train_data = pd.DataFrame([
    #     ["1", "0", "Chiến dịch rất ý nghĩa"],
    #     ["2", "1", "Chiến dịch không xứng đáng để tham gia"],
    #     ["3", "1", "Chiến dịch không đáng để tham gia"],
    #     ["4", "0", "Chiến dịch rất rất ý nghĩa"],
    # ],columns=list(['id','label','comment']))

    X_train, X_test, y_train, y_test = train_test_split(
        train_data.comment, train_data.label, test_size=0.3
    )

    # Thêm mẫu bằng cách lấy trong từ điển Sentiment (nag/pos)
    dictionary_data = []
    for index, word in enumerate(pos_list):
        dictionary_data.append(["pos" + str(index), word, "0"])

    for index, word in enumerate(nag_list):
        dictionary_data.append(["nag" + str(index), word, "1"])

    for index1, word1 in enumerate(not_list):
        for index2, word2 in enumerate(pos_list):
            dictionary_data.append(
                ["notpos" + str(index1) + "_" + str(index2), word1 + " " + word2, "1"]
            )
        for index2, word2 in enumerate(nag_list):
            dictionary_data.append(
                ["notnag" + str(index1) + "_" + str(index2), word1 + " " + word2, "0"]
            )

    # Thêm dữ liệu vào tập train
    dictionary_frame = pd.DataFrame(dictionary_data, columns=data_columns)
    X_train = X_train.append(dictionary_frame.comment)
    y_train = y_train.append(dictionary_frame.label)
    print(len(X_train))

    X_train, y_train = prepare_data_set(X_train, y_train)
    X_test, y_test = prepare_data_set(X_test, y_test)

    print("Starting: Fit model")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report1 = metrics.classification_report(y_test, y_pred, labels=['1', '0'], digits=3)

    # #CROSS VALIDATION
    cross_score = cross_val_score(clf, X_train, y_train, cv=5)

    # #REPORT
    print('EPOCH = ' + str(epo))
    print("DATASET LEN %d" % (len(X_train)))
    print("TRAIN 70/30 \n\n", report1)
    print(u'-'*100)

# #TRAIN 100% (Huấn luyện với toàn bộ dữ liệu)
train_data = pd.concat([train_data, dictionary_frame]).reset_index(drop=True)

X_train, y_train = prepare_data_set(train_data.comment, train_data.label)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_train)
report2 = metrics.classification_report(y_train, y_pred, labels=['1', '0'], digits=3)

# #REPORT
print('EPOCH = ' + str(epo))
print("DATASET LEN %d" % (len(X_train)))
print("TRAIN 100% \n\n", report2)
print(
    "CROSSVALIDATION 5 FOLDS: %0.4f (+/- %0.4f)"
    % (cross_score.mean(), cross_score.std() * 2)
)

# # #SAVE MODEL FILE
print("Save model")
dump(clf, "./model.joblib")
