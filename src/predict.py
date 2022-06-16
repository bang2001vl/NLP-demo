
from cmath import log
from pyvi import ViTokenizer, ViPosTagger
from joblib import load
from flask import Flask, request
from os.path import abspath
import re, string, codecs, json

path_replace_list = abspath("../input/customdata/replace-list.json")
with codecs.open(path_replace_list, "r", encoding="UTF-8") as f:
    replace_json = json.load(f)

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

# Load model
clf = load('./model.joblib') 
# X_test = ["Thằng này là lừa đảo", "Chương trình hay lắm", "Chương trình không hữu ích", "Chẳng đáng để làm"]

# BUILD API SERVER
app = Flask(__name__)

# Predict route
@app.route('/comment', methods=['POST'])
def predict_comment():
    try: 
        comments = list(request.get_json()["comments"])
        temp = []
        for x in comments:
            temp.append(normalize_text(x))
        rs = {}
        rs["result"] = True
        rs["received_comments"] = comments
        probabilities = clf.predict_proba(temp)
        rs["probabilities"] = probabilities.tolist()
        return rs
    except:
        rs = {}
        rs["result"] = False
        rs["errorMessage"] = "Catch exception on server"

# Testing route
@app.route('/test', methods=['GET'])
def get_test():
    return "Hello, world"
app.run('0.0.0.0', 5050)