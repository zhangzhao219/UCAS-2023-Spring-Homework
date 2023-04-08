from flask import Flask, render_template, request
from mel_trans import Waveform
import yaml

from audio_branch.infer import main

app = Flask(__name__)



@app.route('/wav', methods=['GET', 'POST'])
def Wav_1():

    text1 = ""
    text2 = ""
    text3 = " "
    text4 = " "
    audio_filename = "infer.wav"

    bt_a1 = request.values.get('submit1')
    bt_a2 = request.values.get('submit2')
    bt_a3 = request.values.get('submit')

    if bt_a1 == "Upload Text":
        text1 = request.values.get("story")
        return render_template('wav.html',score = "score", text1 = text1, text2 = text2, text3 = text3,text4 = text4,  name2 = audio_filename)
    elif bt_a2 == "Upload Audio":
        text1 = request.values.get("story")
        file2 = request.files["file2"]
        file2.save("./wav/to_infer.wav")
        audio_filename = file2.filename
        Waveform("./wav/to_infer.wav","to_infer.jpg")
        return render_template('wav.html',score = "score", text1 = text1, text2 = text2, text3 = text3,text4 = text4,  name2 = audio_filename)
    elif bt_a3 == "submit":
        text1 = request.values.get("story")
        with open("./infer_config.yaml", "r") as f:
            configs = yaml.safe_load(f)
        score = main(configs)
        cls2idx_mapping = {0:"中性", 1:"开心", 2:"生气", 3:"悲伤"}
        text2 = cls2idx_mapping[score]
        text3 = "a"
        text4 = "b"
        return render_template('wav.html',score = score, text1 = text1, text2 = text2, text3 = text3,text4 = text4,  name2 = audio_filename)
    else:
        return render_template('wav.html',score = "score", text1 = text1, text2 = text2, text3 = text3,text4 = text4,  name2 = audio_filename)

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)