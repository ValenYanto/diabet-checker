from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

def lvq_fit(train, target, learn_rate, b, max_epoch):
    label, train_idx = np.unique(target, return_index=True)
    weight = train[train_idx].astype(np.float64)

    # Hapus instance yang sudah digunakan untuk inisialisasi weight
    train = np.delete(train, train_idx, axis=0)
    target = np.delete(target, train_idx)

    epoch = 0

    while epoch < max_epoch:
        for i, x in enumerate(train):
            distance = [np.sum((w - x) ** 2) for w in weight]
            min_index = np.argmin(distance)
            sign = 1 if target[i] == label[min_index] else -1
            weight[min_index] += sign * learn_rate * (x - weight[min_index])

        learn_rate *= b
        epoch += 1

    return weight, label

def lvq_predict(x, weight_label):
    weight, label = weight_label
    x = np.array(x, dtype=np.float64)  # Konversi input menjadi float64
    distance = [np.sum((w - x) ** 2) for w in weight]
    return label[np.argmin(distance)]

async def check_diabetes(data):
    train = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50],
                      [1, 85, 66, 29, 0, 26.6, 0.351, 31],
                      [8, 183, 64, 0, 0, 23.3, 0.672, 32],
                      [1, 89, 66, 23, 94, 28.1, 0.167, 21],
                      [0, 137, 40, 35, 168, 43.1, 2.288, 33],
                      [5, 116, 74, 0, 0, 25.6, 0.201, 30],
                      [3, 78, 50, 32, 88, 31, 0.248, 26],
                      [10, 115, 0, 0, 0, 35.3, 0.134, 29],
                      [2, 197, 70, 45, 543, 30.5, 0.158, 53],
                      [8, 125, 96, 0, 0, 0, 0.232, 54],
                      [4, 110, 92, 0, 0, 37.6, 0.191, 30],
                      [10, 168, 74, 0, 0, 38, 0.537, 34],
                      [10, 139, 80, 0, 0, 27.1, 1.441, 57],
                      [1, 189, 60, 23, 846, 30.1, 0.398, 59],
                      [5, 166, 72, 19, 175, 25.8, 0.587, 51],
                      [7, 100, 0, 0, 0, 30, 0.484, 32],
                      [0, 118, 84, 47, 230, 45.8, 0.551, 31],
                      [7, 107, 74, 0, 0, 29.6, 0.254, 31],
                      [1, 103, 30, 38, 83, 43.3, 0.183, 33],
                      [1, 115, 70, 30, 96, 34.6, 0.529, 32],
                      [3, 126, 88, 41, 235, 39.3, 0.704, 27],
                      [8, 99, 84, 0, 0, 35.4, 0.388, 50],
                      [7, 196, 90, 0, 0, 39.8, 0.451, 41],
                      [9, 119, 80, 35, 0, 29, 0.263, 29],
                      [11, 143, 94, 33, 146, 36.6, 0.254, 51],
                      [10, 125, 70, 26, 115, 31.1, 0.205, 41],
                      [7, 147, 76, 0, 0, 39.4, 0.257, 43],
                      [1, 97, 66, 15, 140, 23.2, 0.487, 22],
                      [13, 145, 82, 19, 110, 22.2, 0.245, 57],
                      [5, 117, 92, 0, 0, 34.1, 0.337, 38],
                      [5, 109, 75, 26, 0, 36, 0.546, 60],
                      [3, 158, 76, 36, 245, 31.6, 0.851, 28],
                      [3, 88, 58, 11, 54, 24.8, 0.267, 22],
                      [6, 92, 92, 0, 0, 19.9, 0.188, 28],
                      [10, 122, 78, 31, 0, 27.6, 0.512, 45],
                      [4, 103, 60, 33, 192, 24, 0.966, 33],
                      [11, 138, 76, 0, 0, 33.2, 0.42, 35],
                      [9, 102, 76, 37, 0, 32.9, 0.665, 46],
                      [2, 90, 68, 42, 0, 38.2, 0.503, 27],
                      [4, 111, 72, 47, 207, 37.1, 1.39, 56],
                      [3, 180, 64, 25, 70, 34, 0.271, 26],
                      [7, 133, 84, 0, 0, 40.2, 0.696, 37],
                      [7, 106, 92, 18, 0, 22.7, 0.235, 48],
                      [9, 171, 110, 24, 240, 45.4, 0.721, 54],
                      [7, 159, 64, 0, 0, 27.4, 0.294, 40],
                      ])
    target = np.array(["diabetes", 
                       "tidak diabetes", 
                       "diabetes", 
                       "tidak diabetes", 
                       "diabetes", 
                       "tidak diabetes", 
                       "diabetes",
                       "tidak diabetes", 
                       "diabetes", 
                       "diabetes", 
                       "tidak diabetes", 
                       "diabetes", 
                       "tidak diabetes", 
                       "diabetes", 
                       "diabetes", 
                       "diabetes", 
                       "diabetes", 
                       "diabetes", 
                       "tidak diabetes", 
                       "diabetes",
                       "tidak diabetes",
                       "tidak diabetes",
                       "diabetes",
                       "diabetes",
                       "diabetes",
                       "diabetes",
                       "diabetes",
                       "tidak diabetes",
                       "tidak diabetes",
                       "tidak diabetes",
                       "tidak diabetes",
                       "diabetes",
                       "tidak diabetes",
                       "tidak diabetes",
                       "tidak diabetes",
                       "tidak diabetes",
                       "tidak diabetes",
                       "diabetes",
                       "diabetes",
                       "diabetes",
                       "tidak diabetes",
                       "tidak diabetes",
                       "tidak diabetes",
                       "diabetes",
                       "tidak diabetes"])

    weight = lvq_fit(train, target, 0.1, 0.5, 1000)
    output = lvq_predict(data, weight)
    return output

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/service')
def service():
    return render_template('service.html')

@app.route('/predict', methods=['POST'])
async def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    prediction = await check_diabetes([data['JumlahKehamilan'], data['KadarGula'], data['TekananDarah'], data['KetebalanLipatanKulit'], data['JumlahInsulin'], data['BeratBadan'], data['RiwayatDiabetes'], data['Umur']])
    print(prediction)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
