"""Train the bundled MNIST MLP. Run with numpy, scipy and scikit-learn installed.
Dataset: https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
Usage: OPENBLAS_NUM_THREADS=2 python train_model.py /path/to/mnist.npz
"""
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

root = Path(__file__).resolve().parent
data = np.load(sys.argv[1])
x = data['x_train'].reshape(-1, 784).astype(np.float32) / 255
y = data['y_train']
test = data['x_test'].reshape(-1, 784).astype(np.float32) / 255
model = MLPClassifier(hidden_layer_sizes=(128,64), batch_size=256,
    max_iter=35, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=5, random_state=42, verbose=True)
with threadpool_limits(limits=2):
    model.fit(x,y)
    accuracy = float(model.score(test, data['y_test']))
layers=[]
weights=[]
offset=0
for w,b in zip(model.coefs_,model.intercepts_):
    w = w.T.astype('<f4').ravel()
    b = b.astype('<f4')
    layers.append({'input': len(w)//len(b), 'output': len(b), 'offset': offset})
    weights.extend([w,b]); offset += len(w)+len(b)
np.concatenate(weights).astype('<f4').tofile(root/'model.bin')
metadata = {'dataset':'MNIST', 'architecture':[784,128,64,10], 'layers':layers,
    'testAccuracy':accuracy, 'testSamples':len(test), 'trainingSamples':54000,
    'validationSamples':6000, 'epochs':int(model.n_iter_),
    'preprocessing':'pixel / 255; row-major grayscale', 'seed':42}
(root/'model.json').write_text(json.dumps(metadata,indent=2))
# A fixed representative sample for each digit supports numerical parity checks.
indices = [int(np.flatnonzero(data['y_test']==k)[0]) for k in range(10)]
with threadpool_limits(limits=2):
    fixture={'inputs':test[indices].tolist(), 'probabilities':model.predict_proba(test[indices]).tolist(), 'labels':list(range(10))}
if len(sys.argv) > 2:
    Path(sys.argv[2]).write_text(json.dumps(fixture))
print(json.dumps(metadata), flush=True)
