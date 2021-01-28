import numpy as np
import redisai as rai
from ml2rt import load_model, load_script
from sklearn.datasets import load_iris

con = rai.Client()

# Fetch & set data
iris = load_iris()
selectors = iris.data[:, 0] > 5  # the other share
X, y = iris.data[selectors], iris.target[selectors]
index = -1
inp = X[index].astype(np.float32).reshape((1, 4))
con.tensorset("input", inp)

# Load and save model
device = 'cpu'
model = load_model("logistic.onnx")
con.modelset("sklearn_model", 'onnx', device, model)

# Run Model
con.modelrun("sklearn_model", ["input"], ["output", "dummy"])
out = con.tensorget("output")
print(f" Predicted class: {out.item()}", f"Actual class: {y[index]}")


# Load, set and run monitoring script
script = load_script("script.py")
con.scriptset("monitoring", device, script)
con.scriptrun("monitoring", "data_drift", "input", "monitoring_out")
monitoring_out = con.tensorget("monitoring_out")
if monitoring_out.item() == 1.0:
    print(f"Unexpected input found")
elif monitoring_out.item() == 0.0:
    print("Input is in the right range")
else:
    print("Error with monitoring script")
