import numpy as np
import redisai as rai
from ml2rt import load_model
from sklearn.datasets import load_iris

iris = load_iris()
indices = iris.data[:, 0] < 5  # the other share
X, y = iris.data[indices], iris.target[indices]

model = load_model("logistic.onnx")

device = 'cpu'

con = rai.Client()
con.modelset("sklearn_model", 'onnx', device, model)
index = -1
dummydata = X[index].astype(np.float32).reshape((1, 4))

con.tensorset("input", dummydata)

# dummy output because by default sklearn logistic regression outputs
# value and probability. Since RedisAI doesn't support specifying required
# outputs now, we need to keep placeholders for all the default outputs.
con.modelrun("sklearn_model", ["input"], ["output", "dummy"])
outtensor = con.tensorget("output")
print(f" Predicted class: {outtensor.item()}", f"Actual class: {y[index]}")
