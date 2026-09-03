# neede this to dump mnist into a flat binary to make c loader be able to read it. C cant do gz or pickle easily (or pickle at all i think?)
import gzip, pickle, struct
import numpy as np
 
with gzip.open('../data/mnist.pkl.gz', 'rb') as f:
    splits = pickle.load(f, encoding='latin1')
 
for name, (X, y) in zip(("train", "valid", "test"), splits):
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.uint8)
    n, d = X.shape
    with open("../data/%s.bin" % name, "wb") as f:
        f.write(struct.pack("<III", 0x54534E4D, n, d))
        f.write(X.tobytes())
        f.write(y.tobytes())
    print(name, n, d)
 