import numpy as np

class CustomGenerator:
    def flow(self, X, y=None, batch_size=32, shuffle=True):
        if not y is None:
            assert X.shape[0] == y.shape[0]
        n_sample = X_shape[0]
        assert batch_size <= n_sample
        n_batch = n_sample // batch_size
        
        while True:
            indices = np.arange(n_sample)
            if shuffle:
                np.random.shuffle(indices)
            
            for i in range(n_batch):
                current_indices = indices[i*batch_size:(i+1)*batch_size]
                X_batch = (X[current_indices] / 255.0).astype(np.float32)
                if y is None:
                    yield X_batch
                else:
                    y_batch = (y[current_indices]).astype(np.float32)
                    yield X_batch, y_batch