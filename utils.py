import numpy as np

def create_sequences(data, labels, seq_length, step=1):
    X, y = [], []
    for i in range(0, len(data) - seq_length, step):
        X.append(data[i:i + seq_length])
        y.append(labels[i:i + seq_length])

    # Pad or truncate sequences to ensure consistent length
    if len(X[-1]) < seq_length:
        remaining = seq_length - len(X[-1])
        X[-1] = np.pad(X[-1], ((0, remaining), (0, 0), (0, 0), (0, 0), (0, 0)), 'constant')
        y[-1] = np.pad(y[-1], ((0, remaining), (0, 0)), 'constant')
    elif len(X[-1]) > seq_length:
        X[-1] = X[-1][:seq_length]
        y[-1] = y[-1][:seq_length]

    return np.array(X), np.array(y)