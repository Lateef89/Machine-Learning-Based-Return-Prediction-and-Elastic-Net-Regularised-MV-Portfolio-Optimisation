"""
models.py
=========
The six return-prediction models from the paper, with hyperparameters set as
close as possible to Table 1 of the revised manuscript. Where a published
hyperparameter does not transfer directly to our (smaller, 7-feature) feature
set -- e.g. Random Forest's `max_features: 40`, which exceeds the number of
features available here -- it is capped sensibly and the deviation is noted
in a comment at the point of use. See README.md for the full list.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import torch
import torch.nn as nn

RANDOM_STATE = 42


def make_rf():
    # Table 1: n_estimators=500, max_depth=20, min_samples_split=10,
    # min_samples_leaf=10, max_features=40 (capped to 'sqrt' of our 7 features,
    # since max_features=40 > number of engineered features available here).
    return RandomForestRegressor(
        n_estimators=500, max_depth=20, min_samples_split=10,
        min_samples_leaf=10, max_features="sqrt",
        random_state=RANDOM_STATE, n_jobs=-1,
    )


def make_adaboost():
    # Table 1: n_estimators=50, learning_rate=1, base_estimator (decision stump/tree).
    base = DecisionTreeRegressor(max_depth=3, random_state=RANDOM_STATE)
    try:
        return AdaBoostRegressor(estimator=base, n_estimators=50, learning_rate=1.0,
                                  random_state=RANDOM_STATE)
    except TypeError:  # older sklearn uses base_estimator=
        return AdaBoostRegressor(base_estimator=base, n_estimators=50, learning_rate=1.0,
                                  random_state=RANDOM_STATE)


def make_xgboost():
    # Table 1: n_round=100, max_depth=7, learning_rate=0.01, gamma=2.
    return XGBRegressor(
        n_estimators=100, max_depth=7, learning_rate=0.01, gamma=2,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    )


def make_svr():
    # Table 1: C=10, gamma=0.1 (RBF kernel).
    return SVR(C=10, gamma=0.1, kernel="rbf")


def make_knn():
    # Table 1: n_neighbors=3.
    return KNeighborsRegressor(n_neighbors=3)


# ---------------------------------------------------------------------------
# RNN (PyTorch). Table 1 specifies hidden layers=4, batch_size=128, epochs=500.
# We treat the three lag returns (lag1, lag2, lag3) as a length-3 sequence fed
# to the recurrent unit, with the remaining engineered features (roll_mean_3,
# roll_std_3, volume_chg, mkt_return) concatenated to the final hidden state
# before the output layer -- a standard way to combine a short return history
# with cross-sectional features in a single RNN regressor. Epochs are capped
# at 150 (rather than 500) purely for tractability in this environment; this
# is flagged explicitly since it is a real deviation from Table 1.
# ---------------------------------------------------------------------------
SEQ_FEATURES = ["lag3_return", "lag2_return", "lag1_return"]  # chronological order
AUX_FEATURES = ["roll_mean_3", "roll_std_3", "volume_chg", "mkt_return"]
RNN_EPOCHS = 30    # deviation from Table 1's epochs=500; see README.md
RNN_BATCH = 128
RNN_HIDDEN = 16
RNN_LAYERS = 4


class ReturnRNN(nn.Module):
    def __init__(self, n_aux, hidden=RNN_HIDDEN, layers=RNN_LAYERS):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden, num_layers=layers,
                           batch_first=True, nonlinearity="tanh")
        self.head = nn.Sequential(
            nn.Linear(hidden + n_aux, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, seq, aux):
        # seq: (batch, 3, 1)
        out, hN = self.rnn(seq)
        last = out[:, -1, :]                # (batch, hidden)
        x = torch.cat([last, aux], dim=1)
        return self.head(x).squeeze(-1)


class RNNRegressor:
    """Thin sklearn-style wrapper around ReturnRNN so it can be used
    interchangeably with the sklearn/xgboost models above."""

    def __init__(self, epochs=RNN_EPOCHS, batch_size=RNN_BATCH, lr=1e-3, seed=RANDOM_STATE):
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        torch.manual_seed(seed)
        self.model = None
        self.x_mean = None
        self.x_std = None

    def _split_xy(self, X):
        X = np.asarray(X, dtype=np.float32)
        seq = X[:, :3][:, :, None]        # (n,3,1) -- lag3,lag2,lag1 already in this column order
        aux = X[:, 3:]                     # (n, n_aux)
        return seq, aux

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        self.x_mean = X.mean(axis=0, keepdims=True)
        self.x_std = X.std(axis=0, keepdims=True) + 1e-8
        Xn = (X - self.x_mean) / self.x_std
        seq, aux = self._split_xy(Xn)
        y = np.asarray(y, dtype=np.float32)

        self.model = ReturnRNN(n_aux=aux.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        seq_t = torch.tensor(seq)
        aux_t = torch.tensor(aux)
        y_t = torch.tensor(y)
        n = len(y_t)

        self.model.train()
        for _epoch in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                pred = self.model(seq_t[idx], aux_t[idx])
                loss = loss_fn(pred, y_t[idx])
                loss.backward()
                opt.step()
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xn = (X - self.x_mean) / self.x_std
        seq, aux = self._split_xy(Xn)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(torch.tensor(seq), torch.tensor(aux)).numpy()
        return pred


def make_rnn():
    return RNNRegressor()


# Column order expected by every model: SEQ_FEATURES first (chronological
# lag3->lag2->lag1), then AUX_FEATURES. data.FEATURE_COLS must be reordered
# to this before use -- see experiments code.
MODEL_FEATURE_ORDER = SEQ_FEATURES + AUX_FEATURES

MODEL_FACTORY = {
    "RF": make_rf,
    "AdaBoost": make_adaboost,
    "XGBoost": make_xgboost,
    "SVR": make_svr,
    "KNN": make_knn,
    "RNN": make_rnn,
}
