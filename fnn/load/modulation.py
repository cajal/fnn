from fnn.model import modulations
from fnn.load import load


def flat_lstm(in_features, out_features, hidden_features, init_input, init_forget, dropout):

    return modulations.FlatLstm(
        in_features=in_features,
        out_features=out_features,
        hidden_features=hidden_features,
        init_input=init_input,
        init_forget=init_forget,
        dropout=dropout,
    )


def modulation(modulation_id):
    options = {
        "0313510d8e705d8869e59620bcb57aa4": [flat_lstm, [16, 4, 16, -1, 1, 0], {}],
        "7eb30b53e043fec5c354e91b2d16d733": [flat_lstm, [16, 4, 16, -1, 1, 0.1], {}],
        "da07a7da5d1610a0e10ed4d430ddec21": [flat_lstm, [16, 4, 16, 0, 0, 0.1], {}],
    }
    return load(options, modulation_id)
