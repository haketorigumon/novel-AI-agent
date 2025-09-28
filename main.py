#!/usr/bin/env python3
import sys
import numpy as np


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


class RNN:
    def __init__(self, vocab_size, hidden_size=100, seq_length=25, learning_rate=1e-1):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate

        # 参数初始化
        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(vocab_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))

        # Adagrad 缓存
        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mWhy = np.zeros_like(self.Why)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)

    def lossFun(self, inputs, targets, hprev):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0

        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(self.Wxh @ xs[t] + self.Whh @ hs[t - 1] + self.bh)
            ys[t] = self.Why @ hs[t] + self.by
            ps[t] = softmax(ys[t])
            loss += -np.log(ps[t][targets[t], 0])

        # 反向传播
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])

        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            dWhy += dy @ hs[t].T
            dby += dy
            dh = self.Why.T @ dy + dhnext
            dhraw = (1 - hs[t] ** 2) * dh
            dbh += dhraw
            dWxh += dhraw @ xs[t].T
            dWhh += dhraw @ hs[t - 1].T
            dhnext = self.Whh.T @ dhraw

        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]

    def sample(self, h, seed_ix, n):
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        for t in range(n):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            y = self.Why @ h + self.by
            p = softmax(y)
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes

    def update(self, grads):
        dWxh, dWhh, dWhy, dbh, dby = grads
        for param, dparam, mem in zip(
            [self.Wxh, self.Whh, self.Why, self.bh, self.by],
            [dWxh, dWhh, dWhy, dbh, dby],
            [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby],
        ):
            mem += dparam * dparam
            param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)


def main():
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = f.read()

    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print(f"data has {data_size} characters, {vocab_size} unique.")

    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    rnn = RNN(vocab_size)
    n, p = 0, 0
    smooth_loss = -np.log(1.0 / vocab_size) * rnn.seq_length
    hprev = np.zeros((rnn.hidden_size, 1))

    while True:
        if p + rnn.seq_length + 1 >= len(data) or n == 0:
            hprev = np.zeros((rnn.hidden_size, 1))
            p = 0

        inputs = [char_to_ix[ch] for ch in data[p : p + rnn.seq_length]]
        targets = [char_to_ix[ch] for ch in data[p + 1 : p + rnn.seq_length + 1]]

        if n % 100 == 0:
            sample_ix = rnn.sample(hprev, inputs[0], 200)
            txt = "".join(ix_to_char[ix] for ix in sample_ix)
            print(f"----\n{txt}\n----")

        loss, *grads, hprev = rnn.lossFun(inputs, targets, hprev)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        if n % 100 == 0:
            print(f"iter {n}, loss: {smooth_loss:.4f}")

        rnn.update(grads)
        p += rnn.seq_length
        n += 1


if __name__ == "__main__":
    main()
