from __future__ import print_function, division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from keras.models import Model
from keras.layers import Input, Embedding, Dot, Add, Flatten, Dense, Concatenate
from keras.layers import Activation
from keras.optimizers import SGD

df = pd.read_csv(
    r"C:\Users\vitor\OneDrive\Área de Trabalho\RecSys_Course\large_files\movielens-20m-dataset\edited_rating.csv"
)

N = df.userId.max() + 1
M = df.movie_idx.max() + 1

df = shuffle(df)
cutoff = int(0.8 * len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

K = 10
mu = df_train.rating.mean()
epochs = 15
reg = 0.0


u = Input(shape=(1,))
m = Input(shape=(1,))
u_embedding = Embedding(N, K)(u)
m_embedding = Embedding(M, K)(m)


u_bias = Embedding(N, 1)(u)
m_bias = Embedding(M, 1)(m)
x = Dot(axes=2)([u_embedding, m_embedding])
x = Add()([x, u_bias, m_bias])
x = Flatten()(x)


u_embedding = Flatten()(u_embedding)
m_embedding = Flatten()(m_embedding)
y = Concatenate()([u_embedding, m_embedding])
y = Dense(400)(y)
y = Activation("elu")(y)
y = Dense(1)(y)


x = Add()([x, y])

model = Model(inputs=[u, m], outputs=x)
model.compile(
    loss="mse",
    optimizer=SGD(lr=0.08, momentum=0.9),
    metrics=["mse"],
)

r = model.fit(
    x=[df_train.userId.values, df_train.movie_idx.values],
    y=df_train.rating.values - mu,
    epochs=epochs,
    batch_size=128,
    validation_data=(
        [df_test.userId.values, df_test.movie_idx.values],
        df_test.rating.values - mu,
    ),
)


predicted_rating = model.predict([np.array([36003]), np.array([1])]).flatten()[0]

print("Predicted rating for user 36003 and movie 6:", predicted_rating)
model.save("keras_recommendation_model.h5")

plt.plot(r.history["loss"], label="train loss")
plt.plot(r.history["val_loss"], label="test loss")
plt.legend()
plt.show()

plt.plot(r.history["mse"], label="train mse")
plt.plot(r.history["val_mse"], label="test mse")
plt.legend()
plt.show()
