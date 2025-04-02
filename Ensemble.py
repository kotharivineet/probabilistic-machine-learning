import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ECE import ECE

# Generate toy regression data
np.random.seed(0)
# y_train = np.linspace(-5, 5, 100).reshape(-1, 1)
X_train = np.linspace(-10,10,10000).reshape(-1, 1) 
X_train = X_train[(X_train < -1) | (X_train > 1)].reshape(-1, 1)
y_train = np.zeros([*X_train.shape]) + 0.002 * np.random.randn(*X_train.shape)

# Define the ensemble model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(2)
    ])
    return model

def negative_log_likelihood(y_true, y_pred):
    mean = y_pred[:, 0]
    log_var = y_pred[:, 1]
    precision = tf.exp(-log_var)
    return 0.5 * (tf.math.log(2 * np.pi) + log_var + tf.square(y_true - mean) * precision)

num_models = 5
models = []
for _ in range(num_models):
    model = create_model()
    model.compile(loss=negative_log_likelihood, optimizer='adam')
    models.append(model)

# Train the ensemble models
epochs = 200
for model in models:
    model.fit(X_train, y_train, epochs=epochs, verbose=0)

# Predict with ensemble models
X_test = np.linspace(-15, 15, 1000).reshape(-1, 1)
y_truth = np.zeros([*X_test.shape]) + 0.002 * np.random.randn(*X_test.shape)
y_pred = np.zeros((X_test.shape[0],2, num_models))
for i, model in enumerate(models):
    y_pred[:,:, i] = model.predict(X_test)

# Calculate epistemic uncertainty
y_mean = np.mean(y_pred[:,0,:], axis=1)
y_std = np.std(y_pred[:,0,:], axis=1)

########################### ECE and RP ####################################
ece, Acc, Conf = ECE(y_mean, y_std*y_std, y_truth)
print(ece)
plt.figure()
plt.plot(Conf, Acc)
plt.plot([0,1],[0,1],linestyle='--')
plt.xlabel('Conf')
plt.ylabel('Acc')
plt.title('Reliability Plot for Ensemble Toy Regression')
###########################################################################

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(X_train, y_train, 'bo', label='Training data')
plt.plot(X_test, y_mean, '#ff7f0e', label='Mean Prediction')
plt.fill_between(X_test.flatten(), y_mean - 2 * y_std, y_mean + 2 * y_std,
                 color='b', alpha=0.1, label='Epistemic Uncertainty')
# plt.xlabel('X')
# plt.ylabel('y')
# # plt.ylim([-0.02,0.02])
# plt.title('Training Dataset')
# plt.legend()
plt.show()