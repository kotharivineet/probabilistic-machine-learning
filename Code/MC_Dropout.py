import numpy as np
import tensorflow as tf
from ECE import ECE
import matplotlib.pyplot as plt

# Generate toy regression data
np.random.seed(0)
X = np.linspace(-10,10,10000).reshape(-1, 1) 
# X = X[(X < -2) | (X > 2)].reshape(-1, 1)
y = np.zeros([*X.shape]) + 0.01 * np.random.randn(*X.shape)

# Define Monte Carlo Dropout model
class MCDOModel(tf.keras.Model):
    def __init__(self, dropout_rate):
        super(MCDOModel, self).__init__()
        self.dropout_rate = dropout_rate
        self.fc1 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.fc2 = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.fc1(inputs)
        x = self.dropout1(x, training=training)
        x = self.fc2(x)
        return x

# Function to obtain predictions and uncertainties using Monte Carlo Dropout
def mc_dropout_predictions(model, inputs, num_samples):
    predictions = np.zeros((num_samples, inputs.shape[0]))
    for i in range(num_samples):
        predictions[i] = model(inputs, training=True).numpy().flatten()
    mean_prediction = np.mean(predictions, axis=0)
    uncertainty = np.std(predictions, axis=0)
    return mean_prediction, uncertainty

# Configure training parameters
learning_rate = 0.001
dropout_rate = 0.2
num_epochs = 10
batch_size = 32
num_mc_samples = 100

# Create model instance
model = MCDOModel(dropout_rate)

# Define loss function and optimizer
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Compile the model
model.compile(optimizer=optimizer, loss=loss_fn)

# Training loop
for epoch in range(num_epochs):
    indices = np.random.permutation(X.shape[0])
    num_batches = X.shape[0] // batch_size
    for batch in range(num_batches):
        batch_indices = indices[batch*batch_size : (batch+1)*batch_size]
        X_batch = tf.convert_to_tensor(X[batch_indices], dtype=tf.float32)
        y_batch = tf.convert_to_tensor(y[batch_indices], dtype=tf.float32)

        with tf.GradientTape() as tape:
            predictions = model(X_batch, training=True)
            loss_value = loss_fn(y_batch, predictions)

        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_value.numpy():.4f}")

X_test = np.linspace(-15, 15, 1000).reshape(-1, 1)
y_truth = np.zeros([*X_test.shape]) + 0.002 * np.random.randn(*X_test.shape)

# Obtain predictions and uncertainties using Monte Carlo Dropout
mean_pred, uncertainty = mc_dropout_predictions(model, X_test, num_mc_samples)

########################### ECE and RP ####################################
ece, Acc, Conf = ECE(mean_pred, uncertainty*uncertainty, y_truth)
print(ece)
plt.figure()
plt.plot(Conf, Acc)
plt.plot([0,1],[0,1],linestyle='--')
plt.xlabel('Conf')
plt.ylabel('Acc')
plt.title('Reliability Plot for Monte-Carlo Dropout Toy Regression')
###########################################################################

# Plot the results

plt.figure(figsize=(10, 6))
plt.plot(X, y, 'b.', label='Training Data')
plt.plot(X_test, mean_pred, 'r', label='Mean Prediction')
plt.fill_between(X_test.flatten(), mean_pred - 2 * uncertainty, mean_pred + 2 * uncertainty, color='gray', alpha=0.4, label='Uncertainty')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Toy regression with Monte Carlo Dropout for Total Uncertainty Estimation')
plt.legend()
plt.show()