import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from tensorflow import keras
import tensorflow as tf

from parse_data.constants import MODS_FOLDER
from parse_data.read_traces import load_data_with_padded_traces_for_AE, load_data_speed_vs_pressure_small, load_data_speed_vs_pressure_full, load_data_speed_vs_pressure_new2_balanced
from parse_data.utils import split_data_train_test
from pred_trace.tf_models import ConditionalGAN


# params, asps, _ = load_data_speed_vs_pressure_full()
params, asps, _ = load_data_speed_vs_pressure_new2_balanced()


X_train, X_test, y_train, y_test = split_data_train_test(asps, params)


batch_size = 32
epochs = 100
latent_dim = 100

# Convert to tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train.astype(float)))
dataset = dataset.shuffle(1000).batch(batch_size)

condition_dim = y_train.shape[1]


model_name = "cgan_radu_datafull2"
cgan = ConditionalGAN(model_name=model_name, latent_dim=latent_dim, condition_dim=condition_dim, data_dim=X_train.shape[1], gen_layer_config=[512, 256, 128, 256, 512], disc_layer_config=[512, 256, 128])
# cgan = ConditionalGAN(model_name=model_name, latent_dim=latent_dim, condition_dim=condition_dim, data_dim=X_train.shape[1], gen_layer_config=[1000, 1000, 1000, 1000, 1000], disc_layer_config=[512, 256, 128])
# cgan = ConditionalGAN(model_name=model_name, latent_dim=latent_dim, condition_dim=condition_dim, data_dim=X_train.shape[1], gen_layer_config=[128, 256, 512], disc_layer_config=[512, 256, 128])


cgan.compile()
cgan.fit(dataset, epochs=epochs)
cgan.save_weights_models()
cgan.plot_losses()
cgan.evaluate_performance(y_test, X_test)
generator = cgan.generator


def plot_generation(generator, sample_id):
    condition = y_test[sample_id].reshape(1, -1).astype(float)
    print(condition)

    # Generate a random latent vector
    z = np.random.normal(size=(1, latent_dim))

    # Generate data from the generator
    generated_data = generator(tf.concat([tf.cast(z, tf.float32), tf.cast(condition, tf.float32)], axis=1))

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(X_test[sample_id], label="Trace", color='b')
    plt.plot(generated_data[0], label="Generated", color='r')

    # Labels and title
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Trace vs Generation")
    plt.legend()
    plt.grid()
    plt.savefig(MODS_FOLDER+f"{model_name}/gen{sample_id}")


for sample_id in np.random.randint(0, len(X_test), 10):
    plot_generation(generator, sample_id)
plt.show()