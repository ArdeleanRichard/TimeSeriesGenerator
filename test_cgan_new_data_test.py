import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from tensorflow import keras
import tensorflow as tf

from parse_data.constants import MODS_FOLDER
from parse_data.read_traces import load_data_with_padded_traces_for_AE, load_data_speed_vs_pressure_small, load_data_speed_vs_pressure_full, load_data_speed_vs_pressure_new, load_data_speed_new_test_radu, load_data_speed_new_test_radu2, \
    load_data_speed_new_test_radu_shift_speed
from parse_data.utils import split_data_train_test
from pred_trace.tf_models import ConditionalGAN


speeds, traces, meta = load_data_speed_vs_pressure_new()
print(meta)

X_test, y_test = traces, speeds


batch_size = 32
epochs = 100
latent_dim = 100

data_dim = X_test.shape[1]
condition_dim = y_test.shape[1]

model_name = "cgan_radu_datafull2"
cgan = ConditionalGAN(model_name=model_name, latent_dim=latent_dim, condition_dim=condition_dim, data_dim=data_dim, gen_layer_config=[512, 256, 128, 256, 512], disc_layer_config=[512, 256, 128])
# cgan = ConditionalGAN(model_name=model_name, latent_dim=latent_dim, condition_dim=condition_dim, data_dim=X_train.shape[1], gen_layer_config=[1000, 1000, 1000, 1000, 1000], disc_layer_config=[512, 256, 128])
# cgan = ConditionalGAN(model_name=model_name, latent_dim=latent_dim, condition_dim=condition_dim, data_dim=X_train.shape[1], gen_layer_config=[128, 256, 512], disc_layer_config=[512, 256, 128])

cgan.compile()
cgan.load_weights_models()
generator = cgan.generator


def plot_generation(generator, y_test, sample_id):
    condition = y_test[sample_id].reshape(1, -1).astype(float)
    # print(condition)

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
    plt.savefig(MODS_FOLDER+f"{model_name}/top_{top_x}/gen{sample_id}")
    plt.close()



def save_results(generator, y_test):
    results = []
    for sample_id in range(len(y_test)):

        condition = y_test[sample_id].reshape(1, -1).astype(float)
        z = np.random.normal(size=(1, latent_dim))

        generated_data = generator(tf.concat([tf.cast(z, tf.float32), tf.cast(condition, tf.float32)], axis=1))
        results.append(generated_data[0])

    results = np.array(results)
    np.savetxt("results.csv", results, delimiter=",", fmt="%f")


top_x = 1000
losses, top_x_indexes = cgan.evaluate_performance_single(X_test, y_test, top_x=top_x)

# for sample_id in np.random.randint(0, len(X_test), 10):
# for sample_id in top_x_indexes:
#     plot_generation(generator, y_test, sample_id)
# plt.show()

print(np.unique(meta[top_x_indexes], axis=0))

# save_results(generator, y_test)











