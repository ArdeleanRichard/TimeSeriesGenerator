import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers, Model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from parse_data.constants import MODS_FOLDER
from pred_trace.metrics_trace import smoothed_dtw, frechet_trace_distance, piecewise_trend_similarity, one_d_ssim


class Generator(keras.Sequential):
    def __init__(self, latent_dim, condition_dim, output_dim, layers_config, activation):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        self.add(layers.InputLayer(shape=(latent_dim + condition_dim,)))

        for units in layers_config:
            self.add(layers.Dense(units, activation=activation))

        self.add(layers.Dense(output_dim))

    def call(self, inputs, training=False):
        return super().call(inputs, training=training)


class Discriminator(keras.Sequential):
    def __init__(self, input_dim, condition_dim, layers_config, activation):
        super().__init__()
        self.add(layers.InputLayer(shape=(input_dim + condition_dim,)))

        for units in layers_config:
            self.add(layers.Dense(units, activation=activation))

        self.add(layers.Dense(1, activation='sigmoid'))

    def call(self, inputs, training=False):
        return super().call(inputs, training=training)


class ConditionalGAN(keras.Model):
    def __init__(self, model_name, latent_dim, condition_dim, data_dim, gen_layer_config, disc_layer_config, activation="relu", lr=0.0002, beta_1=0.5):
        super(ConditionalGAN, self).__init__()

        self.model_name = model_name

        self.generator = Generator(latent_dim, condition_dim, data_dim, gen_layer_config, activation)
        self.discriminator = Discriminator(data_dim, condition_dim, disc_layer_config, activation)

        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.data_dim = data_dim

        self.lr = lr
        self.beta_1 = beta_1


    def compile(self):
        super(ConditionalGAN, self).compile()
        self.d_optimizer = keras.optimizers.Adam(self.lr, self.beta_1)
        self.G_trace_optimizer = keras.optimizers.Adam(self.lr, self.beta_1)
        self.loss_fn = keras.losses.BinaryCrossentropy()

    def save_models(self):
        """Save generator and discriminator models."""
        os.makedirs(MODS_FOLDER, exist_ok=True)
        self.generator.save(os.path.join(MODS_FOLDER, f"{self.model_name}_generator.keras"))
        self.discriminator.save(os.path.join(MODS_FOLDER, f"{self.model_name}_discriminator.keras"))

    def save_weights_models(self):
        os.makedirs(MODS_FOLDER, exist_ok=True)
        self.generator.save_weights(os.path.join(MODS_FOLDER, f"{self.model_name}_generator.weights.h5"))
        self.discriminator.save_weights(os.path.join(MODS_FOLDER, f"{self.model_name}_discriminator.weights.h5"))

    def load_weights_models(self):
        """Save generator and discriminator models."""
        os.makedirs(MODS_FOLDER, exist_ok=True)
        self.generator.load_weights(os.path.join(MODS_FOLDER, f"{self.model_name}_generator.weights.h5"))
        self.discriminator.load_weights(os.path.join(MODS_FOLDER, f"{self.model_name}_discriminator.weights.h5"))

    def load_models(self):
        """Save generator and discriminator models."""
        os.makedirs(MODS_FOLDER, exist_ok=True)
        self.generator = tf.keras.models.load_model(os.path.join(MODS_FOLDER, f"{self.model_name}_generator.keras"))
        self.discriminator = tf.keras.models.load_model(os.path.join(MODS_FOLDER, f"{self.model_name}_discriminator.keras"))


    def plot_losses(self):
        os.makedirs(MODS_FOLDER + f"{self.model_name}/", exist_ok=True)

        """Plot training loss history."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.history.history["d_loss"], label="Discriminator Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Losses")
        plt.savefig(MODS_FOLDER + f"{self.model_name}/loss_d.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(self.history.history["g_loss"], label="Generator Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Losses")
        plt.savefig(MODS_FOLDER + f"{self.model_name}/loss_g.png")
        plt.close()

    def evaluate_performance(self, test_data, test_conditions):
        """Evaluate model on test data and return the loss."""
        if not isinstance(test_data, tf.Tensor):
            test_data = tf.convert_to_tensor(test_data, dtype=tf.float32)
        if not isinstance(test_conditions, tf.Tensor):
            test_conditions = tf.convert_to_tensor(test_conditions, dtype=tf.float32)

        mask = self.create_mask(test_data)  # Ensure the mask is computed correctly
        latent_vectors = tf.random.normal((tf.shape(test_data)[0], self.latent_dim))

        fake_data = self.generator(tf.concat([tf.cast(latent_vectors, tf.float32), tf.cast(test_conditions, tf.float32)], axis=1), training=False)

        mse_loss = self.mse_loss(test_data, fake_data, mask)

        # Convert to NumPy safely if needed
        mse_loss_value = mse_loss.numpy() if isinstance(mse_loss, tf.Tensor) else mse_loss
        print(f"Test MSE Loss: {mse_loss_value:.4f}")

        with open(MODS_FOLDER + f"{self.model_name}/test_loss.txt", "w") as file:
            file.write(f"Test MSE Loss: {mse_loss:.4f}\n")

        return mse_loss

    def evaluate_performance_single(self, test_data, test_conditions, top_x = 100):
        """Evaluate model on test data and return the loss."""
        if not isinstance(test_data, tf.Tensor):
            test_data = tf.convert_to_tensor(test_data, dtype=tf.float32)
        if not isinstance(test_conditions, tf.Tensor):
            test_conditions = tf.convert_to_tensor(test_conditions, dtype=tf.float32)

        losses = []  # List to store loss values
        metrics = []
        alo = 0
        for (test_condition, test_sample) in zip(test_conditions, test_data):
            mask = self.create_mask([test_sample])  # Ensure the mask is computed correctly
            latent_vectors = tf.random.normal((1, self.latent_dim))  # Fix shape

            test_condition = tf.reshape(test_condition, (1, -1))  # Ensure correct shape
            fake_data = self.generator(tf.concat([latent_vectors, test_condition], axis=1), training=False)

            mse_loss = self.mse_loss([test_sample], fake_data, mask)
            mse_loss_value = mse_loss.numpy() if isinstance(mse_loss, tf.Tensor) else mse_loss

            # print(f"Test MSE Loss: {mse_loss_value:.4f}")
            losses.append(mse_loss_value)

            print(alo)
            score_dtw = smoothed_dtw(test_sample, fake_data[0])
            # score_dtw = 0
            # score_frechet = frechet_trace_distance(test_sample, fake_data[0])
            score_frechet = 0
            score_trend = piecewise_trend_similarity(test_sample, fake_data[0])
            # score_ssim = one_d_ssim(np.array(test_sample), np.array(fake_data[0]))
            score_ssim = 0

            metrics.append([score_dtw, score_frechet, score_trend, score_ssim])

            alo+=1

        # Write all losses to a file, one per line
        with open(MODS_FOLDER + f"{self.model_name}/test_samples_losses.txt", "w") as f:
            for id, loss in enumerate(losses):
                f.write(f"{loss}, {metrics[id][0]}, {metrics[id][1]}, {metrics[id][2]}, {metrics[id][3]}\n")

        losses = np.array(losses)
        top_x_indexes = np.argsort(losses)[-top_x:][::-1]

        print(f"Top {top_x} highest loss sample indexes:", top_x_indexes)

        # Write top 10 indexes to a separate file
        with open(MODS_FOLDER + f"{self.model_name}/test_samples_losses_top{top_x}_ids.txt", "w") as f:
            for idx in top_x_indexes:
                f.write(f"{idx}\n")
        with open(MODS_FOLDER + f"{self.model_name}/test_samples_losses_top{top_x}_losses.txt", "w") as f:
            for idx in top_x_indexes:
                f.write(f"{losses[idx]}\n")

        return losses, top_x_indexes

    def create_mask(self, real_data):
        """
        Create a binary mask where 1 indicates real data and 0 indicates padding (zeros).
        Assumes padding is zero.
        """
        return tf.cast(real_data != 0, tf.float32)

    def masked_loss(self, real_labels, pred, mask):
        """
        Compute the loss, ignoring the padded values (mask applied).
        This multiplies the loss by the mask so that padded values contribute nothing.
        """
        loss = self.loss_fn(real_labels, pred)
        return tf.reduce_mean(loss * mask)  # Apply the mask to the loss

    def mse_loss(self, real_data, generated_data, mask):
        """ Compute the MSE loss between real data and generated data, ignoring the padding. """
        real_data = tf.cast(real_data, tf.float32)  # Ensure both are of type float32
        generated_data = tf.cast(generated_data, tf.float32)
        mse = tf.square(real_data - generated_data)  # MSE loss
        return tf.reduce_mean(mse * mask)  # Apply the mask to ignore padded regions

    def zero_padding_loss(self, generated_data, mask):
        """ Penalize non-zero values in the padding areas. """
        # Apply the mask to only the padded (zero) positions and calculate the L2 loss
        padding_mask = 1 - mask  # Invert the mask to find padding positions (where mask is 0)

        return tf.reduce_mean(tf.square(generated_data * padding_mask))  # Penalize non-zero values in padding regions


    def train_step(self, data):
        real_data, condition = data
        batch_size = tf.shape(real_data)[0]

        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))

        # Create mask for real data
        mask = self.create_mask(real_data)

        # Train Discriminator
        z = tf.random.normal((batch_size, self.latent_dim))
        fake_data = self.generator(tf.concat([tf.cast(z, tf.float32), tf.cast(condition, tf.float32)], axis=1))


        with tf.GradientTape() as tape_D:
            real_pred = self.discriminator(tf.concat([tf.cast(real_data, tf.float32), tf.cast(condition, tf.float32)], axis=1))
            fake_pred = self.discriminator(tf.concat([tf.cast(fake_data, tf.float32), tf.cast(condition, tf.float32)], axis=1))

            # Compute masked loss
            real_loss = self.masked_loss(real_labels, real_pred, mask)
            fake_loss = self.masked_loss(fake_labels, fake_pred, mask)

            d_loss = real_loss + fake_loss


        # Train Generator
        z = tf.random.normal((batch_size, self.latent_dim))
        with tf.GradientTape() as tape_G:
            fake_data = self.generator(tf.concat([tf.cast(z, tf.float32), tf.cast(condition, tf.float32)], axis=1))
            fake_pred = self.discriminator(tf.concat([tf.cast(fake_data, tf.float32), tf.cast(condition, tf.float32)], axis=1))

            # Compute adversarial loss (BCE)
            adv_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(real_labels, fake_pred)
            g_loss = adv_loss

            # Compute generator loss
            g_loss += 1 * self.masked_loss(real_labels, fake_pred, mask)

            # Add zero-padding loss to encourage zero values in padded regions
            z_padding_loss = self.zero_padding_loss(fake_data, mask)
            g_loss += 10 * z_padding_loss  # Add a weight to the zero padding loss

            # Add MSE loss between generated data and real data (ignore padding)
            mse = self.mse_loss(real_data, fake_data, mask)
            g_loss += 5 * mse  # Add a weight to the MSE loss (adjust as needed)

        grads_G = tape_G.gradient(g_loss, self.generator.trainable_variables)
        self.G_trace_optimizer.apply_gradients(zip(grads_G, self.generator.trainable_variables))


        return {"d_loss": d_loss, "g_loss": g_loss}


class CycleGenerator(keras.Sequential):
    def __init__(self, input_dim, output_dim, layers_config, activation):
        super().__init__()

        # Define all layers in the constructor
        self.add(keras.Input(shape=(input_dim,)))  # Use keras.Input instead of layers.InputLayer

        for units in layers_config:
            self.add(layers.Dense(units, activation=activation))

        self.add(layers.Dense(output_dim))  # Output layer without activation for regression

        # No need to override call method when using Sequential


class CycleDiscriminator(keras.Sequential):
    def __init__(self, input_dim, layers_config, activation):
        super().__init__()

        # Define all layers in the constructor
        self.add(keras.Input(shape=(input_dim,)))  # Use keras.Input instead of layers.InputLayer

        for units in layers_config:
            self.add(layers.Dense(units, activation=activation))

        self.add(layers.Dense(1, activation='sigmoid'))



class CycleGAN(keras.Model):
    def __init__(self, model_name, latent_dim, profile_dim, trace_dim, gen_layers, disc_layers, activation="relu", lr=0.0002, beta_1=0.5):
        super(CycleGAN, self).__init__()

        self.model_name = model_name
        
        # Generators
        self.G_trace = CycleGenerator(profile_dim, trace_dim, gen_layers, activation)  # Profile → Trace
        self.G_profile = CycleGenerator(trace_dim, profile_dim, gen_layers, activation)  # Trace → Profile

        # Discriminators
        self.D_trace = CycleDiscriminator(trace_dim, disc_layers, activation)  # Validates generated traces
        self.D_profile = CycleDiscriminator(profile_dim, disc_layers, activation)  # Validates generated profiles

        self.lr = lr
        self.beta_1 = beta_1
    
    def save_models(self):
        """Save generator and discriminator models."""
        os.makedirs(MODS_FOLDER, exist_ok=True)
        self.G_trace.save(os.path.join(MODS_FOLDER, f"{self.model_name}_generator_trace.h5"))
        self.G_profile.save(os.path.join(MODS_FOLDER, f"{self.model_name}_generator_profile.h5"))
        self.D_trace.save(os.path.join(MODS_FOLDER, f"{self.model_name}_discriminator_trace.h5"))
        self.D_profile.save(os.path.join(MODS_FOLDER, f"{self.model_name}_discriminator_profile.h5"))

    
    def compile(self):
        super(CycleGAN, self).compile()
        self.d_optimizer = keras.optimizers.Adam(self.lr, self.beta_1)
        self.G_trace_optimizer = keras.optimizers.Adam(self.lr, self.beta_1)
        self.loss_fn = keras.losses.BinaryCrossentropy()

    def plot_losses(self):
        os.makedirs(MODS_FOLDER + f"{self.model_name}/", exist_ok=True)

        """Plot training loss history."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.history.history["d_loss"], label="Discriminator Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Losses")
        plt.savefig(MODS_FOLDER + f"{self.model_name}/loss_d.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(self.history.history["g_loss"], label="Generator Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Losses")
        plt.savefig(MODS_FOLDER + f"{self.model_name}/loss_g.png")
        plt.close()

    def evaluate_performance(self, real_profiles, real_traces, save_folder="evaluation_results"):
        """
        Evaluate model performance using test data.
        Calculate losses for both the generator and the discriminators in both directions.
        Write the results to a file.
        """

        # Ensure the data is in TensorFlow tensor format
        if not isinstance(real_profiles, tf.Tensor):
            real_profiles = tf.convert_to_tensor(real_profiles, dtype=tf.float32)
        if not isinstance(real_traces, tf.Tensor):
            real_traces = tf.convert_to_tensor(real_traces, dtype=tf.float32)

        batch_size = tf.shape(real_profiles)[0]

        # Evaluate G (Profile → Trace) and F (Trace → Profile)
        fake_traces = self.G_trace(real_profiles, training=False)  # G(Profile → Trace)
        cycle_profiles = self.G_profile(fake_traces, training=False)  # F(G(Profile)) → Profile

        fake_profiles = self.G_profile(real_traces, training=False)  # F(Trace → Profile)
        cycle_traces = self.G_trace(fake_profiles, training=False)  # G(F(Trace)) → Trace

        # Calculate Cycle Consistency Loss (L1 loss)
        cycle_loss_P = self.cycle_loss(real_profiles, cycle_profiles)  # Cycle loss for Profile
        cycle_loss_T = self.cycle_loss(real_traces, cycle_traces)  # Cycle loss for Trace

        # Calculate Generator Adversarial Losses
        adv_loss_G = self.loss_fn(tf.ones((batch_size, 1)), self.D_trace(fake_traces))  # G should fool D_trace
        adv_loss_F = self.loss_fn(tf.ones((batch_size, 1)), self.D_profile(fake_profiles))  # F should fool D_profile

        # Combine the generator losses
        total_gen_loss = adv_loss_G + adv_loss_F + 10 * (cycle_loss_P + cycle_loss_T)

        # Calculate Discriminator Losses (real vs fake)
        real_pred_trace = self.D_trace(real_traces, training=False)
        fake_pred_trace = self.D_trace(fake_traces, training=False)

        real_pred_profile = self.D_profile(real_profiles, training=False)
        fake_pred_profile = self.D_profile(fake_profiles, training=False)

        d_loss_trace = self.loss_fn(tf.ones((batch_size, 1)), real_pred_trace) + self.loss_fn(tf.zeros((batch_size, 1)), fake_pred_trace)
        d_loss_profile = self.loss_fn(tf.ones((batch_size, 1)), real_pred_profile) + self.loss_fn(tf.zeros((batch_size, 1)), fake_pred_profile)

        total_disc_loss = d_loss_trace + d_loss_profile

        # Additional MSE for generator performance
        mse_loss = self.mse_loss(real_profiles, fake_profiles, tf.ones_like(real_profiles)) + self.mse_loss(real_traces, fake_traces, tf.ones_like(real_traces))

        # Logging results to console
        print(f"Generator Loss: {total_gen_loss.numpy():.4f}")
        print(f"Discriminator Loss: {total_disc_loss.numpy():.4f}")
        print(f"MSE Loss (Profile → Trace and Trace → Profile): {mse_loss.numpy():.4f}")

        # Save results to file

        with open(MODS_FOLDER + f"{self.model_name}/test_loss.txt", "w", encoding="utf-8") as file:
            file.write(f"Generator Loss: {total_gen_loss.numpy():.4f}\n")
            file.write(f"Discriminator Loss: {total_disc_loss.numpy():.4f}\n")
            file.write(f"Cycle Consistency Loss (Profile → Trace): {cycle_loss_P.numpy():.4f}\n")
            file.write(f"Cycle Consistency Loss (Trace → Profile): {cycle_loss_T.numpy():.4f}\n")
            file.write(f"MSE Loss (Profile → Trace and Trace → Profile): {mse_loss.numpy():.4f}\n")

        return {
            "gen_loss": total_gen_loss,
            "disc_loss": total_disc_loss,
            "cycle_loss": cycle_loss_P + cycle_loss_T,
            "mse_loss": mse_loss
        }

    def mse_loss(self, real_data, generated_data, mask):
        """ Compute the MSE loss between real data and generated data, ignoring the padding. """
        real_data = tf.cast(real_data, tf.float32)  # Ensure both are of type float32
        generated_data = tf.cast(generated_data, tf.float32)
        mse = tf.square(real_data - generated_data)  # MSE loss
        return tf.reduce_mean(mse * mask)  # Apply the mask to ignore padded regions

    def cycle_loss(self, real, reconstructed):
        """Cycle consistency loss (L1 loss)."""
        return tf.reduce_mean(tf.abs(tf.cast(real, tf.float32) - tf.cast(reconstructed, tf.float32)))

    def masked_cycle_loss(self, real_labels, pred, mask):
        """
        Compute the loss, ignoring the padded values (mask applied).
        This multiplies the loss by the mask so that padded values contribute nothing.
        """
        loss = self.cycle_loss(real_labels, pred)
        return tf.reduce_mean(loss * mask)  # Apply the mask to the loss

    def zero_padding_loss(self, generated_data, mask):
        """ Penalize non-zero values in the padding areas. """
        # Apply the mask to only the padded (zero) positions and calculate the L2 loss
        padding_mask = 1 - mask  # Invert the mask to find padding positions (where mask is 0)

        return tf.reduce_mean(tf.square(generated_data * padding_mask))  # Penalize non-zero values in padding regions


    def create_mask(self, real_data):
        """
        Create a binary mask where 1 indicates real data and 0 indicates padding (zeros).
        Assumes padding is zero.
        """
        return tf.cast(real_data != 0, tf.float32)

    def train_step(self, data):
        real_profile, real_trace = data
        batch_size = tf.shape(real_profile)[0]

        mask_profile = self.create_mask(real_profile)
        mask_trace = self.create_mask(real_trace)

        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))

        # === Train Generators ===
        with tf.GradientTape() as tape_G:
            fake_trace = self.G_trace(real_profile)  # G(Profile) → Trace
            cycle_profile = self.G_profile(fake_trace)  # F(G(Profile)) → Profile

            fake_profile = self.G_profile(real_trace)  # F(Trace) → Profile
            cycle_trace = self.G_trace(fake_profile)  # G(F(Trace)) → Trace

            adv_loss_G = self.loss_fn(real_labels, self.D_trace(fake_trace))  # Fool D_trace
            adv_loss_F = self.loss_fn(real_labels, self.D_profile(fake_profile))  # Fool D_profile

            cycle_loss_P = self.masked_cycle_loss(real_profile, cycle_profile, mask_profile)  # Cycle loss for Profile
            cycle_loss_T = self.masked_cycle_loss(real_trace, cycle_trace, mask_trace)  # Cycle loss for Trace

            zero_pad_loss_P = self.zero_padding_loss(fake_profile, mask_profile)
            zero_pad_loss_T = self.zero_padding_loss(fake_trace, mask_trace)

            total_gen_loss = adv_loss_G + adv_loss_F + 10 * (cycle_loss_P + cycle_loss_T) + 5 * (zero_pad_loss_T + zero_pad_loss_P) # Weighted loss

        grads_G = tape_G.gradient(total_gen_loss, self.G_trace.trainable_variables + self.G_profile.trainable_variables)
        self.G_trace_optimizer.apply_gradients(zip(grads_G, self.G_trace.trainable_variables + self.G_profile.trainable_variables))

        # === Train Discriminators ===
        with tf.GradientTape() as tape_D:
            real_pred_trace = self.D_trace(real_trace)
            fake_pred_trace = self.D_trace(fake_trace)

            real_pred_profile = self.D_profile(real_profile)
            fake_pred_profile = self.D_profile(fake_profile)

            d_loss_trace = self.loss_fn(real_labels, real_pred_trace) + self.loss_fn(fake_labels, fake_pred_trace)
            d_loss_profile = self.loss_fn(real_labels, real_pred_profile) + self.loss_fn(fake_labels, fake_pred_profile)

            total_disc_loss = d_loss_trace + d_loss_profile

        grads_D = tape_D.gradient(total_disc_loss, self.D_trace.trainable_variables + self.D_profile.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads_D, self.D_trace.trainable_variables + self.D_profile.trainable_variables))

        return {"d_loss": total_disc_loss, "g_loss": total_gen_loss}


class CycleGANv2(keras.Model):
    def __init__(self, model_name, latent_dim, profile_dim, trace_dim, gen_layers, disc_layers, activation="relu", lr=0.0002, beta_1=0.5, LAMBDA=10):
        super(CycleGANv2, self).__init__()

        self.model_name = model_name

        # Generators
        self.G_trace = CycleGenerator(profile_dim, trace_dim, gen_layers, activation)  # Profile → Trace
        self.G_profile = CycleGenerator(trace_dim, profile_dim, gen_layers, activation)  # Trace → Profile

        # Discriminators
        self.D_trace = CycleDiscriminator(trace_dim, disc_layers, activation)  # Validates generated traces
        self.D_profile = CycleDiscriminator(profile_dim, disc_layers, activation)  # Validates generated profiles

        self.lr = lr
        self.beta_1 = beta_1
        self.LAMBDA = LAMBDA

    def save_models(self):
        """Save generator and discriminator models."""
        os.makedirs(MODS_FOLDER, exist_ok=True)
        self.G_trace.save(os.path.join(MODS_FOLDER, f"{self.model_name}_generator_trace.h5"))
        self.G_profile.save(os.path.join(MODS_FOLDER, f"{self.model_name}_generator_profile.h5"))
        self.D_trace.save(os.path.join(MODS_FOLDER, f"{self.model_name}_discriminator_trace.h5"))
        self.D_profile.save(os.path.join(MODS_FOLDER, f"{self.model_name}_discriminator_profile.h5"))

    def compile(self):
        super(CycleGANv2, self).compile()
        self.d_trace_optimizer = keras.optimizers.Adam(self.lr, self.beta_1)
        self.d_profile_optimizer = keras.optimizers.Adam(self.lr, self.beta_1)
        self.g_trace_optimizer = keras.optimizers.Adam(self.lr, self.beta_1)
        self.g_profile_optimizer = keras.optimizers.Adam(self.lr, self.beta_1)
        self.loss_fn = keras.losses.BinaryCrossentropy()

    def plot_losses(self):
        os.makedirs(MODS_FOLDER + f"{self.model_name}/", exist_ok=True)

        """Plot training loss history."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.history.history["d_loss"], label="Discriminator Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Losses")
        plt.savefig(MODS_FOLDER + f"{self.model_name}/loss_d.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(self.history.history["g_loss"], label="Generator Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Losses")
        plt.savefig(MODS_FOLDER + f"{self.model_name}/loss_g.png")
        plt.close()

    def evaluate_performance(self, real_profiles, real_traces, save_folder="evaluation_results"):
        """
        Evaluate model performance using test data.
        Calculate losses for both the generator and the discriminators in both directions.
        Write the results to a file.
        """

        # Ensure the data is in TensorFlow tensor format
        if not isinstance(real_profiles, tf.Tensor):
            real_profiles = tf.convert_to_tensor(real_profiles, dtype=tf.float32)
        if not isinstance(real_traces, tf.Tensor):
            real_traces = tf.convert_to_tensor(real_traces, dtype=tf.float32)

        batch_size = tf.shape(real_profiles)[0]

        # Evaluate G (Profile → Trace) and F (Trace → Profile)
        fake_traces = self.G_trace(real_profiles, training=False)  # G(Profile → Trace)
        cycle_profiles = self.G_profile(fake_traces, training=False)  # F(G(Profile)) → Profile

        fake_profiles = self.G_profile(real_traces, training=False)  # F(Trace → Profile)
        cycle_traces = self.G_trace(fake_profiles, training=False)  # G(F(Trace)) → Trace

        # Calculate Cycle Consistency Loss (L1 loss)
        cycle_loss_P = self.cycle_loss(real_profiles, cycle_profiles)  # Cycle loss for Profile
        cycle_loss_T = self.cycle_loss(real_traces, cycle_traces)  # Cycle loss for Trace

        # Calculate Generator Adversarial Losses
        adv_loss_G = self.loss_fn(tf.ones((batch_size, 1)), self.D_trace(fake_traces))  # G should fool D_trace
        adv_loss_F = self.loss_fn(tf.ones((batch_size, 1)), self.D_profile(fake_profiles))  # F should fool D_profile

        # Combine the generator losses
        total_gen_loss = adv_loss_G + adv_loss_F + 10 * (cycle_loss_P + cycle_loss_T)

        # Calculate Discriminator Losses (real vs fake)
        real_pred_trace = self.D_trace(real_traces, training=False)
        fake_pred_trace = self.D_trace(fake_traces, training=False)

        real_pred_profile = self.D_profile(real_profiles, training=False)
        fake_pred_profile = self.D_profile(fake_profiles, training=False)

        d_loss_trace = self.loss_fn(tf.ones((batch_size, 1)), real_pred_trace) + self.loss_fn(tf.zeros((batch_size, 1)), fake_pred_trace)
        d_loss_profile = self.loss_fn(tf.ones((batch_size, 1)), real_pred_profile) + self.loss_fn(tf.zeros((batch_size, 1)), fake_pred_profile)

        total_disc_loss = d_loss_trace + d_loss_profile

        # Additional MSE for generator performance
        mse_loss = self.mse_loss(real_profiles, fake_profiles, tf.ones_like(real_profiles)) + self.mse_loss(real_traces, fake_traces, tf.ones_like(real_traces))

        # Logging results to console
        print(f"Generator Loss: {total_gen_loss.numpy():.4f}")
        print(f"Discriminator Loss: {total_disc_loss.numpy():.4f}")
        print(f"MSE Loss (Profile → Trace and Trace → Profile): {mse_loss.numpy():.4f}")

        # Save results to file

        with open(MODS_FOLDER + f"{self.model_name}/test_loss.txt", "w", encoding="utf-8") as file:
            file.write(f"Generator Loss: {total_gen_loss.numpy():.4f}\n")
            file.write(f"Discriminator Loss: {total_disc_loss.numpy():.4f}\n")
            file.write(f"Cycle Consistency Loss (Profile → Trace): {cycle_loss_P.numpy():.4f}\n")
            file.write(f"Cycle Consistency Loss (Trace → Profile): {cycle_loss_T.numpy():.4f}\n")
            file.write(f"MSE Loss (Profile → Trace and Trace → Profile): {mse_loss.numpy():.4f}\n")

        return {
            "gen_loss": total_gen_loss,
            "disc_loss": total_disc_loss,
            "cycle_loss": cycle_loss_P + cycle_loss_T,
            "mse_loss": mse_loss
        }

    def mse_loss(self, real_data, generated_data, mask):
        """ Compute the MSE loss between real data and generated data, ignoring the padding. """
        real_data = tf.cast(real_data, tf.float32)  # Ensure both are of type float32
        generated_data = tf.cast(generated_data, tf.float32)
        mse = tf.square(real_data - generated_data)  # MSE loss
        return tf.reduce_mean(mse * mask)  # Apply the mask to ignore padded regions

    def cycle_loss(self, real, reconstructed):
        """Cycle consistency loss (L1 loss)."""
        return tf.reduce_mean(tf.abs(tf.cast(real, tf.float32) - tf.cast(reconstructed, tf.float32)))

    def masked_cycle_loss(self, real_labels, pred, mask):
        """
        Compute the loss, ignoring the padded values (mask applied).
        This multiplies the loss by the mask so that padded values contribute nothing.
        """
        loss = self.cycle_loss(real_labels, pred)
        return tf.reduce_mean(loss * mask)  # Apply the mask to the loss

    def zero_padding_loss(self, generated_data, mask):
        """ Penalize non-zero values in the padding areas. """
        # Apply the mask to only the padded (zero) positions and calculate the L2 loss
        padding_mask = 1 - mask  # Invert the mask to find padding positions (where mask is 0)

        return tf.reduce_mean(tf.square(generated_data * padding_mask))  # Penalize non-zero values in padding regions

    def create_mask(self, real_data):
        """
        Create a binary mask where 1 indicates real data and 0 indicates padding (zeros).
        Assumes padding is zero.
        """
        return tf.cast(real_data != 0, tf.float32)

    def calc_cycle_loss(self, real_sample, cycled_sample):
        loss = tf.reduce_mean(tf.abs(tf.cast(real_sample, tf.float32) - tf.cast(cycled_sample, tf.float32)))

        return self.LAMBDA * loss

    def identity_loss(self, real_sample, same_sample):
        loss = tf.reduce_mean(tf.abs(tf.cast(real_sample, tf.float32) - tf.cast(same_sample, tf.float32)))
        return self.LAMBDA * 0.5 * loss

    def generator_loss(self, generated):
        return self.loss_fn(tf.ones_like(generated), generated)

    def discriminator_loss(self, real, generated):
        real_loss = self.loss_fn(tf.ones_like(real), real)

        generated_loss = self.loss_fn(tf.zeros_like(generated), generated)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss * 0.5

    def train_step(self, data):
        real_profile, real_trace = data
        real_profile = tf.cast(real_profile, tf.float32)
        real_trace = tf.cast(real_trace, tf.float32)

        batch_size = tf.shape(real_profile)[0]

        mask_profile = self.create_mask(real_profile)
        mask_trace = self.create_mask(real_trace)

        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))

        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.

            fake_y = self.G_trace(real_profile, training=True)
            cycled_x = self.G_profile(fake_y, training=True)

            fake_x = self.G_profile(real_trace, training=True)
            cycled_y = self.G_trace(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = self.G_profile(real_profile, training=True)
            same_y = self.G_trace(real_trace, training=True)

            disc_real_x = self.D_profile(real_profile, training=True)
            disc_real_y = self.D_trace(real_trace, training=True)

            disc_fake_x = self.D_profile(fake_x, training=True)
            disc_fake_y = self.D_trace(fake_y, training=True)

            # calculate the loss
            gen_g_loss = self.generator_loss(disc_fake_y)
            gen_f_loss = self.generator_loss(disc_fake_x)

            total_cycle_loss = self.calc_cycle_loss(real_profile, cycled_x) + self.calc_cycle_loss(real_trace, cycled_y)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_trace, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_profile, same_x)

            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)

        # Calculate the gradients for generator and discriminator
        g_trace_gradients = tape.gradient(total_gen_g_loss, self.G_trace.trainable_variables)
        g_profile_gradients = tape.gradient(total_gen_f_loss, self.G_profile.trainable_variables)

        d_profile_gradients = tape.gradient(disc_x_loss, self.D_profile.trainable_variables)
        d_trace_gradients = tape.gradient(disc_y_loss, self.D_trace.trainable_variables)

        # Apply the gradients to the optimizer
        self.g_trace_optimizer.apply_gradients(zip(g_trace_gradients, self.G_trace.trainable_variables))

        self.g_profile_optimizer.apply_gradients(zip(g_profile_gradients, self.G_profile.trainable_variables))

        self.d_profile_optimizer.apply_gradients(zip(d_profile_gradients, self.D_profile.trainable_variables))

        self.d_trace_optimizer.apply_gradients(zip(d_trace_gradients, self.D_trace.trainable_variables))


# class ConditionalVAE(Model):
#     def __init__(self, input_dim, conditional_dim, random_dim, intermediate_dims=[256, 128, 64]):
#         super(ConditionalVAE, self).__init__()
#
#         self.input_dim = input_dim
#         self.conditional_dim = conditional_dim
#         self.random_dim = random_dim
#         self.total_latent_dim = conditional_dim + random_dim
#
#         # Much smaller initial weights for stability
#         self.reconstruction_weight = 1.0
#         self.kl_weight = 0.0001  # Start extremely small
#         self.padding_weight = 0.1
#
#         # Encoder with smaller initial layers
#         self.encoder_layers = []
#
#         for dim in intermediate_dims:
#             self.encoder_layers.extend([
#                 layers.Dense(
#                     dim,
#                     kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.01),
#                     bias_initializer='zeros'
#                 ),
#                 layers.LeakyReLU(0.01),  # Smaller slope
#                 layers.BatchNormalization(momentum=0.99),
#                 layers.Dropout(0.1)
#             ])
#
#         self.encoder_network = tf.keras.Sequential(self.encoder_layers)
#
#         # Latent space layers with very conservative initialization
#         self.random_mu = layers.Dense(
#             random_dim,
#             kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.001),
#             bias_initializer='zeros'
#         )
#         self.random_log_var = layers.Dense(
#             random_dim,
#             kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.001),
#             bias_initializer=tf.keras.initializers.Constant(-5.0)  # Start with very small variance
#         )
#
#         # Decoder
#         decoder_dims = list(reversed(intermediate_dims))
#         self.decoder_layers = []
#
#         current_dim = self.total_latent_dim
#         for dim in decoder_dims:
#             self.decoder_layers.extend([
#                 layers.Dense(
#                     dim,
#                     kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.01),
#                     bias_initializer='zeros'
#                 ),
#                 layers.LeakyReLU(0.01),
#                 layers.BatchNormalization(momentum=0.99),
#                 layers.Dropout(0.1)
#             ])
#
#         self.decoder_layers.append(
#             layers.Dense(
#                 input_dim,
#                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.01),
#                 bias_initializer='zeros'
#             )
#         )
#
#         self.decoder_network = tf.keras.Sequential(self.decoder_layers)
#
#     def call(self, inputs):
#         """Forward pass of the model"""
#         # Encode
#         (data, conditional) = inputs
#         encoded = self.encoder_network(data)
#         random_mu = self.random_mu(encoded)
#         random_log_var = self.random_log_var(encoded)
#
#         # Clip values
#         random_mu = tf.clip_by_value(random_mu, -3.0, 3.0)
#         random_log_var = tf.clip_by_value(random_log_var, -5.0, 5.0)
#
#         # Sample from latent space
#         random_z = self.reparameterize(random_mu, random_log_var)
#
#         # Concatenate random and conditional parts
#         z = tf.concat([random_z, conditional], axis=1)
#
#         # Decode
#         reconstructed = self.decoder_network(z)
#
#         return reconstructed, random_mu, random_log_var, self.create_padding_mask(data)
#
#     def create_padding_mask(self, x):
#         return tf.cast(tf.not_equal(x, 0), tf.float32)
#
#     def encode(self, x, mask):
#         x = tf.cast(x, tf.float32)
#         x = x * mask
#
#         # Monitor for NaNs
#         tf.debugging.check_numerics(x, 'NaN found in encode input')
#
#         x = self.encoder_network(x)
#         tf.debugging.check_numerics(x, 'NaN found after encoder network')
#
#         random_mu = tf.clip_by_value(self.random_mu(x), -3.0, 3.0)
#         random_log_var = tf.clip_by_value(self.random_log_var(x), -5.0, 2.0)
#
#         return random_mu, random_log_var
#
#     def reparameterize(self, mu, log_var):
#         eps = tf.random.normal(shape=tf.shape(mu), mean=0.0, stddev=0.1)  # Reduced noise
#         return mu + tf.exp(0.5 * log_var) * eps
#
#     def decode(self, random_z, conditional_input):
#         random_z = tf.cast(random_z, tf.float32)
#         conditional_input = tf.cast(conditional_input, tf.float32)
#
#         random_z = tf.clip_by_value(random_z, -3.0, 3.0)
#         conditional_input = tf.clip_by_value(conditional_input, -3.0, 3.0)
#
#         z = tf.concat([random_z, conditional_input], axis=1)
#         decoded = self.decoder_network(z)
#
#         # Clip output to match input range
#         decoded = tf.clip_by_value(decoded, -10.0, 10.0)
#         return decoded
#
#     def compute_reconstruction_loss(self, x, reconstructed, mask):
#         """Compute reconstruction loss with careful handling of padded values"""
#         # Ensure no NaN inputs
#         tf.debugging.check_numerics(x, 'NaN found in x')
#         tf.debugging.check_numerics(reconstructed, 'NaN found in reconstructed')
#
#         # MSE for non-padded regions
#         squared_diff = tf.square(tf.cast(x, tf.float32) - reconstructed)
#         masked_squared_diff = squared_diff * mask
#
#         # Very small L1 loss for padded regions
#         l1_padding = 0.01 * tf.abs(reconstructed) * (1 - mask)
#
#         # Normalize by sequence lengths with safe division
#         seq_lengths = tf.reduce_sum(mask, axis=-1, keepdims=True)
#         safe_seq_lengths = tf.maximum(seq_lengths, 1.0)  # Avoid division by zero
#
#         reconstruction_loss = tf.reduce_sum(masked_squared_diff, axis=-1) / safe_seq_lengths
#         padding_loss = tf.reduce_sum(l1_padding, axis=-1) / tf.cast(tf.shape(x)[1], tf.float32)
#
#         padding_mask = 1 - mask  # Invert the mask to find padding positions (where mask is 0)
#         padding_loss = tf.square(reconstructed * padding_mask)
#
#         return tf.reduce_mean(reconstruction_loss), tf.reduce_mean(padding_loss)
#
#     def compute_kl_loss(self, mu, log_var):
#         """Compute KL divergence loss with numerical stability"""
#         # Clip values for stability
#         mu = tf.clip_by_value(mu, -3.0, 3.0)
#         log_var = tf.clip_by_value(log_var, -5.0, 2.0)
#
#         kl_loss = 0.5 * tf.reduce_mean(
#             tf.exp(log_var) + tf.square(mu) - 1.0 - log_var
#         )
#
#         return tf.maximum(kl_loss, 0.0)  # Ensure non-negative
#
#     def train_step(self, data):
#         x, conditional_input = data
#
#         with tf.GradientTape() as tape:
#             reconstructed, random_mu, random_log_var, mask = self([x, tf.cast(conditional_input, tf.float32)])
#
#             reconstruction_loss, padding_loss = self.compute_reconstruction_loss(x, reconstructed, mask)
#             kl_loss = self.compute_kl_loss(random_mu, random_log_var)
#
#             total_loss = (
#                     self.reconstruction_weight * reconstruction_loss +
#                     self.padding_weight * padding_loss +
#                     self.kl_weight * kl_loss
#             )
#
#         # Extra careful gradient clipping
#         gradients = tape.gradient(total_loss, self.trainable_variables)
#         gradients = [tf.clip_by_value(grad, -1.0, 1.0) if grad is not None else grad for grad in gradients]
#         self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
#
#         return {
#             "loss": total_loss,
#             "reconstruction_loss": reconstruction_loss,
#             "padding_loss": padding_loss,
#             "kl_loss": kl_loss
#         }
#
#     def generate(self, conditional_input, n_samples=1, temperature=1.0):
#         conditional_input = tf.cast(conditional_input, tf.float32)
#
#         random_z = tf.random.normal(
#             (conditional_input.shape[0] * n_samples, self.random_dim),
#             mean=0.0,
#             stddev=temperature
#         )
#
#         conditional_repeated = tf.repeat(conditional_input, n_samples, axis=0)
#         return self.decode(random_z, conditional_repeated)


import tensorflow as tf
from tensorflow.keras import layers, Model

class ConditionalVAE(Model):
    def __init__(self, model_name, input_dim, conditional_dim, random_dim, layer_config=[256, 128, 64]):
        super(ConditionalVAE, self).__init__()

        self.model_name = model_name

        self.input_dim = input_dim
        self.conditional_dim = conditional_dim
        self.random_dim = random_dim
        self.total_latent_dim = conditional_dim + random_dim

        # Dynamic KL weight adjustment
        self.kl_weight = tf.Variable(0.001, trainable=False)
        self.reconstruction_weight = 5.0
        self.padding_weight = tf.Variable(1.0, trainable=False)

        # Encoder
        self.encoder_layers = []
        for dim in layer_config:
            self.encoder_layers.extend([
                # layers.Dense(dim, activation="relu"),
                layers.Dense(dim, activation=None, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.01)),
                # layers.LeakyReLU(0.01),
                # layers.BatchNormalization(momentum=0.99),
                layers.ELU(),
                # layers.BatchNormalization(momentum=0.9),
                layers.LayerNormalization(),
                layers.Dropout(0.1)
            ])
        self.encoder_network = tf.keras.Sequential(self.encoder_layers)

        # Latent layers
        self.random_mu = layers.Dense(random_dim, kernel_initializer='he_normal')
        self.random_log_var = layers.Dense(random_dim, kernel_initializer='he_normal')

        # Decoder
        self.decoder_layers = []
        decoder_dims = list(reversed(layer_config))
        for dim in decoder_dims:
            self.decoder_layers.extend([
                # layers.Dense(dim, activation="relu"),
                layers.Dense(dim, activation=None, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.01)),
                # layers.LeakyReLU(0.01),
                # layers.BatchNormalization(momentum=0.99),
                layers.ELU(),
                # layers.BatchNormalization(momentum=0.9),
                layers.LayerNormalization(),
                layers.Dropout(0.1)
            ])
        self.decoder_layers.append(layers.Dense(input_dim, activation=None))
        self.decoder_network = tf.keras.Sequential(self.decoder_layers)

    def call(self, inputs):
        (data, conditional) = inputs
        encoded = self.encoder_network(data)
        random_mu = tf.clip_by_value(self.random_mu(encoded), -3.0, 3.0)
        random_log_var = tf.clip_by_value(self.random_log_var(encoded), -3.0, 3.0)
        random_z = self.reparameterize(random_mu, random_log_var)
        z = tf.concat([random_z, conditional], axis=1)
        reconstructed = self.decoder_network(z)  # Removed residual connection
        return reconstructed, random_mu, random_log_var, self.create_padding_mask(data)

    def reparameterize(self, mu, log_var, temperature=1.0):
        stddev = temperature * tf.exp(0.5 * log_var)
        eps = tf.random.normal(shape=tf.shape(mu), mean=0.0, stddev=stddev)
        return mu + eps

    # def reparameterize(self, mu, log_var):
    #     eps = tf.random.normal(shape=tf.shape(mu), mean=0.0, stddev=0.5)
    #     return mu + tf.exp(0.5 * log_var) * eps

    def save_models(self):
        """Save generator and discriminator models."""
        os.makedirs(MODS_FOLDER, exist_ok=True)
        self.encoder_network.save(os.path.join(MODS_FOLDER, f"{self.model_name}_encoder.h5"))
        self.decoder_network.save(os.path.join(MODS_FOLDER, f"{self.model_name}_decoder.h5"))

    def plot_losses(self):
        os.makedirs(MODS_FOLDER + f"{self.model_name}/", exist_ok=True)

        """Plot training loss history."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.history.history["loss"], label="Discriminator Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Losses")
        plt.savefig(MODS_FOLDER + f"{self.model_name}/loss_d.png")
        plt.close()


    def evaluate_performance(self, test_data, test_conditions):
        """Evaluate model on test data and return the loss."""
        """Evaluate model on test data and return the loss."""
        mask = self.create_padding_mask(test_data)

        # Generate latent vector and concatenate with conditions
        random_latent = tf.random.normal((tf.shape(test_data)[0], self.random_dim))
        z = tf.concat([random_latent, test_conditions], axis=1)

        # Generate fake data
        fake_data = self.decoder_network(z)

        # Compute MSE loss
        mse_loss = self.mse_loss(test_data, fake_data, mask).numpy()
        print(f"Test MSE Loss: {mse_loss:.4f}")

        with open(MODS_FOLDER + f"{self.model_name}/test_loss.txt", "w") as file:
            file.write(f"Test MSE Loss: {mse_loss:.4f}\n")

        return mse_loss

    def create_padding_mask(self, x):
        return tf.cast(tf.not_equal(x, 0), tf.float32)
        return tf.cast(tf.abs(x) > threshold, tf.float32)

    # def compute_reconstruction_loss(self, x, reconstructed, mask):
    #     masked_squared_diff = tf.square(tf.cast(x, tf.float32) - reconstructed) * mask
    #     l1_padding = 0.01 * tf.abs(reconstructed) * (1 - mask)
    #     seq_lengths = tf.reduce_sum(mask, axis=-1, keepdims=True)
    #     safe_seq_lengths = tf.maximum(seq_lengths, 1.0)
    #     reconstruction_loss = tf.reduce_sum(masked_squared_diff, axis=-1) / safe_seq_lengths
    #     return tf.reduce_mean(reconstruction_loss), tf.reduce_mean(l1_padding)

    def mse_loss(self, real_data, generated_data, mask, epsilon=1e-8):
        """ Compute the MSE loss between real data and generated data, ignoring the padding. """
        real_data = tf.cast(real_data, tf.float32)
        gene_data = tf.cast(generated_data, tf.float32)

        mse = tf.square(real_data - gene_data)  # MSE loss
        # return tf.reduce_mean(mse * mask)  # Apply the mask to ignore padded regions
        seq_lengths = tf.reduce_sum(mask, axis=-1, keepdims=True)
        return tf.reduce_mean(mse * mask / tf.maximum(seq_lengths, 1.0))  # MODIF: Weighted by sequence length


    def log_cosh_loss(self, real_data, generated_data, mask):
        real_data = tf.cast(real_data, tf.float32)
        generated_data = tf.cast(generated_data, tf.float32)
        return tf.reduce_mean(tf.math.log(tf.cosh(generated_data - real_data)) * mask)

    def zero_padding_loss(self, generated_data, mask):
        """ Penalize non-zero values in the padding areas. """
        # Apply the mask to only the padded (zero) positions and calculate the L2 loss
        padding_mask = 1 - mask  # Invert the mask to find padding positions (where mask is 0)

        return tf.reduce_mean(tf.square(generated_data * padding_mask))  # Penalize non-zero values in padding regions


    def compute_kl_loss(self, mu, log_var):
        kl_loss = 0.5 * tf.reduce_mean(tf.exp(log_var) + tf.square(mu) - 1.0 - log_var)
        return tf.maximum(kl_loss, 0.0)



    def train_step(self, data):
        x, conditional_input = data
        with tf.GradientTape() as tape:
            reconstructed, random_mu, random_log_var, mask = self([x, tf.cast(conditional_input, tf.float32)])

            # reconstruction_loss, padding_loss = self.compute_reconstruction_loss(x, reconstructed, mask)
            reconstruction_loss = self.mse_loss(x, reconstructed, mask)
            # reconstruction_loss = self.log_cosh_loss(x, reconstructed, mask)

            # padding_loss = self.zero_padding_loss(reconstructed, mask)
            padding_loss = tf.reduce_mean(tf.abs(reconstructed) * (1 - mask))

            kl_loss = self.compute_kl_loss(random_mu, random_log_var)

            total_loss = (self.reconstruction_weight * reconstruction_loss + self.padding_weight * padding_loss + self.kl_weight * kl_loss)

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.kl_weight.assign(tf.minimum(self.kl_weight * 1.01, 0.01))  # Gradual KL increase
        # self.kl_weight.assign(tf.sigmoid(self.kl_weight * 100.0) * 0.01)  # Gradual KL increase

        # MODIF: Dynamic padding weight increase over time
        new_padding_weight = tf.minimum(self.padding_weight * 1.01, 50.0)
        self.padding_weight.assign(new_padding_weight)

        return {"loss": total_loss, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss, "padding_loss": padding_loss}

    def generate(self, conditional_input, n_samples=1, temperature=1.0):
        conditional_input = tf.cast(conditional_input, tf.float32)
        random_z = tf.random.normal((conditional_input.shape[0] * n_samples, self.random_dim), mean=0.0, stddev=temperature)
        conditional_repeated = tf.repeat(conditional_input, n_samples, axis=0)
        return self.decode(random_z, conditional_repeated)

    def decode(self, random_z, conditional_input):
        z = tf.concat([random_z, conditional_input], axis=1)  # Removed latent scaling
        decoded = self.decoder_network(z)
        return decoded
