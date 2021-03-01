import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def load_imagenet_labels(file_path):
    labels_file = tf.keras.utils.get_file('ImageNetLabels.txt', file_path)
    with open(labels_file, "r") as reader:
        f = reader.read()
        labels = f.splitlines()
        imagenet_label_array = np.array(labels)

    return imagenet_label_array


def read_image(file_name):
    image = tf.io.read_file(file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_pad(image, target_height=224, target_width=224)
    return image


def top_k_predictions(img, model, labels, k=3):
    # Add batch dim for the model to be able to predict
    image_batch = tf.expand_dims(img, axis=0)
    predictions = model(image_batch)
    probs = tf.nn.softmax(predictions, axis=-1)
    top_probs, top_idxs = tf.math.top_k(input=probs, k=k)
    top_labels = labels[tuple(top_idxs)]
    return top_labels, top_probs[0]


def plot_img_predictions(model, img_tensor, name, labels):
    plt.imshow(img_tensor)
    plt.title(name, fontweight='bold')
    plt.axis('off')
    plt.show()

    pred_label, pred_prob = top_k_predictions(img_tensor, model, labels)
    for label, prob in zip(pred_label, pred_prob):
        print(f'{label}: {prob:0.1%}')
    return None


def interpolate_images(baseline,
                       image,
                       alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x +  alphas_x * delta
    return images


def compute_gradients(model, images, target_class_idx):
    with tf.GradientTape() as tape:
        # We use tape.watch here to tell Tensorflow GradientTape to trace all operations with the tensor 'images'
        # When we train a model this step is not needed, since all trainable variables are "watched" automatically
        tape.watch(images)
        logits = model(images)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    return tape.gradient(probs, images)


def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients



def integrated_gradients(model, baseline, image, target_class_idx, m_steps=300, batch_size=32):
    
    # 1. Generate alphas
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps)

    # Accumulate gradients across batches
    integrated_gradients = 0.0

    # Batch alpha images
    ds = tf.data.Dataset.from_tensor_slices(alphas).batch(batch_size)

    for batch in ds:

        # 2. Generate interpolated images
        batch_interpolated_inputs = interpolate_images(
            baseline=baseline,
            image=image,
            alphas=batch
        )

        # 3. Compute gradients between model outputs and interpolated inputs
        batch_gradients = compute_gradients(
            model=model,
            images=batch_interpolated_inputs,
            target_class_idx=target_class_idx
        )

        # 4. Average integral approximation. Summing integrated gradients across batches.
        integrated_gradients += integral_approximation(gradients=batch_gradients)

    # 5. Scale integrated gradients with respect to input
    scaled_integrated_gradients = (image - baseline) * integrated_gradients
        
    return scaled_integrated_gradients

def plot_img_attributions(model, baseline, image, target_class_idx, m_steps=tf.constant(50), cmap=None, overlay_alpha=0.4):

    attributions = integrated_gradients(
        model=model,
        baseline=baseline,
        image=image,
        target_class_idx=target_class_idx,
        m_steps=m_steps
    )

    # Sum of the attributions across color channels for visualization.
    # The attribution mask is a grayscale image with height and width
    # equal to the original image.
    attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)

    fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8, 8))

    axs[0, 0].set_title('Baseline image')
    axs[0, 0].imshow(baseline)
    axs[0, 0].axis('off')

    axs[0, 1].set_title('Original image')
    axs[0, 1].imshow(image)
    axs[0, 1].axis('off')

    axs[1, 0].set_title('Attribution mask')
    axs[1, 0].imshow(attribution_mask, cmap=cmap)
    axs[1, 0].axis('off')

    axs[1, 1].set_title('Overlay')
    axs[1, 1].imshow(attribution_mask, cmap=cmap)
    axs[1, 1].imshow(image, alpha=overlay_alpha)
    axs[1, 1].axis('off')

    plt.tight_layout()
    return fig