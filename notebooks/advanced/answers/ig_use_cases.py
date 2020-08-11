# Ex 1

# Transform the image
rotate90_labrador_retriever_img = tf.image.rot90(img_name_tensors['Yellow Labrador Retriever'])
upsidedown_labrador_retriever_img = tf.image.flip_up_down(img_name_tensors['Yellow Labrador Retriever'])
zoom_labrador_retriever_img = tf.keras.preprocessing.image.random_zoom(x=img_name_tensors['Yellow Labrador Retriever'], 
                                                                       zoom_range=(0.45, 0.45))
# Plot predictions
yellow_labrador_names = [
    'Yellow Labrador Retriever (original)',
    'Yellow Labrador Retriever (rotated 90 degrees)',
    'Yellow Labrador Retriever (flipped upsidedown)',
    'Yellow Labrador Retriever (zoomed in)']
yellow_labrador_tensors = [
    img_name_tensors['Yellow Labrador Retriever'],
    rotate90_labrador_retriever_img, 
    upsidedown_labrador_retriever_img,
    zoom_labrador_retriever_img
]

for name, tensor in zip(yellow_labrador_names, yellow_labrador_tensors):
    plot_img_predictions(inception_v1, tensor, name, imagenet_labels)
    
# Compute IG
labrador_retriever_attributions = integrated_gradients(
    model=inception_v1,
    baseline=name_baseline_tensors['Baseline Image: Black'],
    image=img_name_tensors['Yellow Labrador Retriever'],
    target_class_idx=tf.constant(209),
    m_steps=tf.constant(1250),
)

zoom_labrador_retriever_attributions = integrated_gradients(
    model=inception_v1,
    baseline=name_baseline_tensors['Baseline Image: Black'],
    image=zoom_labrador_retriever_img,
    target_class_idx=tf.constant(209),
    m_steps=tf.constant(1250),
)

# Plot 
fig, axs = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(16, 12))

axs[0,0].set_title('IG Attributions - Incorrect Prediction: Saluki')
axs[0,0].imshow(tf.reduce_sum(tf.abs(upsidedown_labrador_retriever_attributions), axis=-1), cmap=plt.cm.inferno)
axs[0,0].axis('off')

axs[0,1].set_title('IG Attributions - Correct Prediction: Labrador Retriever')
axs[0,1].imshow(tf.reduce_sum(tf.abs(labrador_retriever_attributions), axis=-1), cmap=None)
axs[0,1].axis('off')

axs[0,2].set_title('IG Attributions - both predictions overlayed')
axs[0,2].imshow(tf.reduce_sum(tf.abs(upsidedown_labrador_retriever_attributions), axis=-1), cmap=plt.cm.inferno, alpha=0.99)
axs[0,2].imshow(tf.reduce_sum(tf.abs(labrador_retriever_attributions), axis=-1), cmap=None, alpha=0.5)
axs[0,2].axis('off')

plt.tight_layout();