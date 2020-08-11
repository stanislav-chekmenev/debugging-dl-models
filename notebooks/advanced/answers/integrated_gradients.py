# Ex 1

# Plot the beetle with a black baseline
_ = plot_img_attributions(
    model=inception_v1,
    image=img_name_tensors_ex['Black Beetle'],
    baseline=baseline,
    target_class_idx=307,
    m_steps=3500,
    cmap=plt.cm.viridis,
    overlay_alpha=0.3
)

# White baseline for the beetle
_ = plot_img_attributions(
    model=inception_v1,
    image=img_name_tensors_ex['Black Beetle'],
    baseline=tf.ones_like(img_name_tensors_ex['Black Beetle']),
    target_class_idx=307,
    m_steps=3000,
    cmap=plt.cm.viridis,
    overlay_alpha=0.3
)

# Ex 2

# Black baseline for the goldfinch
_ = plot_img_attributions(
    model=inception_v1,
    image=img_name_tensors_ex['Goldfinch'],
    baseline=baseline,
    target_class_idx=12,
    m_steps=500,
    cmap=plt.cm.inferno,
    overlay_alpha=0.5
)

# Uniform baseline
uniform_baseline = tf.random.uniform((224, 224, 3), 0, 1)

_ = plot_img_attributions(
    model=inception_v1,
    image=img_name_tensors_ex['Goldfinch'],
    baseline=uniform_baseline,
    target_class_idx=12,
    m_steps=500,
    cmap=plt.cm.inferno,
    overlay_alpha=0.5
)

# Attributions from 5 uniform baselines averaged
attributions = np.zeros((5, 224, 224, 3))
attribution_masks = np.zeros((5, 224, 224))

for i in range(5):
    baseline = tf.random.uniform((224, 224, 3), 0, 1)
    
    attributions[i] = integrated_gradients(
            model=inception_v1,
            baseline=baseline,
            image=img_name_tensors_ex['Goldfinch'],
            target_class_idx=12,
            m_steps=500
        )
    
    attribution_masks[i] = tf.reduce_sum(tf.math.abs(attributions[i]), axis=-1)
    
attribution_mask = tf.reduce_mean(attribution_masks, axis=0)


# Plot with the average uniform baseline
fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8, 8))

axs[0, 0].set_title('Baseline image')
axs[0, 0].imshow(baseline)
axs[0, 0].axis('off')

axs[0, 1].set_title('Original image')
axs[0, 1].imshow(img_name_tensors_ex['Goldfinch'])
axs[0, 1].axis('off')

axs[1, 0].set_title('Attribution mask')
axs[1, 0].imshow(attribution_mask, cmap=plt.cm.inferno)
axs[1, 0].axis('off')

axs[1, 1].set_title('Overlay')
axs[1, 1].imshow(attribution_mask, cmap=plt.cm.inferno)
axs[1, 1].imshow(img_name_tensors_ex['Goldfinch'], alpha=0.5)
axs[1, 1].axis('off')

plt.tight_layout()