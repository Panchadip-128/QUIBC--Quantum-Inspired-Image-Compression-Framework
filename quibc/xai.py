"""
quibc/xai.py
Explainability (XAI) tools for QUIBC:
  - Grad-CAM  (decoder attention maps)
  - Integrated Gradients  (pixel-level attribution)
  - LRP-style relevance propagation
  - Latent space visualisation (PCA, t-SNE, UMAP)
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .model import QUIBCModel


# ── Grad-CAM ────────────────────────────────────────────────────────────────────

def grad_cam(
    model: QUIBCModel,
    image: tf.Tensor,
    layer_name: str = "dec_trans2",
) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap from a decoder layer.

    Args:
        model:      Trained QUIBCModel.
        image:      Float32 tensor (1, H, W, 3) in [0, 1].
        layer_name: Name of the decoder Conv layer to visualise.

    Returns:
        Heatmap as a (H, W) float32 array in [0, 1].
    """
    # Build a sub-model up to the target layer
    grad_model = tf.keras.Model(
        inputs=model.decoder.input,
        outputs=[
            model.decoder.get_layer(layer_name).output,
            model.decoder.output,
        ],
    )

    bits, _ = model.encoder(image, training=False)

    with tf.GradientTape() as tape:
        conv_outputs, recon = grad_model(bits, training=False)
        # Use mean reconstruction quality as the scalar loss
        loss = tf.reduce_mean(recon)

    grads = tape.gradient(loss, conv_outputs)           # (1, h, w, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    conv_outputs = conv_outputs[0]                        # (h, w, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)                         # (h, w)
    heatmap = tf.nn.relu(heatmap)

    # Normalise to [0, 1]
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_grad_cam(
    image: tf.Tensor,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on the original image (RGB uint8)."""
    img_np = np.uint8(image.numpy()[0] * 255)

    # Resize heatmap to match image spatial size
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], img_np.shape[:2]
    ).numpy()[..., 0]

    colormap = cm.get_cmap("jet")
    heatmap_rgb = np.uint8(colormap(heatmap_resized) * 255)[..., :3]
    overlay = np.uint8(img_np * (1 - alpha) + heatmap_rgb * alpha)
    return overlay


# ── Integrated Gradients ─────────────────────────────────────────────────────────

def integrated_gradients(
    model: QUIBCModel,
    image: tf.Tensor,
    baseline: tf.Tensor = None,
    steps: int = 50,
) -> np.ndarray:
    """
    Compute Integrated Gradients attribution for each pixel.

    Args:
        model:    Trained QUIBCModel.
        image:    Float32 tensor (1, H, W, 3) in [0, 1].
        baseline: Black image baseline; defaults to zeros.
        steps:    Number of interpolation steps.

    Returns:
        Attribution map (H, W, 3) – absolute values summed over channels.
    """
    if baseline is None:
        baseline = tf.zeros_like(image)

    # Interpolate between baseline and image
    alphas = tf.linspace(0.0, 1.0, steps + 1)              # (steps+1,)
    interpolated = baseline + alphas[:, None, None, None] * (image - baseline)  # (steps+1, H, W, 3)

    grads_list = []
    for i in range(steps + 1):
        inp = interpolated[i:i+1]
        with tf.GradientTape() as tape:
            tape.watch(inp)
            bits, _ = model.encoder(inp, training=False)
            recon = model.decoder(bits, training=False)
            loss = tf.reduce_mean(tf.square(inp - recon))   # reconstruction error
        grads_list.append(tape.gradient(loss, inp))

    grads = tf.concat(grads_list, axis=0)                   # (steps+1, H, W, 3)
    # Trapezoidal integration
    avg_grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = tf.reduce_mean(avg_grads, axis=0)           # (H, W, 3)

    ig = (image[0] - baseline[0]) * avg_grads               # (H, W, 3)
    return ig.numpy()


def plot_integrated_gradients(
    image: tf.Tensor,
    attributions: np.ndarray,
    save_path: str = None,
):
    """Plot original image alongside IG attribution heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image.numpy()[0])
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    attr_map = np.abs(attributions).sum(axis=-1)
    attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)
    im = axes[1].imshow(attr_map, cmap="RdYlGn")
    axes[1].set_title("Integrated Gradients Attribution")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── LRP-style relevance propagation ─────────────────────────────────────────────

def lrp_attribution(
    model: QUIBCModel,
    image: tf.Tensor,
) -> np.ndarray:
    """
    Simplified gradient × input (GI) as an LRP proxy.

    Args:
        model: Trained QUIBCModel.
        image: Float32 tensor (1, H, W, 3).

    Returns:
        Relevance map (H, W, 3).
    """
    img = tf.Variable(image)
    with tf.GradientTape() as tape:
        bits, _ = model.encoder(img, training=False)
        recon = model.decoder(bits, training=False)
        loss = tf.reduce_mean(tf.square(img - recon))

    grads = tape.gradient(loss, img)
    relevance = (img * grads).numpy()[0]     # gradient × input proxy
    return relevance


def plot_lrp(
    image: tf.Tensor,
    relevance: np.ndarray,
    recon: tf.Tensor = None,
    save_path: str = None,
):
    """Plot original, reconstruction, error map, and LRP attribution."""
    ncols = 4 if recon is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    axes[0].imshow(image.numpy()[0])
    axes[0].set_title("Original")
    axes[0].axis("off")

    lrp_map = np.abs(relevance).sum(axis=-1)
    lrp_map = (lrp_map - lrp_map.min()) / (lrp_map.max() - lrp_map.min() + 1e-8)
    axes[1].imshow(lrp_map, cmap="hot")
    axes[1].set_title("LRP Attribution")
    axes[1].axis("off")

    if recon is not None:
        axes[2].imshow(recon.numpy()[0])
        axes[2].set_title("Reconstruction")
        axes[2].axis("off")

        err = np.abs(image.numpy()[0] - recon.numpy()[0]).mean(axis=-1)
        axes[3].imshow(err, cmap="hot")
        axes[3].set_title("Error Map")
        axes[3].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Latent space visualisation ───────────────────────────────────────────────────

def extract_latent_codes(
    model: QUIBCModel,
    dataset: tf.data.Dataset,
    max_batches: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract flattened latent codes and per-sample reconstruction errors.

    Returns:
        (codes_flat, errors) arrays of shape (N, latent_dim) and (N,).
    """
    codes_list, errors_list = [], []

    for batch_idx, (x, _) in enumerate(dataset):
        if batch_idx >= max_batches:
            break
        bits, _ = model.encoder(x, training=False)
        recon = model.decoder(bits, training=False)

        err = tf.reduce_mean(tf.square(x - recon), axis=(1, 2, 3)).numpy()
        flat = bits.numpy().reshape(bits.shape[0], -1)

        codes_list.append(flat)
        errors_list.append(err)

    return np.concatenate(codes_list, axis=0), np.concatenate(errors_list, axis=0)


def plot_latent_pca(
    codes: np.ndarray,
    errors: np.ndarray,
    save_path: str = None,
):
    """PCA 2D scatter of latent codes coloured by reconstruction error."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(codes)

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(reduced[:, 0], reduced[:, 1], c=errors, cmap="viridis", alpha=0.7, s=20)
    plt.colorbar(sc, label="Reconstruction error")
    plt.title("PCA of QUIBC Latent Codes")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_latent_tsne(
    codes: np.ndarray,
    errors: np.ndarray,
    perplexity: int = 30,
    save_path: str = None,
):
    """t-SNE 2D scatter of latent codes coloured by reconstruction error."""
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(codes) - 1), random_state=42)
    reduced = tsne.fit_transform(codes)

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(reduced[:, 0], reduced[:, 1], c=errors, cmap="viridis", alpha=0.7, s=20)
    plt.colorbar(sc, label="Reconstruction error")
    plt.title("t-SNE of QUIBC Latent Codes")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_latent_umap(
    codes: np.ndarray,
    errors: np.ndarray,
    save_path: str = None,
):
    """UMAP 2D scatter (requires `umap-learn` package)."""
    try:
        import umap
    except ImportError:
        raise ImportError("Install umap-learn: pip install umap-learn")

    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(codes)

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(reduced[:, 0], reduced[:, 1], c=errors, cmap="viridis", alpha=0.7, s=20)
    plt.colorbar(sc, label="Reconstruction error")
    plt.title("UMAP of QUIBC Latent Codes")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Feature evolution ─────────────────────────────────────────────────────────────

def visualise_encoder_features(
    model: QUIBCModel,
    image: tf.Tensor,
    save_path: str = None,
    n_filters: int = 8,
):
    """
    Visualise feature maps from each encoder Conv layer.

    Args:
        model:     Trained QUIBCModel.
        image:     Float32 tensor (1, H, W, 3).
        save_path: Optional path to save the figure.
        n_filters: Number of filter channels to display per layer.
    """
    layer_names = ["enc_conv1", "enc_conv2", "enc_conv3"]
    feature_model = tf.keras.Model(
        inputs=model.encoder.input,
        outputs=[model.encoder.get_layer(n).output for n in layer_names],
    )

    features = feature_model(image, training=False)   # list of tensors

    fig, axes = plt.subplots(len(layer_names), n_filters, figsize=(n_filters * 2, len(layer_names) * 2 + 1))
    fig.suptitle("Encoder Feature Maps", fontsize=14)

    for row, (feat_map, name) in enumerate(zip(features, layer_names)):
        feat = feat_map.numpy()[0]   # (h, w, C)
        for col in range(n_filters):
            ax = axes[row, col]
            ch = feat[..., col]
            ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
            ax.imshow(ch, cmap="viridis")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(name, fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
