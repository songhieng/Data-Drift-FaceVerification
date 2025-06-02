**Overview**
In a face‐recognition system, the feature extractor (often a deep CNN) maps input images to embedding vectors. Over time, incoming face images (e.g., from new cameras, new lighting conditions, or shifts in the subject population) may differ statistically from the data used to train the feature extractor. If the embedding distribution drifts far enough, downstream tasks—such as identity verification, clustering, or re‐enrollment—can degrade. Detecting such “data drift” efficiently is therefore critical. Below is a structured guide to understanding, choosing, and implementing a practical drift‐detection pipeline for face‐recognition embeddings.

---

## 1. What Is Data Drift in Feature‐Extractor Outputs?

* **Covariate (Input) Drift**
  Occurs when the raw input distribution changes (e.g., variations in camera sensors, new ethnicities, aging faces). Even if the extractor itself remains unchanged, these shifts manifest in the embedding space.
* **Feature (Embedding) Drift**
  Refers more directly to shifts in the embedding vectors (deep features) that the model produces. For example, if the mean embedding norm or distribution of principal components drifts, downstream similarity scores can shift.
* **Why Monitor Embeddings Directly?**

  * Embedding vectors are lower‐dimensional (e.g., 512‐D) than raw images, making statistical tests more tractable.
  * Drift in embeddings directly correlates with changes in similarity distributions—crucial for face verification thresholds.
  * By keeping an up‐to‐date “reference embedding distribution,” you can detect drift without access to labels (unlabeled drift detection).

---

## 2. Common Drift‐Detection Techniques for Embeddings

1. **Population Stability Index (PSI)**

   * **What it measures**: The PSI quantifies how much a distribution has shifted between two samples. Applied dimension‐wise (or on a univariate projection), it’s

     $$
     \text{PSI} = \sum_{i=1}^k \left( P_i - Q_i \right) \ln \frac{P_i}{Q_i}
     $$

     where $P_i$ is the proportion of reference data in bin $i$, and $Q_i$ is the proportion of new data in the same bin.
   * **Pros**:

     * Simple to implement.
     * Fast for univariate or low‐dimensional features.
   * **Cons**:

     * Binning strategy in high‐D can be tricky.
     * Doesn’t capture multi‐dimensional dependencies if applied separately to each feature.
   * **Usage in embeddings**:

     * **Option A**: Compute PSI for each embedding dimension independently, then aggregate (e.g., average or max across dimensions).
     * **Option B**: Project embeddings onto top‐$k$ principal components (via PCA) and compute PSI on those components (reduces dimensionality to, say, 10–20).

2. **Maximum Mean Discrepancy (MMD)**

   * **What it measures**: A kernel‐based two‐sample test that measures the distance between distributions $P$ (reference) and $Q$ (new) in a reproducing‐kernel Hilbert space (RKHS).
   * **Pros**:

     * Captures high‐order differences in distributions.
     * Nonparametric (no binning).
   * **Cons**:

     * Computation cost is $O(n^2)$ if done naïvely (though linear approximations exist).
     * Requires tuning a kernel (usually Gaussian) bandwidth.
   * **Usage in embeddings**:

     1. Sample $n$ embeddings from reference and $n$ from current (e.g., 1,000 each).
     2. Compute MMD statistic using an RBF kernel (bandwidth chosen via median heuristic).
     3. Compare against a threshold or use a bootstrap to assess significance.

3. **Kolmogorov–Smirnov (KS) Test (Dimension‐wise)**

   * **What it measures**: For each dimension $d$, the KS test compares the 1D empirical CDFs of reference vs. current data.
   * **Pros**:

     * Nonparametric; no binning.
     * Fast for individual dimensions.
   * **Cons**:

     * Only tests univariate differences; loses joint‐distribution information.
   * **Usage in embeddings**:

     1. Reduce dimensionality (e.g., via PCA) to a handful of principal components.
     2. Run KS test on each principal component’s distribution.
     3. If any PC fails at $p < 0.01$ (Bonferroni‐corrected), flag drift.

4. **Mahalanobis‐Distance‐Based Drift**

   * **What it measures**: If embeddings are approximately Gaussian under reference, measure how far new embeddings are from the reference mean in terms of Mahalanobis distance.
   * **Pros**:

     * Captures correlation via covariance matrix.
     * Single‐statistic check possible if projecting onto low‐D.
   * **Cons**:

     * Reference covariance must be estimated robustly (requires sufficient data).
     * Assumes roughly Gaussian distribution in embedding space.
   * **Usage in embeddings**:

     1. Fit mean $\mu$ and covariance $\Sigma$ on a baseline set of embeddings.
     2. For each new embedding $x$, compute $D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$.
     3. Monitor the distribution of $D_M$ (e.g., by quantiles). If the upper‐quantile of new data’s $D_M$ exceeds a threshold (e.g., 99th percentile from reference), flag drift.

5. **Online/Streaming Detectors (e.g., ADWIN, DDM)**

   * **What they do**: Maintain an online “window” of recent embeddings (or of a univariate statistic like embedding norm, or top‐PC projection) and detect significant changes over time.
   * **Pros**:

     * Adapt to gradual drift in real‐time.
     * Lightweight (operate on one dimension or summary statistic).
   * **Cons**:

     * If you use only one statistic (e.g., embedding norm), you might miss drift elsewhere in space.
     * Less sensitive to multi‐dimensional shifts unless you run multiple detectors.
   * **Usage in embeddings**:

     * For **embedding norm**: keep ADWIN on $\|x\|$.
     * For **PC‐1 projection**: keep ADWIN on $\text{PC}_1(x)$.
     * For **similarity scores**: if your pipeline computes similarity (e.g., cosine similarity) between pairs, monitor the distribution of those scores.

---

## 3. Putting It All Together: A Practical Pipeline

Below is a step‐by‐step outline for an **efficient drift‐detection pipeline** tailored to face‐recognition embeddings. The key principle is to (a) keep a compact but representative baseline of embeddings, (b) apply a drift test in a reduced subspace, and (c) automate alerts when thresholds are crossed.

---

### 3.1. Step 1: Choose a Reference (Baseline) Embedding Set

1. **Collect a “Reference Period”**

   * Use a diverse set of face images from your known “good” distribution (e.g., images used in initial training or a held‐out validation set).
   * Aim for at least 5,000–10,000 embeddings to estimate stable statistics (mean, covariance, PC loadings).
2. **Compute Embeddings Once**

   ```python
   import torch
   from your_face_model import FeatureExtractor  # whatever library you use

   extractor = FeatureExtractor(pretrained=True)
   extractor.eval().cuda()

   # Suppose `ref_images` is a list of tensors [B×3×224×224]
   ref_embeddings = []
   for batch in ref_images_batches:
       with torch.no_grad():
           emb = extractor(batch.cuda())  # shape: [B, D]
           ref_embeddings.append(emb.cpu())
   ref_embeddings = torch.cat(ref_embeddings, dim=0).numpy()  # shape: [N_ref, D]
   ```
3. **Store Key Statistics**

   * **Mean vector** $\mu \in \mathbb{R}^D$
   * **Covariance matrix** $\Sigma \in \mathbb{R}^{D \times D}$ (or an estimate thereof)
   * **Principal Components** (fit PCA once, keep top $k$, e.g., $k=10$).
   * **Univariate distributions** (histograms) of each PC component (for PSI or KS).
   * Optionally, store a small subset (e.g., 1,000) of raw embeddings for a “bootstrap reference” if you plan to run MMD with sub‐sampling.

---

### 3.2. Step 2: Decide on a Drift Test

Below is a ranked order of **efficiency vs. sensitivity trade‐offs**. For a production system, start with a lightweight test (e.g., PSI on top PCs) and then optionally add a stronger test (e.g., MMD).

| Method                      | Complexity                                          | Sensitivity                             | Recommended Use Case                                                                              |
| --------------------------- | --------------------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **PSI on Top $k$ PCs**      | $O(N + Dk)$ per batch                               | Moderate: catches univariate shifts     | If you need very fast checks and interpretability. Best as first‐line monitoring.                 |
| **KS on Top $k$ PCs**       | $O(N \log N)$ per batch                             | Moderate–High for shifts in marginals   | If you want a non‐parametric check without binning. Slightly more expensive but still fast.       |
| **Mahalanobis Distance**    | $O(N D^2)$ to estimate $\Sigma$; $O(ND)$ per sample | Good for Gaussian‐like reference        | If embeddings are roughly Gaussian. Good single‐valued drift metric. Requires stable $\Sigma$.    |
| **ADWIN on 1D Projections** | $O(N)$ streaming                                    | Good for real‐time gradual drift        | If you want continuous monitoring (e.g., production service). Best on a single summary statistic. |
| **MMD (RBF Kernel)**        | $O(n^2)$ naive, $O(n)$ with approximations          | High: captures distribution differences | If you need a “gold‐standard” batch test occasionally (e.g., daily or weekly). More costly.       |

> **Recommendation**
>
> * **First‐line (online)**: Monitor a univariate statistic such as the norm $\|x\|$ or the score on the first principal component PC₁ via ADWIN. This gives a quick signal if something “major” is changing.
> * **Second‐line (batch, daily/weekly)**: Run PSI or KS on the top $k$ PCA projections. If PSI > 0.1 for any PC, flag drift. Optionally, follow up with an MMD test on a subsample for confirmation.

---

### 3.3. Step 3: Extract and Stream New Embeddings

1. **Collect Incoming Images in “Batches” or “Windows”**

   * E.g., every hour or every day, sample up to $N_{\text{new}}=2{,}000$ new face embeddings (could be random if throughput is high).
2. **Compute Embeddings with the Same Preprocessing**

   ```python
   # Example: every hour, collect new_images (list of tensors):
   new_embeddings = []
   for batch in new_images_batches:
       with torch.no_grad():
           emb = extractor(batch.cuda())
           new_embeddings.append(emb.cpu())
   new_embeddings = torch.cat(new_embeddings, dim=0).numpy()  # [N_new, D]
   ```
3. **Update Online Detectors (if using ADWIN/DDM)**

   * E.g., feed $\|new\_emb_i\|$ or $\text{PC}_1(new\_emb_i)$ into ADWIN one at a time (in streaming fashion).
   * If ADWIN outputs “change detected,” immediately raise an alert.

---

### 3.4. Step 4: Batch Drift Test (PSI or KS on Top $k$ PCs)

1. **Project Both Reference and New Embeddings into Top $k$ PCs**

   ```python
   from sklearn.decomposition import PCA

   # Suppose you already fit PCA on ref_embeddings and saved pca_model:
   # pca_model.components_ is [k, D]
   ref_pca = pca_model.transform(ref_embeddings)  # shape: [N_ref, k]
   new_pca = pca_model.transform(new_embeddings)  # shape: [N_new, k]
   ```

2. **Compute PSI per Dimension**

   ```python
   import numpy as np

   def compute_psi(ref_vals: np.ndarray, new_vals: np.ndarray, bins=20):
       """
       Compute PSI between reference and new 1D arrays of values.
       Returns PSI scalar.
       """
       hist_ref, bin_edges = np.histogram(ref_vals, bins=bins, density=False)
       hist_new, _ = np.histogram(new_vals, bins=bin_edges, density=False)
       # Convert counts to proportions
       per_ref = hist_ref / len(ref_vals)
       per_new = hist_new / len(new_vals)
       # To avoid division by zero, replace 0s with a small value
       eps = 1e-8
       per_ref = np.where(per_ref == 0, eps, per_ref)
       per_new = np.where(per_new == 0, eps, per_new)
       psi = np.sum((per_ref - per_new) * np.log(per_ref / per_new))
       return psi

   # Compute PSI on each PCA dimension
   psi_values = []
   for dim in range(k):
       psi_dim = compute_psi(ref_pca[:, dim], new_pca[:, dim], bins=50)
       psi_values.append(psi_dim)
   ```

3. **Aggregate and Compare to Threshold**

   * Common rule‐of‐thumb:

     * PSI < 0.1  → No significant drift
     * 0.1 ≤ PSI < 0.25 → Moderate drift (investigate)
     * PSI ≥ 0.25 → Major drift
   * If **any** of the top $k$ PC dimensions has PSI ≥ 0.1, trigger a “drift alert.” Optionally, compute a “total PSI” by summing all $k$ PSIs and compare to a threshold (e.g., total ≥ 0.5).

4. **(Alternatively) KS Test for Each PC**

   ```python
   from scipy.stats import ks_2samp

   ks_pvalues = []
   for dim in range(k):
       stat, pvalue = ks_2samp(ref_pca[:, dim], new_pca[:, dim])
       ks_pvalues.append(pvalue)

   # Bonferroni correction: significance = 0.05 / k
   sig_level = 0.05 / k
   drift_flags = [p < sig_level for p in ks_pvalues]
   if any(drift_flags):
       print("KS test indicates drift on dims:", [i for i, f in enumerate(drift_flags) if f])
   ```

   * Even faster than PSI (direct CDF comparison), but less interpretable than PSI’s “magnitude.”

---

### 3.5. Step 5: (Optional) MMD Test for Confirmation

1. **Subsample (if too large)**

   * Choose $n_{\text{sub}} = 500$ from reference, $n_{\text{sub}} = 500$ from new.
2. **Compute MMD with RBF Kernel**

   ```python
   import numpy as np

   def rbf_kernel(x, y, gamma):
       # x: [n, d], y: [m, d]
       # Returns Gram matrix [n, m]
       sq_dists = (
           np.sum(x**2, axis=1)[:, None]
           + np.sum(y**2, axis=1)[None, :]
           - 2 * np.dot(x, y.T)
       )
       return np.exp(-gamma * sq_dists)

   def mmd_rbf(x_ref, x_new, gamma=None):
       n, d = x_ref.shape
       m, _ = x_new.shape
       if gamma is None:
           # median heuristic: gamma = 1/(2*sigma^2), sigma^2 = median pairwise distance
           combined = np.vstack([x_ref, x_new])
           dists = np.sum((combined[:, None, :] - combined[None, :, :])**2, axis=2)
           median_val = np.median(dists)
           gamma = 1.0 / (2 * median_val + 1e-8)
       K_xx = rbf_kernel(x_ref, x_ref, gamma)
       K_yy = rbf_kernel(x_new, x_new, gamma)
       K_xy = rbf_kernel(x_ref, x_new, gamma)
       mmd_stat = (
           np.sum(K_xx) / (n * n)
           + np.sum(K_yy) / (m * m)
           - 2 * np.sum(K_xy) / (n * m)
       )
       return mmd_stat

   # Example usage:
   ref_sub = ref_embeddings[np.random.choice(len(ref_embeddings), 500, replace=False)]
   new_sub = new_embeddings[np.random.choice(len(new_embeddings), 500, replace=False)]
   mmd_value = mmd_rbf(ref_sub, new_sub)
   print("MMD statistic:", mmd_value)
   ```
3. **Determine Significance**

   * Use a **permutation test**: shuffle combined data labels many times (e.g., 100–200 permutations) to build a null distribution of MMD, then see if observed MMD > 95th percentile of null.
   * If so, flag drift.

> **Note:** MMD is more computationally expensive and typically run less frequently (e.g., daily/weekly), whereas PSI/KS can run in near‐real‐time (every batch).

---

## 4. Additional Considerations

1. **Embedding Normalization**

   * Many face models $L^2$-normalize embeddings to unit length. In that case:

     * **PSI on Norm** will be meaningless (constant = 1). Instead, monitor the distribution of the embedding’s first few coordinates or similarity scores with known anchor embeddings.
     * If embeddings are unit‐norm, consider monitoring the distribution of cosine similarity between random pairs in the batch—if the overall similarity shifts, that indicates drift.

2. **Similarity‐Score Monitoring**

   * If your pipeline already computes similarity scores (e.g., between newly enrolled faces and a gallery), monitor the **histogram of top‐1 cosine similarity**.
   * A sudden shift in similarity distributions (e.g., more “low‐similarity” matches) signals drift.
   * Use a univariate drift detector (ADWIN or PSI) on these scores.

3. **Dimensionality Reduction**

   * Directly running drift tests in a 512‐D space is both memory‐ and compute‐heavy. Always reduce to $k \ll 512$ (e.g., $k=10$ or $k=20$) via PCA or even UMAP (for visualization).
   * Save the PCA transform fit on reference data. **Do not re‐fit PCA on the new batch**, since that would conflate drift. Always project new data onto the **frozen** PCA from reference.

4. **Automating Alerts & Logging**

   * Once a test (e.g., PSI on PC dims) exceeds threshold, automatically raise an alert (e.g., send Slack/Email, spin up a Kubernetes job to re‐evaluate the feature extractor).
   * Log the following for each batch:

     * Number of samples processed
     * Mean PSI per PC
     * Number of PCs exceeding threshold
     * MMD value (if run)
     * Drift flag (yes/no)
   * Over time, you can visualize drift metrics trending upward to anticipate when to retrain/fine‐tune the feature extractor.

5. **Retraining Triggers**

   * Decide on a policy (e.g., if drift is flagged two days in a row, schedule a model check).
   * When flagged, you might:

     1. Sample a small labeled set to evaluate actual face‐verification performance.
     2. If performance drops (e.g., AUC < 0.98), schedule retraining/fine‐tuning on more recent data.

6. **Tools & Libraries**

   * **[Alibi Detect](https://github.com/SeldonIO/alibi-detect)**

     * Contains implementations of MMD drift, KDE drift, PSI, and more. Works out‐of‐the‐box on embedding arrays.
   * **[DeepChecks](https://github.com/deepchecks/deepchecks)**

     * Provides sanity tests and drift tests for tabular and embedding data, at both batch and slice levels.
   * **[River](https://riverml.xyz/)**

     * A streaming ML library with ADWIN, DDM, EDDM for online drift. Can keep a rolling window of a univariate statistic.
   * **[NannyML](https://github.com/NannyML/nannyml)**

     * Focuses on performance‐based drift detection (needs some labeled data).
   * You can also implement custom PSI/MMD/KS without external dependencies, as shown above.

---

## 5. Concrete Implementation Example

Below is a **concise blueprint** you can adapt. It combines:

1. Online monitoring on the first PCA component via ADWIN.
2. Daily PSI check on top $k=10$ PCA components.
3. Weekly MMD confirmation if PSI flags on more than 3 PCs.

```python
import numpy as np
import torch
from sklearn.decomposition import PCA
from river import drift  # for ADWIN
from datetime import datetime, timedelta

class EmbeddingDriftMonitor:
    def __init__(self, ref_embeddings: np.ndarray, extractor, device="cuda"):
        """
        ref_embeddings: [N_ref, D] baseline
        extractor: PyTorch model mapping image→embedding
        """
        # 1) Fit PCA on reference
        self.pca = PCA(n_components=10)
        self.ref_pca = self.pca.fit_transform(ref_embeddings)  # [N_ref, 10]
        
        # 2) Compute and store histograms (bins) for PSI
        self.bins_per_dim = 50
        self.ref_hist = []
        for dim in range(10):
            hist, bins = np.histogram(
                self.ref_pca[:, dim],
                bins=self.bins_per_dim,
                density=False
            )
            # store both counts and edges
            self.ref_hist.append((hist, bins))
        
        # ADWIN on PC₁ (streaming)
        self.adwin = drift.ADWIN(delta=0.002)  # delta tuned for sensitivity
        self.device = device
        self.extractor = extractor.eval().to(device)
        
        # For scheduling
        self.last_daily_check = datetime.min
        self.last_weekly_check = datetime.min
        
        # Keep a small reservoir of embeddings for weekly MMD
        self.weekly_buffer = []
        self.weekly_buffer_size = 1000  # max embeddings
        
    def _compute_embedding(self, images: torch.Tensor) -> np.ndarray:
        # images: [B, 3, H, W], normalized appropriately
        with torch.no_grad():
            emb = self.extractor(images.to(self.device))  # [B, D]
        return emb.cpu().numpy()
    
    def _psi_dim(self, ref_vals, new_vals, bins):
        hist_ref, bin_edges = bins
        hist_new, _ = np.histogram(new_vals, bins=bin_edges, density=False)
        per_ref = hist_ref / len(ref_vals)
        per_new = hist_new / len(new_vals)
        eps = 1e-8
        per_ref = np.where(per_ref == 0, eps, per_ref)
        per_new = np.where(per_new == 0, eps, per_new)
        psi = np.sum((per_ref - per_new) * np.log(per_ref / per_new))
        return psi
    
    def _batch_psi(self, new_embeddings: np.ndarray) -> list:
        new_pca = self.pca.transform(new_embeddings)  # [N_new, 10]
        psi_vals = []
        for dim in range(10):
            ref_vals = self.ref_pca[:, dim]
            new_vals = new_pca[:, dim]
            hist, bins = self.ref_hist[dim]
            psi_vals.append(self._psi_dim(ref_vals, new_vals, (hist, bins)))
        return psi_vals
    
    def _mmd(self, ref_sub, new_sub):
        # Use the mmd_rbf implementation from §3.5
        return mmd_rbf(ref_sub, new_sub)
    
    def process_batch(self, new_images: torch.Tensor, timestamp: datetime):
        """
        - new_images: [B, 3, H, W] for this time window
        - timestamp: current datetime
        """
        # Step A: Compute embeddings
        new_embs = self._compute_embedding(new_images)  # [B, D]
        
        # Step B: Online ADWIN on PC₁
        new_pca1 = self.pca.transform(new_embs)[:, 0]  # 1D array
        for val in new_pca1:
            if self.adwin.update(val):
                print(f"[{timestamp}] ADWIN triggered on PC₁ streaming—potential drift!")
                break
        
        # Step C: Add to weekly buffer
        self.weekly_buffer.append(new_embs)
        if len(self.weekly_buffer) * new_embs.shape[0] > self.weekly_buffer_size:
            # Keep only the most recent embeddings
            self.weekly_buffer = self.weekly_buffer[-(self.weekly_buffer_size // new_embs.shape[0] + 1):]
        
        # Step D: Time‐based checks
        #  - Daily PSI check
        if timestamp - self.last_daily_check > timedelta(days=1):
            all_new_embs = np.vstack(self.weekly_buffer)  # maybe up to 1k embeddings
            psi_vals = self._batch_psi(all_new_embs)
            flagged_dims = [i for i, psi in enumerate(psi_vals) if psi >= 0.1]
            if flagged_dims:
                print(f"[{timestamp}] PSI drift on dimensions: {flagged_dims}")
                # If more than 3 dims drifted, schedule weekly MMD immediately
                if len(flagged_dims) > 3:
                    self.last_weekly_check = datetime.min  # force weekly MMD now
            self.last_daily_check = timestamp
        
        #  - Weekly MMD check
        if timestamp - self.last_weekly_check > timedelta(weeks=1):
            # Take subsample of ref and new
            ref_sub = self.ref_pca[np.random.choice(len(self.ref_pca), 500, replace=False)]
            new_sub = self.pca.transform(all_new_embs)
            new_sub = new_sub[np.random.choice(len(new_sub), 500, replace=False)]
            mmd_stat = self._mmd(ref_sub, new_sub)
            if mmd_stat > 1e-3:  # threshold depends on scale; test based on historical runs
                print(f"[{timestamp}] MMD indicates drift: {mmd_stat:.6f}")
            self.last_weekly_check = timestamp
```

> **How This Pipeline Works**
>
> 1. **Streaming (Real‐Time) Check**: Every new embedding’s PC₁ value feeds into ADWIN. If ADWIN decides that the distribution of PC₁ has changed (fast), it raises an immediate alert.
> 2. **Daily Batch PSI**: Once per day, we gather up to 1,000 new embeddings, project them to the same PCA space, and compute PSI across each of the top 10 PCs. If more than a threshold of PCs show PSI ≥ 0.1, we log a moderate‐drift alert—and if it’s a large drift (e.g., ≥ 3 dims), we expedite the weekly MMD check.
> 3. **Weekly MMD Confirmation**: Independent of PSI, once per week we run an MMD two‐sample test between subsampled reference and new embeddings. If MMD exceeds a preset threshold (determined empirically), we log a confirmed drift event.

---

## 6. Why This Approach Is Efficient and Effective

1. **Dimensionality Reduction to $k=10$**

   * Face‐recognition embeddings are often 512–1,024 D. Direct high‐D tests are expensive. By fitting PCA once and projecting to the top 10 PCs, we preserve most variance (\~80–90%) while drastically reducing test complexity.

2. **Combination of Online + Batch Tests**

   * **ADWIN on PC₁** catches sudden, dramatic shifts in “dominant” variation direction. It’s $O(1)$ per sample, so negligible overhead.
   * **PSI on PCs** is $O(N_{\text{batch}} \times k)$ per day. Even for $N_{\text{batch}}=1{,}000$, $k=10$, that’s 10,000 operations—trivial.
   * **MMD, run weekly** on subsamples of size 500, is $O(n^2)\approx O(250{,}000)$ kernel operations once a week. It’s acceptable as a relatively infrequent “gold‐standard” check.

3. **Modular Alerts**

   * You get **immediate feedback** if something goes catastrophically wrong (e.g., camera feed change) via ADWIN.
   * You get **daily sense** of slower shifts via PSI.
   * You get **robust quarterly or weekly check** with MMD.

4. **Few Hyperparameters**

   * PSI bins (typically 50–100).
   * PCA dimension $k$ (10 is a common sweet spot).
   * ADWIN delta (can be tuned to typical noise in PC₁).
   * MMD kernel bandwidth—use the median heuristic or fix from initial runs.

5. **Relative Independence from Labels**

   * This entire pipeline relies only on unlabeled embeddings. You do **not** need ground‐truth face IDs to detect drift—enabling unsupervised monitoring in production.

---

## 7. Key Takeaways and Recommendations

1. **Start Small, Scale Later**

   * If you need a quick proof‐of‐concept: monitor just the embedding norm or PC₁ with ADWIN. That requires almost zero extra storage or CPU.
2. **Combine Multiple Tests**

   * A single univariate test can miss drift in some directions. Use at least the top $k=5$ PCs for PSI.
3. **Regularly Update Your Reference**

   * Every **quarter** (or after significant retraining), re‐compute your reference embeddings/PCA from the latest stable dataset. Otherwise, your drift detector itself may become obsolete.
4. **Tune Thresholds Based on Historical Data**

   * Run PSI and MMD on hold‐out splits of your reference data to establish a baseline distribution of metrics. Set alert thresholds (e.g., PSI = 0.1, MMD above 90th percentile of bootstrapped null) accordingly.
5. **Automate Retraining Triggers**

   * When drift is flagged, integrate a small pipeline: sample 200–500 labeled face pairs, compute actual verification accuracy, and if performance < target, queue model retraining/fine‐tuning.
6. **Use Existing Libraries When Possible**

   * If you prefer not to implement from scratch, consider using [Alibi Detect’s drift modules](https://github.com/SeldonIO/alibi-detect) for MMD, and [River’s drift detectors](https://riverml.xyz/latest/api/drift/). They already implement robust, tested code.

---

### Example Summary Table of Methods

| Method                   | Complexity (per batch)              | Detects                        | Pros                                     | Cons                                               |
| ------------------------ | ----------------------------------- | ------------------------------ | ---------------------------------------- | -------------------------------------------------- |
| **ADWIN on PC₁**         | $O(1)$ per sample                   | Sudden shifts in dominant axis | Real‐time, very fast                     | Only 1D; may miss shifts in other PCs              |
| **PSI on Top $k$ PCs**   | $O(N_{\text{batch}} \times k)$      | Moderate to large shifts       | Interpretable, fast univariate check     | Loses joint distribution info                      |
| **KS on Top $k$ PCs**    | $O(N_{\text{batch}} \log N)$        | Univariate distribution shift  | Nonparametric, no binning needed         | Less interpretable; needs Bonferroni correction    |
| **Mahalanobis Distance** | $O(ND)$ per sample                  | Gaussian latent‐space shift    | Single‐value metric; captures covariance | Assumes Gaussianity; expensive covariance estimate |
| **MMD (RBF Kernel)**     | $O(n^2)$ or $O(n)$ w/ approximation | Any distribution difference    | Sensitive to high‐order differences      | Computationally expensive; kernel tuning needed    |

---

## 8. Final Recommendations

1. **Implement a Two‐Tier System**

   * **Tier 1 (Real Time)**:

     * Track PC₁ via ADWIN.
     * Bonus: also track embedding norm if extractor is not unit‐norm.
   * **Tier 2 (Daily)**:

     * Compute PSI on PCs (at least $k=5$, preferably $k=10$).
   * **Tier 3 (Weekly)**:

     * Run MMD on subsamples (confirm large drifts).

2. **Use Dimension Reduction**

   * Always freeze PCA from reference; do not re‐fit on new data.
   * Keep $k=5$–$10$ principal components, which usually capture > 80 % of variance in face embeddings.

3. **Define Concrete Alert Thresholds**

   * **ADWIN**: Let River’s default delta = 0.002 initially; adjust if too many false positives.
   * **PSI**: Per‐dimension threshold = 0.1. If more than 2 dims exceed 0.1 in a day, investigate.
   * **MMD**: After bootstrapping on reference, set threshold at, say, 95th percentile of null.

4. **Automate Data Collection and Logging**

   * Wrap embedding extraction and drift‐testing logic into a microservice (e.g., a Python script scheduled via cron or Kubernetes CronJob).
   * Log metrics (PSI, ADWIN drift event timestamps, MMD) to a time‐series DB (e.g., InfluxDB, Prometheus).
   * Build a lightweight dashboard to visualize drift metrics over time (e.g., Grafana).

5. **Plan for Remediation**

   * When drift is detected:

     1. **Validate**: Quickly sample 100–200 labeled pairs from new batch to check verification accuracy.
     2. **If accuracy < target**: Schedule fine‐tuning of the feature extractor on more recent data (e.g., incorporate new demographics or new lighting conditions).

By following this structured, multi‐tiered approach—combining both streaming checks (ADWIN) and batch tests (PSI, KS, MMD)—you’ll achieve an efficient yet robust data‐drift detection system tailored for face‐recognition feature extractors. If you want ready‐made implementations, look into **Alibi Detect** for batch‐mode drift testing (PSI, MMD) and **River** for stream‐mode detectors (ADWIN, DDM). Tailor thresholds based on your system’s historical variations, and you’ll have an early‐warning setup that preserves your face‐recognition model’s reliability over time.
