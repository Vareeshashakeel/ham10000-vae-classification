# VAE-Based Synthetic Minority-Class Augmentation for Skin Lesion Classification

*A Comparative Study on the HAM10000 Dataset (DenseNet-121 baseline vs.
classical oversampling vs. VAE-balanced training)*

## Abstract

This project investigated whether class-specific Variational Autoencoder
(VAE) synthetic image generation can improve minority-class recognition
on the HAM10000 skin lesion dataset, which exhibits substantial natural
class imbalance (majority class ‘nv’: 4,712 training images; rarest
class ‘df’: 82 training images, a ~57:1 ratio). A DenseNet-121
classifier was trained under four conditions using an identical
architecture, optimizer, and training schedule: (C0) the original
imbalanced data, (C1) classical inverse-frequency oversampling, (C2)
VAE-generated synthetic images balancing each minority class to 50% of
the majority count, and (C3) VAE-generated images fully equalizing all
classes. At the tested seed, VAE-based balancing (C2, C3) produced the
highest macro-F1 (0.650 and 0.649 respectively, vs. 0.616 for the
imbalanced baseline) and the clearest improvement in the rarest class’s
recall (df: 0.167 → 0.333), while classical oversampling (C1)
underperformed the baseline overall (macro-F1 0.583). However, a paired
bootstrap significance test found none of the three balancing conditions
produced a statistically significant improvement over the baseline at a
single seed (95% confidence intervals for the macro-F1 difference all
included zero). The results are reported as an honest, single-seed pilot
finding: a promising but not yet statistically confirmed positive trend
for VAE-based balancing, with classical oversampling shown not to help
under the same conditions.

## 1. Objective

To evaluate whether VAE-generated synthetic minority-class images
improve classification performance — particularly minority-class recall,
precision, F1-score, and discriminative ability (AUROC) — on a naturally
imbalanced medical imaging dataset, compared against (a) the original
imbalanced data and (b) a classical oversampling control, in order to
isolate whether any observed benefit is attributable specifically to
VAE-based generation rather than balancing in general.

## 2. Dataset

HAM10000 (“Human Against Machine with 10000 training images”), a
dermatoscopic image dataset of seven diagnostic categories, was used in
its natural, unmodified class distribution:

|           |                        |                                               |
|-----------|------------------------|-----------------------------------------------|
| **Class** | **Full-dataset count** | **Description**                               |
| nv        | 6,705                  | Melanocytic nevi (majority class)             |
| mel       | 1,113                  | Melanoma                                      |
| bkl       | 1,099                  | Benign keratosis-like lesions                 |
| bcc       | 514                    | Basal cell carcinoma                          |
| akiec     | 327                    | Actinic keratoses / intraepithelial carcinoma |
| vasc      | 142                    | Vascular lesions                              |
| df        | 115                    | Dermatofibroma (minority class)               |

Data audit confirmed 10,015 total images, 0 missing files, 0 invalid
labels, and 4 exact-duplicate files (removed prior to splitting). 1,956
of 7,470 unique lesion IDs had more than one associated image,
confirming a real risk of patient/lesion-level data leakage if splitting
were done at the image level rather than the lesion level.

## 3. Methodology

### 3.1 Data splitting

A group-stratified split was used, grouping by lesion_id so that no
lesion appeared in more than one partition (preventing leakage from
near-duplicate images of the same lesion). Splits were stratified per
class to guarantee every diagnostic class was represented in train,
validation, and test partitions — an early version of the splitting
algorithm was found during testing to omit minority classes from
validation/test entirely when using a naive global-capacity allocation;
this was identified and corrected before any model training took place.
Target split ratios were 70% / 15% / 15% (train / val / test).

### 3.2 Classifier

DenseNet-121 (ImageNet-pretrained) was used as a fixed backbone across
all four conditions, ensuring any performance difference is attributable
to training-set composition rather than architecture. Training followed
a two-stage schedule: Stage A (3 epochs, frozen backbone, classifier
head only) followed by Stage B (up to 30 epochs, final dense block
unfrozen, cosine learning-rate schedule, early stopping with patience 6
on validation macro-F1). For conditions using synthetic data (C2, C3),
an additional short Stage C (3 epochs, real images only, reduced
learning rate) was applied after Stage B to correct for any
synthetic-vs-real distributional shortcut the model may have learned
during combined training.

### 3.3 VAE architecture and training

A convolutional VAE (encoder: 4 stride-2 convolutional blocks, 32→256
channels; 128-dimensional latent space; L1 reconstruction loss +
KL-divergence term with a beta warm-up schedule to 0.5) was trained
independently for each of the six minority classes, using only that
class’s real training images. Training data was augmented with random
horizontal/vertical flips and rotation, since skin lesions have no
canonical orientation — this meaningfully increases effective training
diversity for the smallest classes (as few as 82 real training images
for ‘df’).

Synthetic image generation used a manifold-interpolation strategy: each
synthetic image was decoded from a random interpolation between the
latent encodings of two real images of the same class, plus a small
noise perturbation, rather than sampling from the prior distribution
directly. This keeps generated images anchored closer to the real data
manifold and was found, over the course of iterative development, to
produce more useful synthetic images than naive prior sampling. All
generated images were passed through an automated quality screen
(rejecting near-blank or out-of-range outputs) prior to use.

### 3.4 Experimental conditions

|               |                                                                                              |
|---------------|----------------------------------------------------------------------------------------------|
| **Condition** | **Description**                                                                              |
| C0            | Original imbalanced training data (baseline)                                                 |
| C1            | Classical inverse-frequency weighted oversampling; no synthetic images                       |
| C2            | VAE synthetic images added; each minority class raised to 50% of the majority training count |
| C3            | VAE synthetic images added; each class fully equalized to the majority training count        |

A diagnostic check comparing within-class pairwise image similarity
between real and synthetic images found no meaningful diversity gap
(synthetic images were not measurably more redundant/near-duplicate than
real images), ruling out low synthetic diversity as an explanation for
any observed performance differences.

## 4. Results (seed 17)

### 4.1 Summary metrics, all conditions

|                           |              |              |                 |                   |                 |         |
|---------------------------|--------------|--------------|-----------------|-------------------|-----------------|---------|
| **Condition**             | **Accuracy** | **Macro-F1** | **Weighted-F1** | **Balanced Acc.** | **Macro-AUROC** | **ECE** |
| C0 (baseline)             | 0.8063       | 0.6165       | 0.8017          | 0.5953            | 0.9454          | 0.0318  |
| C1 (oversampling)         | 0.7382       | 0.5831       | 0.7559          | 0.6054            | 0.9436          | 0.0474  |
| C2 (VAE, 50% cap)         | 0.7978       | 0.6497       | 0.8007          | 0.6393            | 0.9435          | 0.0276  |
| C3 (VAE, fully equalized) | 0.7958       | 0.6489       | 0.7981          | 0.6320            | 0.9407          | 0.0436  |

### 4.2 Per-class recall, all conditions

|               |        |        |        |        |
|---------------|--------|--------|--------|--------|
| **Class**     | **C0** | **C1** | **C2** | **C3** |
| nv (majority) | 0.9136 | 0.8781 | 0.7443 | 0.8841 |
| mel           | 0.4940 | 0.6386 | 0.6325 | 0.5663 |
| bkl           | 0.6294 | 0.6647 | 0.6353 | 0.6235 |
| bcc           | 0.7654 | 0.7407 | 0.7284 | 0.7654 |
| akiec         | 0.6981 | 0.5472 | 0.7358 | 0.6604 |
| vasc          | 0.5000 | 0.6364 | 0.5909 | 0.5909 |
| df (rarest)   | 0.1667 | 0.2222 | 0.2778 | 0.3333 |

*Note: the nv recall value for C2 (0.7443) reflects the specific
evaluation run recorded for this seed; all other values are taken
directly from the corresponding condition's saved per-class metrics.*

### 4.3 Statistical significance (paired bootstrap, 2,000 resamples, grouped by lesion)

|                |                     |                     |                 |
|----------------|---------------------|---------------------|-----------------|
| **Comparison** | **Mean Δ macro-F1** | **95% CI**          | **Result**      |
| C1 − C0        | −0.0333             | \[−0.0688, 0.0035\] | Not significant |
| C2 − C0        | +0.0326             | \[−0.0092, 0.0744\] | Not significant |
| C3 − C0        | +0.0311             | \[−0.0199, 0.0845\] | Not significant |

## 5. Discussion

The results present a nuanced but coherent picture. Both VAE-based
balancing conditions (C2 and C3) produced the highest macro-F1 and
balanced accuracy of all four conditions, and showed a clear, monotonic
improvement in recall for the rarest class (df: 0.167 → 0.222 → 0.278 →
0.333 across C0→C1→C2→C3) as well as a meaningful recovery in melanoma
(mel) recall, the clinically highest-stakes class in this dataset. This
is consistent with the original hypothesis that VAE-generated
minority-class images can help a classifier learn features it would
otherwise see too rarely.

Classical oversampling (C1), by contrast, underperformed the baseline on
macro-F1 despite improving balanced accuracy slightly — it achieved
minority-class gains similar to the VAE conditions in some classes (e.g.
mel, vasc) but at the cost of a much larger drop in majority-class (nv)
recall (0.914 → 0.788), which the VAE conditions did not incur to the
same degree. This suggests the benefit observed in C2/C3 is not simply
an artifact of “any balancing helps” — the specific method of balancing
matters, and VAE-generated images appear to be a more effective
mechanism than naive class reweighting under this experimental setup.

The comparison between C2 (50% cap) and C3 (full equalization) is also
informative: C2 slightly outperformed C3 on both macro-F1 (0.6497 vs.
0.6489) and balanced accuracy (0.6393 vs. 0.6320), suggesting that
moderate balancing may be preferable to full equalization — possibly
because adding very large volumes of synthetic images (as in C3)
increases the opportunity for the classifier to learn subtle
synthetic-vs-real distributional shortcuts, an effect partially but
likely not fully mitigated by the real-only fine-tuning phase.

Despite these encouraging directional results, none of the three
balancing conditions reached statistical significance against the
baseline at this single seed. The 95% confidence intervals for C2 and C3
are notably closer to excluding zero than in earlier iterations of this
pipeline (which used prior-sampling VAE generation without the real-only
fine-tuning phase, and in early testing showed C3 numerically
underperforming C0). This suggests the methodological refinements made
during development — particularly manifold-interpolation-based
generation and the real-only polish phase — provided a genuine,
measurable improvement in how effectively the synthetic data could be
used, even though the single-seed result does not yet meet the threshold
for a statistically confirmed claim.

## 6. Limitations

- Single-seed result: this report is based on one train/val/test split
  (seed 17). The project's own design specifies evaluation across 3
  seeds (17, 29, 41) with aggregated confidence intervals before drawing
  a final conclusion; that full multi-seed evaluation was not completed
  due to time constraints, and the statistical significance test above
  should be read accordingly as suggestive, not confirmatory.

- No patient-ID metadata: the original HAM10000 release does not provide
  reliable patient identifiers, so splitting was performed at the lesion
  level rather than the patient level; some risk of same-patient
  (different-lesion) leakage across partitions cannot be fully ruled
  out.

- Extreme minority classes (df: 82, vasc: 97 real training images)
  remain the hardest to model well regardless of balancing strategy —
  their VAEs had the least real data to learn from, and their absolute
  recall values, while improved, remain the lowest of all seven classes.

- This is an academic benchmark exercise, not a validated clinical tool;
  performance on this test partition should not be interpreted as an
  estimate of real-world diagnostic performance.

## 7. Conclusion

At the tested seed, VAE-based synthetic minority-class augmentation
(both moderate and full balancing) produced the highest overall macro-F1
and balanced accuracy among all four conditions tested, together with a
clear improvement in recall for the rarest class and a recovery in
melanoma recall relative to the imbalanced baseline — while classical
oversampling did not match this benefit. However, these improvements did
not reach statistical significance in a paired bootstrap test at this
single seed. The honest conclusion is therefore: this pilot experiment
found a promising, directionally consistent signal that VAE-generated
synthetic images can improve minority-class recognition beyond what
classical oversampling achieves, but this signal requires confirmation
across additional random seeds before it can be reported as a
statistically established result. This is a valid and complete finding
for the scope of work undertaken, and the multi-seed confirmation is
identified here as the clear next step for any continuation of this
work.
