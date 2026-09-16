# Assignment 3: Building a Custom LLM

This repository contains my completed baseline experiment for **Assignment 3: Building a Custom LLM**. I trained the supplied small nanoGPT transformer from scratch on the synthetic classroom corpus. This is a controlled next-token-prediction experiment, not a pretrained chat model.

The grading entry points are the [executed notebook](custom_llm.ipynb), the [loss plot](results/training_curves.svg), the [full loss table](results/training.csv), and the [results artifacts](results/).

## 1. My three choices

| Choice | Value | Reason |
|---|---:|---|
| Corpus | `classroom` | The supplied synthetic corpus is permitted, reproducible, and large enough to show how repeated word-level patterns are learned without introducing private or copyrighted files. |
| Training steps | `3000` | This is the assignment's recommended baseline budget: long enough to observe learning while remaining practical on a CPU. |
| Initial learning rate | `0.001` | The notebook combines this with warmup and cosine decay, allowing useful updates without the instability of an excessively large rate or the slow learning of a very small rate. |

The run used seed 42, a 64-dimensional embedding, 2 transformer layers, 4 attention heads, block size 48, and batch size 32. The final model had 111,872 parameters and trained for 35.78 seconds on a Colab CPU.

## 2. Pre-training prediction

Before training, I expected the randomly initialized model to generate mostly incoherent token sequences and to assign fairly diffuse next-token probabilities. By 1,500 and 3,000 steps, I expected both fixed-panel training and validation loss to decrease, and I expected generated samples to reproduce more of the classroom corpus's sentence templates and domain groupings. For the word **customer**, I expected its 64-number embedding to change and its closest neighbors to move toward related words such as client, buyer, shopper, consumer, or subscriber. Because the corpus is synthetic and repetitive and validation uses the same templates, I did not expect plausible samples or lower validation loss to demonstrate broad language understanding or generalization to new domains.

## 3. Corpus, split, vocabulary, and UNK rates

- Corpus mode: `classroom`; no external files were added.
- Generated passages before deduplication: **6,360**.
- Unique passages after deduplication: **4,632**; **1,728** duplicates removed.
- Training split: **4,168** passages (90%).
- Validation split: **464** passages (10%).
- Vocabulary size: **136** tokens: 133 retained training types plus `<UNK>`, `<BOS>`, and `<EOS>`.
- Training UNK rate: **0.0%**.
- Validation UNK rate: **0.0%**.

The split unit is a deduplicated passage. Validation contains new combinations from the **same sentence templates and domains** as training. It therefore measures held-out fit inside this controlled generator, not generalization to unseen sources, templates, or subject areas. See [corpus manifest](results/corpus_manifest.json), [split](results/split.json), and [vocabulary report](results/vocabulary_report.json).

## 4. Training results

Loss is the fixed-panel mean next-token cross-entropy; lower is better. Both panels fell sharply from their random-initialization values. Between 1,500 and 3,000 steps, training loss increased slightly while validation loss continued to decline slightly, so the later result is essentially a plateau rather than a large additional gain.

| Step | Training loss | Validation loss |
|---:|---:|---:|
| 0 | 4.923803 | 4.924668 |
| 1,500 | 0.692860 | 0.711261 |
| 3,000 | 0.695557 | 0.705652 |

![Training and validation loss](results/training_curves.svg)

The full machine-readable records are [training.csv](results/training.csv) and [history.json](results/history.json).

## 5. Generated samples

The untrained output is a largely incoherent sequence of corpus words:

> pear professor bond doctor course harvest team physician journey checking buyer delivery traffic report the lecturer item offering and system `<UNK>` taste recommended mentioned bus question customer at mortgage nurse in instructor

At 1,500 steps, all four fixed-seed samples follow classroom templates:

> our school has a question about the new educator and lesson .  
> a review of risk helped us understand the different deposit .  
> the report about the nurse explains the health in detail .  
> the consumer compared the merchandise after checking the price .

The 3,000-step samples were identical to the halfway samples under this fixed random seed. That consistency, together with the loss plateau, suggests the model had already learned the dominant templates by halfway. It does **not** establish broad language ability. Full samples: [untrained](results/samples/step_0000.txt), [halfway](results/samples/step_1500.txt), and [final](results/samples/step_3000.txt).

## 6. Token → ID → 64D embedding

The whole-word tokenizer maps `customer` to token ID **28**. That integer indexes row 28 of the learned token-embedding matrix. The row is a 64-number vector supplied to the transformer; it is not itself a word definition.

<details>
<summary>Initial 64D embedding for <code>customer</code></summary>

```text
[-0.057591915130615234,-0.004809952806681395,0.04263188689947128,0.019338956102728844,0.015643136575818062,-0.02882436476647854,0.025609055534005165,0.00005246092769084498,0.02470681630074978,0.0206917654722929,0.0073690167628228664,-0.033089615404605865,-0.05354786291718483,-0.0057429298758506775,-0.024166762828826904,-0.014716118574142456,0.004685705993324518,-0.01045426819473505,-0.008381075225770473,-0.018258560448884964,-0.020133700221776962,0.005098649766296148,-0.010916502214968204,-0.012633351609110832,0.028389625251293182,-0.002631223062053323,-0.004071928560733795,0.013641919940710068,-0.009891724213957787,-0.01671762391924858,0.0019060791237279773,-0.0014535071095451713,0.01602652110159397,-0.005674874875694513,-0.0006723481928929687,-0.001290727173909545,-0.0073194727301597595,-0.0009307070868089795,0.001507619977928698,-0.004976638592779636,-0.028987020254135132,0.018092988058924675,-0.007348013576120138,-0.005440254230052233,0.01564120501279831,-0.004543505609035492,0.04156793653964996,0.052355434745550156,0.02264268510043621,-0.015414278954267502,-0.025121202692389488,-0.006797463167458773,0.02935275062918663,-0.0025336795952171087,0.029801232740283012,-0.022797005251049995,-0.030237913131713867,0.006436783354729414,0.050490811467170715,0.007490998134016991,-0.01072286069393158,0.02473733201622963,-0.014468724839389324,0.013235910795629025]
```

</details>

<details>
<summary>Final 64D embedding for <code>customer</code></summary>

```text
[0.03709550201892853,-0.004770858678966761,0.1357932835817337,0.12259624153375626,0.07157423347234726,0.031156593933701515,0.1410778909921646,0.09497924894094467,-0.07123769074678421,-0.011708877049386501,0.011737842112779617,-0.059505946934223175,-0.039963267743587494,-0.08282631635665894,-0.14553362131118774,-0.030693694949150085,-0.16128674149513245,-0.1482156366109848,0.0015606125816702843,-0.066569484770298,-0.07938406616449356,0.010203034617006779,-0.07635767012834549,-0.008454341441392899,0.023043528199195862,-0.06297734379768372,0.11937720328569412,-0.03988614305853844,0.03164634108543396,-0.15456311404705048,-0.0869724228978157,0.07473238557577133,-0.04121365398168564,0.13912783563137054,0.08229125291109085,0.07174324989318848,0.01575295254588127,-0.15111741423606873,0.09685888141393661,-0.03868136554956436,0.06345273554325104,-0.016498789191246033,-0.13917434215545654,0.03919057175517082,-0.02606603503227234,-0.10754356533288956,0.01751401275396347,0.049244318157434464,-0.03224155679345131,-0.1633545309305191,0.029620906338095665,-0.12926983833312988,0.005758319515734911,0.09184397011995316,0.11319208890199661,0.08813076466321945,0.10771920531988144,0.019909657537937164,0.06437268108129501,0.09442995488643646,0.12516166269779205,-0.010162289254367352,0.0921124666929245,-0.0879240408539772]
```

</details>

Using cosine similarity over all 64 dimensions, the nearest neighbors changed as follows:

| Stage | Five nearest tokens to `customer` |
|---|---|
| Before | `bus` (0.2133), `educator` (0.2033), `helped` (0.2022), `bank` (0.2005), `risk` (0.1975) |
| After | `client` (0.9847), `buyer` (0.9808), `subscriber` (0.9797), `consumer` (0.9790), `shopper` (0.9780) |

The learned neighborhood matches the shared contexts in the synthetic corpus. A 3D PCA view is only a lossy projection of the 64D geometry, so apparent distances in the viewer can differ from full-dimensional cosine similarity. The complete vectors are in [inspection.json](results/inspection.json) and [checkpoint.json](results/checkpoint.json); they can be explored with [embedding-viewer.html](embedding-viewer.html).

## 7. Next-token probabilities

For the prefix `the customer`, the top five next-token probabilities changed from a diffuse random distribution to the verbs used by the corpus templates:

| Rank | Before training | Probability | After training | Probability |
|---:|---|---:|---|---:|
| 1 | `customer` | 0.0160 | `ordered` | 0.1927 |
| 2 | `bus` | 0.0107 | `reviewed` | 0.1921 |
| 3 | `educator` | 0.0104 | `recommended` | 0.1771 |
| 4 | `us` | 0.0103 | `selected` | 0.1722 |
| 5 | `application` | 0.0101 | `compared` | 0.1357 |

Training examples produce a next-token prediction; cross-entropy measures how much probability the model assigned to the actual next token. Backpropagation then computes gradients that indicate how each parameter should change to reduce that loss.

## 8. A real gradient and parameter update

The notebook captured the first optimizer step for coordinate 0 of `customer`'s embedding:

- Before: **-0.057591915130615234**
- Gradient: **-0.0004056147299706936**
- Step-1 scheduled learning rate: **0.00001**
- After AdamW update: **-0.057581909000873566**
- Observed change: **+0.0000100061**

The negative gradient points toward increasing this coordinate to lower the current batch loss, which matches the positive observed change. The exact update is not simply `-learning_rate × gradient`: AdamW also uses momentum, adaptive scaling, gradient clipping, and weight decay.

## 9. Attention snapshot

For `<BOS> the customer`, head 1 in block 1 produced these causal attention rows after training:

```text
[[1.0000, 0.0000, 0.0000],
 [0.5855, 0.4145, 0.0000],
 [0.5193, 0.4368, 0.0440]]
```

Future tokens are masked, so each row can attend only to itself and earlier positions. This is one head in one layer for one input; it is not a universal explanation of the model.

## 10. Temperature comparison

All samples below use the trained model and the notebook's fixed seed.

| Temperature | Observed samples |
|---:|---|
| 0.3 | `our school has a question about the new educator and lesson .`<br>`a review of risk helped us understand the different deposit .`<br>`the new peach was mentioned in the juice report yesterday .`<br>`the local consumer was mentioned in the purchase report yesterday .` |
| 0.8 | `our school has a question about the new educator and lesson .`<br>`a review of risk helped us understand the different deposit .`<br>`the report about the nurse explains the health in detail .`<br>`the consumer compared the merchandise after checking the price .` |
| 1.2 | `our school has a question about the new educator and lesson .`<br>`a review of risk helped us understand the different deposit .`<br>`the report about the nurse explains the health in detail .`<br>`the consumer compared the merchandise after checking the price .` |

Lower temperature sharpens the distribution and generally favors more likely continuations; higher temperature flattens it and normally allows more variety. In this particular four-sample, fixed-seed run, 0.8 and 1.2 happened to produce the same text, while 0.3 changed two samples. That small observation should not be generalized beyond this run. Raw data: [temperature_comparison.json](results/temperature_comparison.json).

## 11. Limitation and next experiment

**Limitation.** The synthetic corpus is narrow and highly repetitive. The random held-out split reuses the same domains and sentence templates, so a low validation loss can result from learning those patterns. The near-identical halfway/final samples and high within-category embedding similarities are convincing evidence of template learning, but not of factual knowledge, open-domain language ability, or robust transfer.

**Next experiment.** I would keep the 3,000-step budget and 0.001 initial learning rate fixed, but hold out an entire sentence-template family (or one complete domain) for validation. Comparing that loss and the generated text with this baseline would test whether the model can generalize beyond combinations of patterns it already saw, rather than merely extending training time.

## 12. How to run

### Google Colab

1. Open [custom_llm.ipynb](custom_llm.ipynb) in Colab.
2. Confirm `CORPUS = "classroom"`, `TRAINING_STEPS = 3000`, and `LEARNING_RATE = 0.001` in the first code cell.
3. Choose **Runtime → Run all**.
4. Download the newly generated ZIP and the executed notebook from Colab.

### Local Python/Jupyter

1. Use Python 3.10+ and install `pip install -r requirements.txt`.
2. Open `custom_llm.ipynb` in Jupyter and run all cells from top to bottom.
3. The notebook creates a timestamped folder under `llm_runs/` plus a ZIP archive.

The saved run used Python 3.13.15 and PyTorch 2.11.0+cpu. Results are deterministic for the fixed software stack and seed, although exact floating-point values can vary slightly across PyTorch versions or hardware.

## 13. Artifact index

- [Executed notebook](custom_llm.ipynb)
- [Run configuration](results/config.json)
- [Training summary](results/training_summary.json)
- [Full loss table](results/training.csv) and [loss plot](results/training_curves.svg)
- [Untrained, halfway, and final samples](results/samples/)
- [Tokenization example](results/tokenization.json) and [vocabulary report](results/vocabulary_report.json)
- [Embedding, probability, update, and attention inspection](results/inspection.json)
- [Full initial/final embedding checkpoint](results/checkpoint.json)
- [Temperature comparison](results/temperature_comparison.json)
- [Corpus manifest](results/corpus_manifest.json), [corpus text](results/corpus.txt), and [exact split](results/split.json)
- [Trained PyTorch model](results/model.pt)
- [Embedding viewer](embedding-viewer.html)

The model code is pinned to nanoGPT commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`. Its MIT license is included in [NANOGPT_LICENSE](NANOGPT_LICENSE).
