# VFRW (Visual Font Recognition in the Wild), project handoff

## identity
- **student:** ibrahim, ai @ fast nuces karachi
- **repo:** local, not yet public (git initialized, no remote confirmed in this handoff)
- **local path:** `C:\Ibrahim\Personal\University Stuff\Portfolio Projects\font-identifier`
- **purpose:** cv portfolio project demonstrating a full custom-trained deep learning pipeline, not a wrapper around an existing model or api

## what this project is
given an image, either a clean synthetic render or a real photo, detect the text regions in it,
then classify which font each region is set in. the classifier covers 3,407 distinct
(font family, style role) classes drawn from 1,976 real, freely licensed font families.

## why it exists (the actual thesis)
existing font identification tools (whatthefont and similar sites) are inaccurate, especially
on real world photos rather than clean screenshots. this project's core claim is that a
purpose built, custom trained classifier can beat those tools on real photo test data, and the
evaluation phase is built specifically to prove or disprove that claim with a direct,
apples to apples comparison, not just report an isolated accuracy number.

## use case
a user uploads a photo containing text (a sign, a book cover, packaging, a screenshot) and gets
back a ranked list of likely fonts. the target audience is designers, typographers, and anyone
trying to identify a font from the wild rather than from a clean specimen sheet.

## architecture

```
input image -> [stage a: text detection] -> boxes -> [stage b: crop and normalize]
            -> [stage c: font classifier, custom trained cnn, 3407 classes] -> prediction
```

- **stage a (text detection):** starts with a pretrained detector (easyocr or craft). a custom
  trained detector is planned later as a bounded, separate benchmark experiment, not a
  replacement for the pretrained one.
- **stage b (crop and normalize):** takes detected text boxes and prepares them the same way
  the training crops were prepared (resize, letterbox to a fixed square).
- **stage c (font classifier):** the core deliverable. fully custom trained, primarily on
  synthetic data, with augmentation designed to close the gap between synthetic training data
  and real photo inference data. this is the stage currently being built.

## hardware and constraints
- local gpu: gtx 1650, 4gb vram, 16gb system ram. this card has no tensor cores (turing
  tu117), which shapes both the backbone choice and the batch size / gradient accumulation
  strategy.
- everything in the project must stay free. font sources are free or open licensed only.
  dataset hosting, if it happens, goes through a free tier (hugging face datasets hub or
  kaggle).
- python 3.13, venv at `.venv/`.

## phase by phase status

### phase 0, project setup: done
venv created, folder structure scaffolded, base dependencies installed.

### phase 1, font sourcing: done
- 1,976 font families downloaded from google fonts, recorded in a manifest.
- `data_gen/font_loader.py` resolves each family into its actual usable classes. for variable
  fonts it checks the font's real named instances (regular, bold, italic, bold italic) instead
  of assuming all four exist, and expands each family into however many roles it actually
  supports.
- this expansion produced 3,423 initial (family, role) classes from the 1,976 families, before
  cleanup (see phase 2 audit below).

### phase 2, synthetic dataset generation: done, including a follow up audit and cleanup
- pipeline renders labeled text crops per class: text sampled as a mix of real words,
  phrases, sentences, and random alphanumeric strings, composited onto procedural or stock
  photo backgrounds, then augmented (rotation, perspective warp, blur, noise, brightness and
  contrast jitter, jpeg recompression) and letterboxed to a fixed 224x224 square.
- the generation script is resumable and crash safe. each class is only recorded as complete
  once every one of its images has been written, so killing the process at any point loses at
  most the one class in progress.
- full run produced 150 images per class.
- **data quality audit, done in two passes.** the first pass caught fonts that render without
  throwing an exception but produce meaningless output: emoji and symbol fonts (notoemoji,
  notocoloremoji, notosanssymbols, notosanssymbols2, notoznamennymusicalnotation) render as
  tofu boxes or unrelated glyph shapes, not real characters. the second pass checked every
  non-latin-script font family in the manifest (171 families, mostly noto sans and noto serif
  script variants) by actually rendering english text and inspecting the output. this found
  that the overwhelming majority render clean latin glyphs correctly, since noto's script
  families share a common latin base design, but three families
  (karlatamilinclined, karlatamilupright, phetsarath) render fully blank on latin text and were
  removed. a final full coverage scan (not a sample) confirmed no other classes are silently
  corrupted.
- **final clean dataset: 3,407 classes, 511,050 images, 150 images per class, split 85 percent
  train / 15 percent validation.** class taxonomy keeps family and role separate rather than
  collapsing bold/italic/regular into one family level class, a deliberate decision to preserve
  style information rather than simplify the label space.

### phase 3, font classifier (stage c) training: architecture decided and validated, full run not yet started
architecture and methodology decided and implemented, then genuinely stress tested rather than
assumed correct. the path here was not a straight line and the detour is worth knowing:

- **backbone:** efficientnet-b0, pretrained on imagenet, loaded via timm, head replaced for
  3,407 classes.
- **two phase progressive unfreezing.** phase one trains only the newly added classifier head
  with the backbone fully frozen, which is cheap since no gradient graph needs to be built
  through the backbone at all. phase two unfreezes the last two mbconv blocks of the backbone
  alongside the head, since font discrimination lives in mid to high level shape features
  rather than the generic edge and texture features the early layers already capture well from
  imagenet.
- **optimizer and schedule:** adamw, discriminative learning rates (higher for the head, an
  order of magnitude lower for the unfrozen backbone blocks in phase two), cosine decay with a
  short warmup, label smoothing, gradient clipping, mixed precision throughout.
- **batch size is not guessed.** a dedicated probe script binary searches the real maximum
  batch size that fits in 4gb of vram for each phase, on the actual model, before any real
  training run starts.
- **smoke testing is split into two independent checks**, not one conflated test. check a is a
  short run that only verifies the pipeline works and the batch size actually fits, nothing
  more. check b tests whether the architecture itself is viable by training two class subsets
  in parallel to pre committed accuracy thresholds decided in advance: a hand picked set of
  visually confusable font pairs (same family across styles, plus known lookalike families
  across different sources), and a separate random sample of 500 classes. both subsets have to
  clear their threshold, since a small subset passing alone would not prove the approach scales
  to the full 3,407 classes, and a broad subset passing alone would not prove it can actually
  separate near identical fonts.
- **full monitoring and crash recovery are built in**, not planned for later. every training
  run logs per step loss, learning rate, gradient norm, throughput, and gpu memory, and per
  epoch train and validation loss, top 1 and top 5 accuracy, accuracy on the hard confusable
  subset specifically, and the gap between macro and micro accuracy (a signal that the model is
  collapsing onto easy classes even though the dataset itself is balanced). an automatic drift
  check flags when the train and validation loss gap has been widening for several epochs in a
  row. checkpoints save the full model, optimizer, scaler, and scheduler state every epoch and
  periodically mid epoch, so a run can be resumed exactly where it left off after any
  interruption. a plain sentinel file lets a run be paused cleanly on request instead of being
  killed mid write. any nan or inf loss value triggers an immediate emergency checkpoint and a
  hard stop rather than continuing to train on a corrupted state.
- **known open gap:** the real photo evaluation set (`data/real_test/`) is currently empty.
  check b's convergence threshold is synthetic data only until real photos are collected. the
  plan is to re anchor the viability check against real photo accuracy once that folder has
  content, rather than treat synthetic only convergence as sufficient proof the architecture
  works.
- environment installed and verified: pytorch with cuda support, torchvision, and timm, all
  confirmed working against the actual gpu.

**check b, run for real, failed the first two times, then passed.** a plain softmax head
plateaued at 30 percent top 5 on the confusable subset against the 50 percent threshold, while
the broad 500 class subset passed comfortably. swapping to an arcface head (the standard fix
for this exact failure signature) only moved confusable to 34 percent, still failing. a
confusion matrix check on that result showed the true font family landed in the model's top 5
guesses 69 percent of the time and the true style landed in top 5 89 percent of the time,
individually strong, but getting both right at once in the same 5 guesses was the hard part.
that led to trying a hierarchical model, a separate family head and a separate style head
sharing one backbone. it improved the confusable subset to 41 percent but broke the broad
subset outright, dropping it from a passing 33 percent down to a failing 22 percent, confirmed
by direct testing to be a real structural cost of splitting the label space, not a data
artifact and not fixable by changing how the two heads scores get combined. that architecture
was reverted.

the fix that actually worked: keep the single flat arcface head, and oversample the twenty
confusable classes five times over during training so the model sees more of exactly the
images it struggles with, without changing the model itself. trained on the confusable and
broad subsets together this way, the same model scored 52 percent on confusable and 28 percent
on the broad subset, clearing both thresholds at once for the first time. this result was
checked for the two most likely ways it could be misleading, silent data leakage between the
oversampled training images and the validation images, and a training accuracy number that
looked suspiciously low, and both checks came back clean rather than being assumed fine.

not yet done: wiring hard negative oversampling into the actual full phase one and phase two
training run across all 3,407 classes, not just the smoke test subsets, then running it for
real.

### phase 4, text detection (stage a): not started
planned to start from a pretrained detector (easyocr or craft) rather than building one from
scratch. a custom trained detector is a possible later addition, treated as a separate bounded
experiment rather than a blocking requirement.

### phase 5, evaluation and benchmark: not started
the phase that actually tests the project's thesis. needs a real photo test set (shared with
the stage c real photo gap above), a defined comparison protocol against whatthefont and
similar tools, and metrics beyond simple top 1 accuracy given how many classes are near
duplicates of each other (top 5 accuracy and per class confusion analysis are both planned).

### phase 6, web demo: not started
fastapi backend plus a frontend, following the same general shape as the author's previous
cxr-vision project (upload an image, get predictions back, live inference not mocked).

## folder layout
```
font-identifier/
├── data_gen/          font sourcing and synthetic dataset generation pipeline, done
├── data/               gitignored, fonts, synthetic dataset, real photo test set
├── models/             model definitions, classifier.py holds the efficientnet-b0 build and
                         freeze and unfreeze logic
├── training/           training driver, config, dataset loader, checkpointing, monitoring,
                         smoke test logic. checkpoints and logs gitignored.
├── eval/                evaluation and benchmark scripts, not yet built
├── web_demo/            fastapi backend and frontend, not yet built
├── notebooks/           exploratory work backing the write up, not yet used
```

## run commands so far
```bash
# font sourcing and dataset generation, already run to completion
python data_gen/generate_dataset.py --images-per-class 150

# training pipeline, code complete, not yet executed
python training/probe_batch_size.py       # find real max batch size for this gpu
python training/train.py --check-a         # pipeline and vram sanity, about 40 steps
python training/train.py --check-b         # architecture viability on hard subsets
python training/train.py --phase 1          # full head warmup run
python training/train.py --phase 2 --resume # full fine tune run, resumable
```

## decisions made and why, worth remembering
- **family and role kept as separate classes rather than collapsed**, a deliberate choice to
  preserve style information (regular vs bold vs italic) rather than simplify the label space
  down to family only.
- **efficientnet-b0 over resnet18.** a previous project (cxr-vision, chest x-ray classification)
  used resnet18 successfully on the same gpu, but that was a 2 class, coarse texture problem on
  roughly 27,000 images. this project is a 3,407 class, fine grained, near duplicate heavy
  problem on over 500,000 images, a different regime entirely. efficientnet-b0 has fewer
  parameters than resnet18 while giving better fine grained feature quality per unit of vram,
  which matters more here than raw parameter count.
- **no live mixup or cutmix augmentation.** blending two font crops together has no coherent
  meaning the way it does for natural image classification, it would manufacture nonsense
  labels rather than augment real ones.
- **no extra live rotation, blur, or noise augmentation beyond light crop and color jitter.**
  the dataset generation phase already deliberately built that diversity into the images
  themselves, so duplicating it at training time would be redundant rather than additive.
- **dependencies for pytorch, torchvision, and timm were installed with `--no-deps` and
  explicit version pinning**, after discovering that installing timm normally would silently
  pull in a cpu only build of pytorch from the default package index, overwriting the cuda
  build that had already been installed and verified. this is now the standard way any new
  dependency touching torch gets installed in this project.

## code style
- lowercase comments
- no decorative dividers or long dash sequences
- no artificial line spacing
- comments only where genuinely needed, explaining why a choice was made rather than restating
  what the code already says
