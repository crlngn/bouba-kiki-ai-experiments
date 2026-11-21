# Do AI models share human shape–sound correspondences?

Most people match a round-sounding made-up word like *maluma* to a round blob and a sharp-sounding one like *takete* to a spiky star: the bouba/kiki effect. This repo tests whether language and vision-language models make the same matches, and whether they still do once spelling cues and memorized examples are removed.

The framing comes from **ideaesthesia** (Jürgens & Nikolić, 2012), which my MIT Media Lab thesis *[PerForm](https://hdl.handle.net/1721.1/124080)* built on: the senses connect through shared high-level concepts, not directly to each other. A sharp sound and a jagged shape feel related because both reach the idea of *sharpness*. If a model links them through a shared concept, the same internal "sharpness" representation should respond to *takete*, to a spiky star and to the word *knife*. Phase 1 (this code) tests the behavior. Phase 2 will look for that shared representation inside open-weights models.

> **Status:** phase 1 pipeline built and tested with a mock model. No real model results yet.

## Method (phase 1)

### Stimuli: `src/make_stimuli.py`

- **Fresh pseudowords** in a 2×2 design, 50 per cell, generated from CV syllables with a fixed seed (recorded in `stimuli/meta.json`):

  | | Round vowels (o, u) | Sharp vowels (i, e) |
  |---|---|---|
  | **Round consonants** (m, b, l, n, w) | *mobolu* (round) | *meli* (mixed) |
  | **Sharp consonants** (k, t, p, z, s) | *kotuzo* (mixed) | *kitezi* (sharp) |

  The mixed cells show whether a model weights consonants or vowels more.
- **Famous words** (*bouba/kiki*, *takete/maluma*) as a separate condition. The gap between famous and fresh words estimates how much of the effect is memorized.
- **Five renderings per word**, to separate sound from letter shapes (round letters like *o* and *b* look round):
  - `spelled`: *mobolu*
  - `ipa`: /mo.ˈbo.lu/
  - `sounds_like`: "moh-boh-loo"
  - `articulatory`: a description of each sound with no letters at all ("a voiced nasal made with both lips followed by a mid back vowel; …"). Descriptions avoid words like *rounded* or *spread* that would give the answer away.
  - `katakana`: モボル, a script whose letter shapes don't follow Latin roundness

  For this phoneme set, IPA is almost identical to the spelling, so it barely controls for letter shape. The articulatory and katakana renderings are the real controls.
- **Shapes:** radial frequency patterns after Chen, Huang, Woods & Spence (2016), a 5×5×5 grid over frequency (3–11 bumps), amplitude (0.05–0.25) and spikiness (1–16), all scaled to the same area. Three round/jagged pairs from the grid are the anchors for the two-choice image task.

### Tasks: `src/tasks.py`

| Task | Question | Scored as |
|---|---|---|
| `match` | Which of two shapes (text descriptions or images) is called *X*? | round word → round shape |
| `rating` | 1–7, how round or sharp does *X* sound? (scale direction counterbalanced) | mean sharpness |
| `crossmodal` | Is *X* the name of calm or excited, vanilla or lemon, sweet or sour, a soft hum or a harsh beep? | round word → calm, vanilla, sweet, hum |
| `classic` | Pitch and loudness with brightness and size (Marks, 1974; Mondloch, 2004) | % congruent |
| `graded` | One grid shape: is it called *[round word]* or *[sharp word]*? In English and Mandarin | logistic weights for frequency, amplitude and spikiness |
| `screen` | Does *X* already mean something? | words to drop |

Answer order is swapped on alternate trials. The system prompt is neutral and never mentions bouba/kiki or sound symbolism. Each trial is sampled 10 times at temperature 1.

### Models: `src/models.py`

`provider:model_id`, with `anthropic`, `gemini`, `ollama` (local open-weights) and `mock` (random answers, for testing). Every sample is logged to `results/raw/*.jsonl` with model, prompt, raw output, parsed answer and timestamp.

### Analysis: `src/analyze.py`

Samples of the same word aren't independent, so 95% confidence intervals are bootstrapped over words, and the binomial test against 50% counts words whose majority answer matches the human pattern. The human baseline line (95%) is the low end of the 95–98% reported by Ramachandran & Hubbard (2001).

## Running it

```bash
uv sync
cp .env.example .env   # add ANTHROPIC_API_KEY and GEMINI_API_KEY
uv run python src/make_stimuli.py

# 1. Screen the fresh words, then set keep=0 in stimuli/pseudowords.csv for any flagged word,
#    and for any word a web search shows is a real word, name, brand or place.
uv run python src/run_behavioral.py --model gemini:<model id> --tasks screen --samples 1 --temperature 0

# 2. Behavioral runs (resumable; rerun the same command after an interruption)
uv run python src/run_behavioral.py --model ollama:gemma3:27b --tasks match,rating,crossmodal,classic
uv run python src/run_behavioral.py --model ollama:gemma3:27b --tasks match,graded --shape-mode image

# 3. Tables and chart
uv run python src/analyze.py
```

Use `--dry-run` to see trial counts and a sample prompt, and `--limit N` for a quick subset. The full text-mode run is about 2,000 match trials × 10 samples per model.

## Limitations

- The stimulus set is small and the phoneme classes are coarse.
- Results may be sensitive to prompt wording.
- Famous words, and possibly some fresh ones, may be in the training data.
- Text-only models see shapes as verbal descriptions, which already contain words like *rounded* and *spiky*.

## Key references

- Jürgens & Nikolić (2012), Ideaesthesia, *Translational Neuroscience* 3(1).
- Ramachandran & Hubbard (2001), Synaesthesia, *Journal of Consciousness Studies*.
- Chen, Huang, Woods & Spence (2016), When "Bouba" equals "Kiki", *Scientific Reports* 6, 26681.
- Ćwiek et al. (2022), The bouba/kiki effect is robust across cultures and writing systems, *Phil. Trans. R. Soc. B*.
- Maurer, Pathman & Mondloch (2006), The shape of boubas, *Developmental Science* 9(3).
- Marks (1974), *American Journal of Psychology* 87(1/2); Mondloch (2004), *Cognitive, Affective, & Behavioral Neuroscience* 4(2).
- Alper & Averbuch-Elor (2023), Kiki or Bouba? Sound Symbolism in Vision-and-Language Models, NeurIPS.
- Rozendo Xavier dos Santos (2019), *PerForm*, MIT. https://hdl.handle.net/1721.1/124080
