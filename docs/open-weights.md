# Open-weights models in this experiment

This note explains what "open weights" means, how open-weights models are used in this study, and why they matter for the question it asks. It assumes no machine-learning background.

## The short version

- **API models** (Claude, Gemini) can only be studied from the outside: you send a question and read the answer.
- **Open-weights models** (Gemma, running on your laptop) can also be studied from the inside: you can watch what happens in the model while it reads a word, and even change it.
- Phase 1 asks *whether* models link sounds to shapes the way people do. Both kinds of model can answer that.
- Phase 2 asks *how*: whether the link goes through a shared idea of "sharpness", as the ideaesthesia hypothesis predicts. Only open-weights models can answer that.

## First, some basic concepts

### What a "weight" is

A language model is a very large mathematical function. Text goes in, gets turned into numbers, passes through many layers of arithmetic, and comes out as a prediction of the next word.

The arithmetic in each layer is controlled by **weights**: billions of numbers that were set during training. You can think of them like the settings on an enormous mixing desk. Training is the process of nudging every knob until the model's predictions get good. Once training ends, the knobs are fixed, and they *are* the model. "Gemma 3 27B" means a model with about 27 billion of these numbers.

### Open weights vs. API models

- An **API model** keeps its weights on the company's servers. You can talk to it, but you never get the numbers, so you can't look inside.
- An **open-weights model** is published as a file of those numbers. You download it and run it on your own computer. Gemma (Google DeepMind), Llama (Meta) and Qwen (Alibaba) are examples.

"Open weights" isn't quite the same as "open source": you usually get the trained numbers but not the training data or the full training code. For this study the weights are what matter.

### Activations: the model's working state

When a model reads a word, each layer turns it into a long list of numbers that describes the word *in context*. These are **activations**. Weights are fixed; activations change with every input.

A useful analogy: weights are the brain's wiring, and activations are the pattern of brain activity while thinking about something. Neuroscientists put people in brain scanners and look at which regions light up for a word or an image. With an open-weights model, we can record the equivalent for every layer, perfectly and at no cost, which no brain scan can do.

### Representations and "directions"

Each activation is a point in a space with thousands of dimensions. Researchers have found that many concepts correspond to a **direction** in that space. Moving the point one way makes the model treat the input as more of something (more positive, more formal, more French), and moving it the other way makes it less.

If the model has a concept of *roundness vs. sharpness*, we'd expect a direction where "ball", "bubble" and "pillow" sit at one end and "knife", "thorn" and "spike" sit at the other.

## How open weights are used in phase 1 (behavior)

Phase 1 gives every model the same tasks ("Which shape is called *molubo*?") and counts how often its answers match the human pattern. API models and open-weights models are treated the same way here: question in, answer out.

We include an open-weights model in phase 1 for three reasons:

1. **It's the baseline for phase 2.** Phase 2 inspects the inside of Gemma. That's only meaningful if we know Gemma shows the effect from the outside. If Gemma 27B doesn't match sounds to shapes, there's nothing inside to explain. If it does, phase 2 can ask *where* that behavior comes from.
2. **It doesn't change under you.** API models are updated quietly over time, and older versions get retired, so a result from today may not be reproducible next year. A downloaded model file is frozen: anyone can rerun the exact same model and get the same distribution of answers.
3. **It costs nothing per question.** A full phase 1 run is tens of thousands of questions per model. Locally that only costs time.

In phase 1 we run Gemma through **Ollama** in a *quantized* form (see below), which is fine when we only care about the answers.

### Why two sizes (4B and 27B)

- **27B** is the main model. It's much closer in ability to the API models we compare against.
- **4B** is fast, so it's where we debug the phase 2 code. It also gives a size comparison: if the effect is strong at 27B and weak at 4B, that suggests the sound-shape link is something models pick up as they get more capable, which is itself a finding.

## How open weights are used in phase 2 (mechanism)

This is where open weights become necessary. Phase 2 has four steps.

### 1. Find a "roundness" direction (a probe)

Feed the model clearly round words (*ball, bubble, curve, soft pillow*) and clearly sharp ones (*knife, thorn, spike, shard*), and record the activations at every layer. Then train a **probe**: a very simple classifier, essentially a line drawn through the activation space, that separates the round words from the sharp ones.

If a simple line separates them well, the model has an organized internal notion of roundness at that layer.

### 2. Test whether that direction transfers (the key test)

This is the direct test of the ideaesthesia hypothesis. The probe was trained only on *real words with meanings*. Now we show it things it never saw:

- **Made-up words** like *molubo* and *kitezi*. They have no meaning, only sound.
- **Images** of round and jagged shapes, in a model that can see.

If the same probe sorts *molubo* to the round side and *kitezi* to the sharp side, the model is mapping sound onto the same internal concept it uses for meaning. Ideaesthesia predicts exactly that: different senses meeting at a shared high-level idea. If the probe fails on made-up words, the model's phase 1 behavior must come from somewhere else, such as letter shapes or memorized trivia.

Plotting probe accuracy layer by layer for words, made-up words and images shows *where* in the model the senses meet: early layers handle surface features like spelling, and later layers handle more abstract meaning.

### 3. Check that the direction is actually used (steering)

Finding a direction shows the information is *present*. It doesn't show the model *uses* it. To test that, we **steer**: while the model reads a neutral made-up word, we push its activations a little along the roundness direction, then ask the matching question again.

If pushing toward "round" makes the model pick the round shape more often, the direction is part of how the model makes the decision. This is like stimulating a brain region and seeing behavior change, rather than only watching it light up.

### 4. Look for interpretable features (sparse autoencoders)

A single activation mixes many concepts together. A **sparse autoencoder (SAE)** is a separate small model trained to break activations into thousands of individual **features**, each of which tends to mean one recognizable thing ("references to the Golden Gate Bridge", "code with errors"). Anthropic used this method to find features inside Claude that respond to the same concept in different languages and in images.

For Gemma 3, Google DeepMind has published ready-made SAEs (**Gemma Scope 2**) for every layer, with a browser explorer on Neuronpedia. That's the main reason Gemma 3 is the phase 2 model: we can search for a "roundness" or "sharpness" feature right away and check whether it fires on *molubo* as well as on *ball*, instead of spending weeks training SAEs ourselves.

## Why open weights may be important here

**They're the only way to test the hypothesis, not just the effect.** Behavior alone can't distinguish "the model links sounds and shapes through a shared concept" from "the model learned a shortcut". Both can produce the same answers. Telling them apart means looking inside, and that requires the weights.

**They connect the study to how people work.** The human evidence for ideaesthesia includes brain studies (for example, Kovic, Plunkett & Westermann, 2010, measured brain responses to sound-shape matches). An open-weights model allows the same kind of evidence: not just "does it answer like a person", but "is its internal organization similar".

**They make the work checkable.** Anyone can download the same Gemma file and rerun the analysis. With an API model, results depend on a version that may disappear.

**They're the model type interpretability research builds on.** Most published tools for looking inside models (probing libraries, SAEs, Neuronpedia) are built for open-weights models, so results can be compared with other work.

## Limitations to keep in mind

- **Open models aren't frontier models.** Gemma 27B is capable but smaller than Claude or Gemini. Whatever we find inside Gemma suggests, but doesn't prove, how larger models work.
- **Quantization is fine for answers, not for activations.** To fit big models into memory and make them faster, tools like Ollama store weights at lower precision (**quantization**, like saving a photo as a smaller JPEG). The answers barely change, so it's fine for phase 1. But it slightly distorts the activations, which are exactly what phase 2 measures. So phase 2 loads the full-precision weights (about 54 GB for 27B) with PyTorch instead of Ollama.
- **A probe can find patterns the model doesn't use.** That's why the steering step matters.
- **Spelling is still a confound inside the model.** Models read text as chunks of letters, not sounds. A "roundness" signal for *molubo* could come from the letters *o* and *b*. The articulatory-description and katakana renderings from phase 1 carry over to phase 2 to check this. Gemma 4, which can take audio input, could remove the confound completely by letting the model *hear* the words, as people do in the human studies.

## Glossary

| Term | Plain meaning |
|---|---|
| Weights | The fixed numbers learned in training; the model itself |
| Open weights | The weights are published so you can run the model yourself |
| API model | A model you can only use through a company's servers |
| Activations | The model's internal working numbers while it reads a specific input |
| Layer | One stage of processing; models have dozens, stacked in sequence |
| Representation / direction | A pattern in the activations that corresponds to a concept |
| Probe | A simple classifier that tests whether a concept can be read from activations |
| Steering | Nudging activations along a direction to see if behavior changes |
| Sparse autoencoder (SAE) | A tool that splits activations into individual, more interpretable features |
| Feature | One unit found by an SAE that tends to respond to one concept |
| Quantization | Storing weights at lower precision to save memory; slightly blurs activations |
| Parameters (4B, 27B) | The number of weights: 4 billion, 27 billion |
