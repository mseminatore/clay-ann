# clay-ann

Neural networks for [Clay](https://github.com/mseminatore/clay), built on
[libann](https://github.com/mseminatore/ann), a fast, portable neural network
library in plain C. This is a **native brick**: its C sources are compiled the first time
you import it, then loaded into Clay, or linked into executables built with `clay --compile`.

```clay
import("ann")

net = Ann network(#{"learningRate": 0.5, "epochs": 5000, "quiet": true})
net input(2)
net dense(4, "sigmoid")
net output(1, "sigmoid")

net train([[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]])
net predict([1, 0])        // [0.97...]
```

## Installation

```sh
clay bricks add github.com/mseminatore/clay-ann ann
```

The first `import("ann")` compiles the brick with your Clay build's C compiler, which takes a few
seconds. The result is kept in the brick's `build/` directory and rebuilt only when the sources,
the compiler or the Clay version change. It needs a Clay with native bricks (plugin API version 2)
and works on macOS, Linux and Windows.

## Networks

`Ann network(options)` makes an empty network. All options are optional:

| Option | Values |
|--------|--------|
| `optimizer` | `"adam"` (default), `"sgd"`, `"momentum"`, `"rmsprop"`, `"adagrad"` |
| `loss` | `"mse"` (default), `"cross_entropy"` |
| `learningRate` | a number |
| `epochs` | the most epochs to train for |
| `batchSize` | samples per weight update |
| `convergence` | stop training once the loss is below this |
| `quiet` | `true` hides libann's training progress |

Then add layers, from input to output:

| Method | |
|--------|---|
| `input(n)` | the input layer: `n` values per sample |
| `dense(n, activation)` | a fully connected hidden layer |
| `output(n, activation)` | the output layer |

Activations: `"sigmoid"` (default), `"relu"`, `"leaky_relu"`, `"tanh"`, `"softsign"`, `"softmax"`,
`"gelu"`, `"none"`.

## Image and sequence layers

`input(channels, height, width)` starts a network for images: each sample is a row of
`channels * height * width` numbers, channel by channel, row by row, so MNIST's 784-column rows
are `input(1, 28, 28)`. Then add layers, each returning the network:

| Method | Layer |
|--------|-------|
| `conv2d(filters, kernel, activation, stride, padding, groups)` | 2-D convolution. `kernel`, `stride` and `padding` are a number or `[height, width]`; `padding` can also be `"same"` (keeps the size at stride 1) or `"valid"` (none, the default). Defaults: activation `"none"`, stride 1, groups 1 |
| `depthwiseConv2d(multiplier, kernel, activation, stride, padding)` | each input channel gets its own filters (follow it with `conv2d(n, 1)` to mix channels) |
| `conv1d(filters, kernel, activation, stride, padding)` | convolution along a sequence: `input(features, 1, length)`, or an embedding's output |
| `maxPool2d(size, stride)` | max pooling; `size` defaults to 2, `stride` to `size`; a number or `[height, width]` |
| `maxPool1d(size, stride)` | max pooling along a sequence |
| `batchNorm(activation)` | normalizes each channel over the batch, then applies the activation: `conv2d(16, 3)` then `batchNorm("relu")` |
| `flatten` | `(C, H, W)` to a flat vector (a `dense` layer after a convolution reads it flat anyway) |
| `globalAvgPool`, `globalMaxPool` | each channel's mean or largest value: `(C, 1, 1)` |
| `upsample2d(scale)`, `reshape(channels, height, width)` | for decoders and autoencoders |
| `embedding(vocabulary, dimension, paddingId, masked)` | token ids to learned vectors; with `masked` true, steps holding `paddingId` are left out of the layers above |

```clay
net = Ann network(#{"loss": "cross_entropy", "learningRate": 0.001, "quiet": true})
net input(1, 28, 28)
net conv2d(8, 3, "relu")
net maxPool2d(2)
net conv2d(16, 3, "relu")
net maxPool2d(2)
net flatten
net dense(64, "relu")
net output(10, "softmax")
net fit(x, y, #{"epochs": 3, "batchSize": 64})      // 98% on MNIST in a few seconds
net summary println
```

`net layers` describes the layers, one Dict per layer with `kind`, `shape` (`[channels, height,
width]`; dense layers are `[nodes, 1, 1]`), `activation` and `params`. `net summary` is the same as
a table of text with the total number of parameters:

```
#   layer             output shape    activation  params
0   input             1x28x28         none        0
1   conv2d            8x26x26         relu        80
...
total parameters: 27562
```

`net layerCount` is the number of layers, the input layer (0) included, and `net weights(i)` and
`net biases(i)` read layer `i`'s parameters, or give `nil` for a layer without them:

| Layer | `weights(i)` | `biases(i)` |
|-------|--------------|-------------|
| dense | a row per node, one value per node below | one per node |
| conv2d, conv1d | a row per filter: channel by channel, then the kernel row by row | one per filter |
| embedding | a row per token id | `nil` |
| batchNorm | one row: the scale (gamma) of each channel | the shift (beta) of each channel |

Before any training they are the values training would start from. They are copies: changing
the Arrays doesn't change the network.

`net setWeights(i, values)` and `net setBiases(i, values)` change them. `values` is in the shape
`weights` and `biases` give it (rows, or one Array), a flat Array of every value in order, or an
AnnData table, so pre-trained vectors can come from a CSV file. Training starts from the values
instead of random ones. A wrong shape raises `ArgumentError`, and nothing changes.

`net freeze(i)` keeps layer `i`'s weights and biases as they are while the rest of the network
trains (gradients still pass through it to the layers below), and `net unfreeze(i)` lets it train
again. Both return the network. Saves keep frozen layers frozen. Together they fine-tune a
trained network, or keep pre-trained embedding vectors fixed (see
[Pre-trained word vectors](#pre-trained-word-vectors) for vectors from a GloVe file):

```clay norun
vectors = Ann loadCsv("glove_50d.csv")         // one row of 50 numbers per token id
net = Ann network(#{"loss": "cross_entropy"}) input(20) embedding(vectors rows, 50) flatten
net output(2, "softmax")
net setWeights(1, vectors) freeze(1)           // learn only the layers above
net train(x, y)
```

`net layers` gives each layer's `trainable` (false once frozen) and `weightFormat` (below), and
`net summary` marks frozen and low-bit layers.

`net activations(inputs, layer)` is what a layer outputs for every row, after its activation:
an AnnData for an AnnData, else an Array of rows. `layer` counts as in `net layers`, or from the
end when negative (`-1` is the output layer, `-2` the one below it). A conv layer's row is its
feature maps one after another, each row by row (`layers` gives the shape). The forward pass
stops at the layer, so it is quicker than `predictBatch`, and nothing is kept from the hidden
layers when you don't ask:

```clay norun
codes = net activations(x, -2)                 // the last hidden layer, e.g. for clay-plot's embedding
maps = net activations([image], 1)             // one image's first conv feature maps
```

Everything else works on these networks as before: `train`, `fit`, `predictBatch`, `save`,
`quantize` and the exports. Mistakes (a convolution with no input layer, a window bigger than its
input, a `reshape` that changes the size) raise `ArgumentError`.

## Text: `Ann tokenizer`

An embedding reads token ids. `Ann tokenizer` makes them from texts: it builds a vocabulary
(the most frequent words first) and turns each text into ids, and `encodePadded` makes them
rows of one length, as an AnnData ready for `train` or `fit`. Id 0 is left for padding, so the
embedding can mask it; with an `oovToken`, id 1 stands for words outside the vocabulary.

```clay norun
tok = Ann tokenizer(#{"numWords": 10000, "oovToken": "<unk>", "preprocess": "stripHtml"})
tok fit(trainTexts)                              // an Array of Strings
x = tok encodePadded(trainTexts, 200)            // an AnnData: 200 ids per text, 0-padded at the end

net = Ann network(#{"loss": "cross_entropy", "quiet": true})
net input(200) embedding(tok vocabSize, 64, 0, true)
net conv1d(64, 5, "relu") globalMaxPool output(2, "softmax")
net fit(x, y, #{"epochs": 5})

tok save("tokenizer.json")                       // keep it with the network: new texts need the same ids
```

It is a port of [pyann](https://github.com/mseminatore/pyann)'s `Tokenizer`, written in C:
the same settings give the same ids, and the saved JSON is pyann's format, so a tokenizer
saved by either loads in the other. It is fast. Fitting and encoding 50,000 IMDB-sized reviews
(11 million words, with `stripHtml`) takes under a second.

`Ann tokenizer(options)` options (all optional):

| option | default | |
|---|---|---|
| `numWords` | `nil` | keep only the ids below this (the most frequent words); the rest become the oov id |
| `oovToken` | `nil` | the word standing for unknown ones, as id 1; without one they are dropped |
| `lower` | `true` | lower-case texts first |
| `filters` | ASCII punctuation but `'`, tab, newline | characters turned into spaces before splitting |
| `split` | `nil` | the word separator; `nil` splits on any whitespace |
| `charLevel` | `false` | make each character a token |
| `preprocess` | `nil` | `"stripHtml"`, `"splitContractions"`, a Block of your own (a String in, a String out), or an Array of them, applied in order |

| method | |
|---|---|
| `tok fit(texts)` | count the words of texts and (re)build the vocabulary from all counts so far |
| `tok encode(texts)` | each text as an Array of ids |
| `tok encodePadded(texts, maxlen, padding, truncating)` | an AnnData, a row of `maxlen` ids per text (`maxlen` defaults to the longest) |
| `tok tokenize(text)` | a text's tokens (Strings), before they become ids |
| `tok decode(sequences)` | ids (Arrays, or an AnnData) back to texts, padding left out |
| `tok vocabSize` | the ids the rows can hold, padding included: the embedding's vocabulary |
| `tok wordIndex`, `tok wordCounts` | Dicts of word to id (every word seen) and word to count |
| `tok word(id)`, `tok id(word)` | one word's id (looked up as given), or an id's word |
| `tok toJson`, `tok save(path)` | the settings and vocabulary as JSON; `Ann loadTokenizer(path)` and `Ann tokenizerFromJson(json)` read it |

`Ann padSequences(sequences, maxlen, padding, truncating, value)` pads Arrays of ids you made
yourself the same way. `padding` and `truncating` are `"post"` (the default) or `"pre"`.
Padding goes at the end by default, unlike Keras: with a masked embedding, a text padded at the
end gives exactly what it would unpadded, even through strided layers.

Scraped text needs cleaning. `Ann stripHtml(text)` turns tags into spaces, removes comments,
scripts and styles, and decodes entities (`&amp;` to `&`). Without it, `<br />` becomes the word
"br". `Ann splitContractions(text)` splits contractions the way GloVe's and word2vec's training
text was split (`don't` to `do n't`, `it's` to `it 's`). Both work as `preprocess` steps by name.
A tokenizer whose `preprocess` includes Blocks saves `"custom"`; give the Blocks to
`loadTokenizer(path, preprocess)` again.

Texts are UTF-8. Character tokens, filters and whitespace work on Unicode characters, as
Python's do. Lower-casing covers ASCII, Latin-1, Latin Extended-A, Greek and Cyrillic letters
(Python's covers every script). `stripHtml` decodes numeric entities and HTML 4's named ones,
not all 2,231 of HTML5's.

### Pre-trained word vectors

`Ann loadWordVectors(path, tok, options)` reads GloVe's text format, or word2vec's and fastText's
(with their `count dim` first line), and gives a Dict: `vectors`, an AnnData whose row *i* is the
vector of the word with id *i*, and `found`, how many of the vocabulary's words had one. Only the
vocabulary's words are kept, so a large file needs little memory. Gzipped files aren't read:
gunzip them first.

```clay norun
v = Ann loadWordVectors("glove.6B.100d.txt", tok)
("found vectors for " + (v get("found")) toString + " words") println
net = Ann network(#{"loss": "cross_entropy"}) input(200) embedding(tok vocabSize, 100, 0, true)
net conv1d(64, 5, "relu") globalMaxPool output(2, "softmax")
net setWeights(1, v get("vectors")) freeze(1)    // learn only the layers above
```

Options: `vocabSize` (the rows; default `tok vocabSize`), `dim` (checked against the file),
`missing` (`"zeros"`, the default, or `"random"`, normal with the found vectors' mean and spread,
so the words can train apart in an unfrozen embedding), `seed`, and `paddingId` (the row kept
zero, default 0). `tok` can also be a Dict of word to id. GloVe's words are lower case and its
text had contractions split, so `preprocess: ["stripHtml", "splitContractions"]` finds more of
them.

## Training controls

`Ann network` takes these options too, and each is also a method that returns the network, so
they chain (`net dropout(0.2) weightDecay(0.0001)`):

| Option / method | |
|--------|---|
| `weightDecay`, `l1` | L2 and L1 regularization strengths |
| `dropout` | a rate for every hidden layer; `net dropout(layer, rate)` sets one layer |
| `gradientClip` | clip gradients to this size |
| `weightInit` | `"auto"` (default), `"uniform"`, `"xavier"`, `"he"` |
| `lossReduction` | `"sum"` or `"mean"` over the batch |
| `inputNoise` | `["gaussian", 0.05]`, `["masking", 0.1]` or `["salt_pepper", 0.1, 0, 1]`: noise added to each training input |
| `lrSchedule` | `["step", stepSize, gamma]`, `["exponential", gamma]`, `["cosine", tMax, minLr]` or `["none"]` |
| `validation` | `[inputs, targets]`, held out: its loss is reported each epoch |
| `earlyStopping` | patience in epochs, or `[patience, minDelta, restoreBest]`: stop when the loss (the validation loss, if there is one) stops improving |

After training with `train` or `fit`: `net epochsTrained` (in the last run), `net bestEpoch` (with
early stopping), `net history` (a row `[loss, learningRate, validationLoss]` per epoch, with -1 when
there was no validation data; it accumulates across runs), `net clearHistory`, and
`net learningCurve(path)` to write the history as a CSV file.

```clay
net = Ann network(#{"loss": "cross_entropy", "epochs": 50, "dropout": 0.2,
                    "validation": [xTest, yTest], "earlyStopping": 5, "quiet": true})
```

## Training and using a network

Data are Arrays of rows: one row of numbers per sample.

| Method | Returns |
|--------|---------|
| `train(inputs, targets)` | the final loss |
| `predict(row)` | the outputs, an Array |
| `classify(row)` | the index of the largest output |
| `predictBatch(inputs)` | the outputs of every row at once: an AnnData for an AnnData, else an Array of rows |
| `classifyBatch(inputs)` | the class of every row: a one-column AnnData for an AnnData, else an Array |
| `confusionMatrix(inputs, targets)` | counts as rows of `matrix[actual][predicted]`; targets are one-hot rows or a column of class numbers |
| `accuracy(inputs, targets)` | the fraction classified correctly, 0 to 1 |
| `evaluate(inputs, targets)` | the loss over the data |
| `sizes` | the layer sizes, e.g. `[2, 4, 1]` |
| `save(path)` / `Ann load(path)` | write a network to a file / read one back |

`Ann seed(n)` makes weight initialization, and so training, repeatable. `Ann oneHot(labels,
classes)` turns class numbers into target rows: `Ann oneHot([2, 0], 3)` is
`[[0, 0, 1], [1, 0, 0]]`.

Mistakes such as the wrong number of inputs, ragged rows or an unknown activation raise a Clay
`ArgumentError`. Numbers are 32-bit floats, as everywhere in Clay and libann.

## Saving, exporting and quantizing

| Method | |
|--------|---|
| `save(path)` / `Ann load(path)` | libann's text format (readable) |
| `saveBinary(path)` / `Ann loadBinary(path)` | the compact binary format: smaller and faster for big networks |
| `exportOnnx(path)` / `Ann importOnnx(path)` | ONNX JSON, to move a dense network to or from other tools |
| `exportC(path, prefix)` | the float32 network as a C header for embedded targets; `prefix` (default `"model"`) names its functions, e.g. `model_predict` |
| `exportJs(path)` | the float32 network as a JavaScript module for the browser or Node |
| `diagram(path)` | the layers as a [pikchr](https://pikchr.org) diagram |
| `quantize(inputs)` | an int8 copy of the network (an `AnnQuant`), its value ranges found by running `inputs` (a few hundred training rows are enough) |
| `weightFormat(format)`, `weightFormat(layer, format)` | store dense and conv weights in `"int4"`, `"ternary"` or `"binary"` (4, 2 or 1 bits each), or `"float"` again; without a layer, every dense and conv layer but the output layer. Returns the network |

An `AnnQuant` has `predict(row)`, `classify(row)`, `accuracy(inputs, targets)`, `ioParams`
(`[inputScale, inputZeroPoint, outputScale, outputZeroPoint]`) and `exportC(path, prefix)`, which
writes the integer-only version for targets without an FPU:

```clay
q = net quantize(x slice(0, 500))
Console println("float ${net accuracy(xTest, yTest)}  int8 ${q accuracy(xTest, yTest)}")
q exportC("model_q.h", "model")
```

Low-bit weights make exported models much smaller. Once a layer has a format, it computes with
its weights rounded to it: `predict`, `accuracy` and the exports all use them, `quantize` packs
them in its int8 model, and training is quantization-aware (it updates float weights underneath
the rounded ones, which `weights` reads). The usual recipe is to train in float, set the formats,
then train a few more epochs so the network adjusts to them:

```clay norun
net train(x, y)
net weightFormat("ternary")                    // 2 bits per weight, the output layer stays float
net fit(x, y, #{"epochs": 5})                  // fine-tune with the rounded weights
q = net quantize(x slice(0, 500))
q exportC("model_q.h", "model")
```

Low-bit layers train on the CPU. Saves keep the formats, but older versions of libann can't load
such files, and `exportOnnx` writes the rounded weights as floats.

An exported header or module needs libann's small runtime beside it: copy `ann_rt.h` (for C) or
`ann_rt.js` (for JavaScript) from the brick's `native/libann/` directory. Both exports use the
same weights as `predict`, so the outputs match. The C header's `model_predict(input, output,
workspace)` takes a `float workspace[MODEL_WORKSPACE_SIZE]`.

## Training from Clay: `fit` and the step methods

`train` runs all the epochs inside libann. `fit` runs them one at a time from Clay, so a block can
watch the training or stop it:

```clay
history = net fit(x, y, #{
    "epochs": 30, "batchSize": 64,
    "validation": [xTest, yTest], "patience": 3,
    "lrSchedule": ["cosine", 30, 0.0001],
    "onEpoch": { |e|
        epoch = e get("epoch")
        loss = e get("loss")
        val = e get("valLoss")
        Console println("epoch ${epoch}: loss ${loss} validation ${val}")
    }
})
```

(Compute the values first: a double quote inside `"${...}"` ends the string in Clay.)

| Option | |
|--------|---|
| `epochs` | passes over the data (default 10) |
| `batchSize` | rows per weight update (default: the network's) |
| `shuffle` | `false` keeps the rows in order; by default each epoch uses a new random order |
| `validation` | `[inputs, targets]` held out; their loss is measured after each epoch |
| `lrSchedule` | the same forms as the `network` option, applied per epoch |
| `patience`, `minDelta` | stop after `patience` epochs without the loss (the validation loss, if there is one) improving by more than `minDelta` |
| `restoreBest` | finish with the weights of the epoch whose loss (the validation loss, if there is one) was lowest, not the last epoch's. Default: `true` with `patience`, as for `earlyStopping`; otherwise `false` |
| `onEpoch` | a block called after each epoch with a Dict of `epoch`, `loss`, `valLoss` (-1 without validation data) and `lr`; returning `false` stops the training |

`fit` returns the history, one such Dict per epoch, and puts the learning rate back as it was. Like
`train`, it also adds a row per epoch to `net history` (and so to `net learningCurve`), and sets
`net epochsTrained` and `net bestEpoch`: the epoch whose weights `restoreBest` kept, else the best
one `patience` saw, else 0.

The pieces are available directly, for custom batching, streaming data or several networks in
step:

| Method | |
|--------|---|
| `trainEpoch(inputs, targets, batchSize, shuffle)` | one pass in mini-batches; returns the mean loss |
| `trainBegin`, `trainStep(inputs, targets)`, `trainEnd` | an online session: `trainStep` updates the weights on one mini-batch of any size and returns its loss |
| `learningRate`, `learningRate(r)` | read or set the learning rate |
| `snapshot`, `restore` | `snapshot` remembers the weights as they are now (one snapshot per network; a new one replaces it) and `restore` puts them back. `fit`'s `restoreBest` uses them |
| `Ann scheduledRate(spec, base, epoch)` | the rate a schedule gives an epoch |
| `recordEpoch(loss, learningRate, validationLoss)`, `trainingRun(epochs, bestEpoch)` | report epochs trained with these methods, so `history`, `learningCurve`, `epochsTrained` and `bestEpoch` include them (`validationLoss` defaults to -1, `bestEpoch` to 0). `fit` calls them for you |

Optimizer state (such as Adam's moments) carries over between steps and epochs. These methods use
the CPU; dropout and input noise apply while a session or epoch runs.

## Large data: `Ann loadCsv`

An Array of rows boxes every number, so a data set like MNIST (60,000 rows of 785 values) is slow
to parse and takes gigabytes as Arrays. `Ann loadCsv(path, header)` instead reads the file in C into
an `AnnData` table, which `train`, `accuracy` and `evaluate` accept wherever they take Arrays:

```clay
data = Ann loadCsv("mnist_train.csv", true)      // true: skip the header line
x = (data columns(1)) scale(255)                 // pixels, 0 to 1
y = (data column(0)) oneHot(10)                  // labels -> 10 target columns
net train(x, y)
```

Every value in the file must be a number (spaces, tabs or commas separate them); a bad value or a
ragged line raises `ArgumentError` naming the line. Loading all of `mnist_train.csv` takes under a
second. The methods below return new tables, so they chain; `to` is exclusive and optional.

| Method | Returns |
|--------|---------|
| `rows`, `cols` | the table's size |
| `slice(from, to)` | those rows (`to` defaults to the end): for a train/test split |
| `columns(from, to)` | those columns |
| `column(n)` | one column, as a one-column table |
| `oneHot(classes)` | a one-column table of labels `0..classes-1` as rows of `classes` columns |
| `scale(divisor)` | every value divided by `divisor`; an Array of one divisor per column divides each column by its own: `x scale(x max)` puts every column in 0 to 1 (a column whose divisor is 0 is left as it is) |
| `row(i)` | row `i` as an Array |
| `AnnData fromArray(rows)` | a table from an Array of rows |
| `shuffle` | the rows in random order (repeatable after `Ann seed(n)`) |
| `split(fraction)` | `[first, rest]`: `split(0.8)` is an 80% / 20% split (shuffle first) |
| `mean`, `std` | the mean and standard deviation of each column, as Arrays |
| `min`, `max`, `median` | the smallest, largest and middle value of each column, as Arrays (the median of an even number of rows is the mean of the middle two) |
| `standardize`, `standardize(means, stds)` | every column shifted to mean 0 and scaled to standard deviation 1; with Arrays of one mean and one standard deviation per column, `(x - mean) / std` with those instead of the table's own (a std of 0 counts as 1), so the validation and test splits can use the training split's: `test standardize(train mean, train std)` |
| `argmax` | a one-column table: the index of the largest value in each row |
| `pca(k, maxRows)` | the first `k` principal components (2 by default): a Dict of `projected` (every row as `k` numbers), `components` (`k` unit vectors over the columns, most variance first), `explained` (the fraction of the variance each holds) and `mean`. Fitted on `maxRows` evenly spaced rows at most, or on all |
| `toArray` | an Array of rows (boxes every value, so avoid on big tables) |

`mean`, `std`, `min`, `max` and `median` return one value per column as an Array, and `scale` takes
such an Array, so a table can be normalized in a single pass over its columns:

```clay norun
parts = (Ann loadCsv("measurements.csv", true)) split(0.8)
train = parts at(0)
test = parts at(1)
top = train max                                  // each column's largest value in the training rows
x = train scale(top)                             // every column divided by its own: 0 to 1
xTest = test scale(top)                          // the test rows by the same divisors
```

Any Array of one divisor per column works, such as a scale you know in advance
(`data scale([255, 100, 1])`). A column whose divisor is 0 is left as it is, which keeps columns
that are always 0 (MNIST's border pixels) from failing.

`pca` sums the covariance in C a chunk of rows at a time, so it needs no copy of the table:
all of MNIST's 60,000 x 784 pixels take a few seconds. clay-plot's `embedding` and `pca` use it
when given an AnnData.

## Speed

On macOS the brick is built against Apple's Accelerate (BLAS), which makes training several times
faster than libann's own loops. Elsewhere it uses those loops. `Ann backend` says what is in use:
a Dict with `blas`, `gpu`, `gpuActive`, `threads`, `realBytes` and `optimized` (false means a slow
debug build). `Ann threads(n)` sets the CPU threads for libann's own loops; `Ann threads` reads
them; `Ann version` is libann's version.

## Examples

- [`examples/xor.clay`](examples/xor.clay): the classic problem a single layer can't learn.
- [`examples/digits.clay`](examples/digits.clay): recognizes 5×7 pixel digits, then the same digits
  with pixels flipped.
- [`examples/mnist.clay`](examples/mnist.clay): handwritten digits from the MNIST CSV files, with a
  dense network or (`--cnn`) a small convolutional one: `Ann loadCsv`, a validation split, `fit`
  with early stopping, a confusion matrix and saving the model. About 98% test accuracy in a couple
  of seconds on a Mac. With one file it holds back a fifth as the test set, so it also runs on the
  Fashion-MNIST file that comes with libann, with nothing to download:

  ```sh
  clay bricks/ann/examples/mnist.clay bricks/ann/native/libann/data/fashion-mnist_test.csv
  ```

- [`examples/text_classification.clay`](examples/text_classification.clay): short synthetic reviews
  whose sentiment depends on negations ("not bad"). `Ann tokenizer` and `encodePadded`, a masked
  embedding, `conv1d` and `globalMaxPool` (100% test accuracy) against a bag-of-words network
  (about 76%), then saving the network with its tokenizer. A port of pyann's example; it runs in
  about a second.

## Development

libann is a git submodule (`native/libann`); clone with `--recurse-submodules`. The plugin is
`native/clay_ann.c` (the tokenizer is `native/clay_text.c`), and the Clay API is `src/main.clay`.
Run the tests from the brick's directory with `clay tests/test_ann.clay` and
`clay tests/test_text.clay`; GitHub Actions runs them on Linux, macOS and Windows
(`.github/workflows/test.yml`).

## License

MIT, as is libann.
