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

Everything else works on these networks as before: `train`, `fit`, `predictBatch`, `save`,
`quantize` and the exports. Mistakes (a convolution with no input layer, a window bigger than its
input, a `reshape` that changes the size) raise `ArgumentError`.

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

After training: `net epochsTrained`, `net bestEpoch` (with early stopping),
`net history` (a row `[loss, learningRate, validationLoss]` per epoch, with -1 when there was no
validation data), `net clearHistory`, and `net learningCurve(path)` to write the history as a CSV file.

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

An `AnnQuant` has `predict(row)`, `classify(row)`, `accuracy(inputs, targets)`, `ioParams`
(`[inputScale, inputZeroPoint, outputScale, outputZeroPoint]`) and `exportC(path, prefix)`, which
writes the integer-only version for targets without an FPU:

```clay
q = net quantize(x slice(0, 500))
Console println("float ${net accuracy(xTest, yTest)}  int8 ${q accuracy(xTest, yTest)}")
q exportC("model_q.h", "model")
```

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

`fit` returns the history, one such Dict per epoch, and puts the learning rate back as it was.

The pieces are available directly, for custom batching, streaming data or several networks in
step:

| Method | |
|--------|---|
| `trainEpoch(inputs, targets, batchSize, shuffle)` | one pass in mini-batches; returns the mean loss |
| `trainBegin`, `trainStep(inputs, targets)`, `trainEnd` | an online session: `trainStep` updates the weights on one mini-batch of any size and returns its loss |
| `learningRate`, `learningRate(r)` | read or set the learning rate |
| `snapshot`, `restore` | `snapshot` remembers the weights as they are now (one snapshot per network; a new one replaces it) and `restore` puts them back. `fit`'s `restoreBest` uses them |
| `Ann scheduledRate(spec, base, epoch)` | the rate a schedule gives an epoch |

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
| `scale(divisor)` | every value divided by `divisor` |
| `row(i)` | row `i` as an Array |
| `AnnData fromArray(rows)` | a table from an Array of rows |
| `shuffle` | the rows in random order (repeatable after `Ann seed(n)`) |
| `split(fraction)` | `[first, rest]`: `split(0.8)` is an 80% / 20% split (shuffle first) |
| `mean`, `std` | the mean and standard deviation of each column, as Arrays |
| `standardize` | every column shifted to mean 0 and scaled to standard deviation 1 |
| `argmax` | a one-column table: the index of the largest value in each row |
| `toArray` | an Array of rows (boxes every value, so avoid on big tables) |

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

## Development

libann is a git submodule (`native/libann`); clone with `--recurse-submodules`. The plugin is
`native/clay_ann.c`, and the Clay API is `src/main.clay`. To run the tests from a project with
this brick installed: `clay tests/test_ann.clay`.

## License

MIT, as is libann.
