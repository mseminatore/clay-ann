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

## Training and using a network

Data are Arrays of rows: one row of numbers per sample.

| Method | Returns |
|--------|---------|
| `train(inputs, targets)` | the final loss |
| `predict(row)` | the outputs, an Array |
| `classify(row)` | the index of the largest output |
| `accuracy(inputs, targets)` | the fraction classified correctly, 0 to 1 |
| `evaluate(inputs, targets)` | the loss over the data |
| `sizes` | the layer sizes, e.g. `[2, 4, 1]` |
| `save(path)` / `Ann load(path)` | write a network to a file / read one back |

`Ann seed(n)` makes weight initialization, and so training, repeatable. `Ann oneHot(labels,
classes)` turns class numbers into target rows: `Ann oneHot([2, 0], 3)` is
`[[0, 0, 1], [1, 0, 0]]`.

Mistakes such as the wrong number of inputs, ragged rows or an unknown activation raise a Clay
`ArgumentError`. Numbers are 32-bit floats, as everywhere in Clay and libann.

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
| `toArray` | an Array of rows (boxes every value, so avoid on big tables) |

## Examples

- [`examples/xor.clay`](examples/xor.clay): the classic problem a single layer can't learn.
- [`examples/digits.clay`](examples/digits.clay): recognizes 5×7 pixel digits, then the same digits
  with pixels flipped.

## Development

libann is a git submodule (`native/libann`); clone with `--recurse-submodules`. The plugin is
`native/clay_ann.c`, and the Clay API is `src/main.clay`. To run the tests from a project with
this brick installed: `clay tests/test_ann.clay`.

## License

MIT, as is libann.
