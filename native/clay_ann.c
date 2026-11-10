/*
 * clay_ann.c - the Clay plugin for libann (native/libann, a git submodule)
 *
 * Registers the AnnNet prototype: each network object holds a libann
 * PNetwork, freed with the object; AnnData objects hold a table of numbers
 * (a libann tensor) for loading and handing big data sets to a network. Training data crosses into C as float
 * buffers (Clay floats and libann's `real` are both 32-bit floats). The Clay
 * side (src/main.clay) adds the friendlier `Ann` API on top.
 */
#include "clay_plugin.h"
#include "libann/ann.h"
#include "libann/tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const ClayPluginAPI *api;

/* --- helpers ------------------------------------------------------------ */

static PNetwork net_of(ClayPluginObject self, const char *who) {
    PNetwork n = (PNetwork)api->get_data(self);
    if (!n) {
        char msg[128];
        snprintf(msg, sizeof(msg), "%s: not a network (make one with Ann network)", who);
        api->raise(msg, "RuntimeError");
    }
    return n;
}

static int str_eq(const char *a, const char *b) { return strcmp(a, b) == 0; }

static const char *str_arg(int argc, ClayPluginValue *argv, int i, const char *dflt) {
    if (i >= argc || !api->is_string(argv[i])) return dflt;
    return api->to_string(argv[i]);
}

static int activation_of(const char *s, Activation_type *out) {
    static const struct { const char *name; Activation_type a; } map[] = {
        {"none", ACTIVATION_NULL}, {"linear", ACTIVATION_NULL}, {"sigmoid", ACTIVATION_SIGMOID},
        {"relu", ACTIVATION_RELU}, {"leaky_relu", ACTIVATION_LEAKY_RELU}, {"tanh", ACTIVATION_TANH},
        {"softsign", ACTIVATION_SOFTSIGN}, {"softmax", ACTIVATION_SOFTMAX}, {"gelu", ACTIVATION_GELU},
    };
    for (size_t i = 0; i < sizeof(map) / sizeof(map[0]); i++)
        if (str_eq(s, map[i].name)) { *out = map[i].a; return 1; }
    return 0;
}

/* rows of numbers -> a rows x cols tensor (cols from the data) */
static PTensor tensor_from(ClayPluginValue rows, const char *what) {
    if (!api->is_array(rows) || api->array_size(rows) == 0) {
        char msg[128];
        snprintf(msg, sizeof(msg), "%s must be a non-empty Array of rows", what);
        api->raise(msg, "ArgumentError");
        return NULL;
    }
    size_t r = api->array_size(rows);
    size_t n = api->to_floats(rows, NULL, 0);
    if (n % r != 0) {
        char msg[128];
        snprintf(msg, sizeof(msg), "%s: every row must have the same length", what);
        api->raise(msg, "ArgumentError");
        return NULL;
    }
    real *buf = (real *)malloc(n * sizeof(real));
    api->to_floats(rows, buf, n);
    PTensor t = tensor_create_from_array((int)r, (int)(n / r), buf);
    free(buf);
    return t;
}


/* --- AnnData: a rows x cols table of numbers held in C -------------------
 * Loading MNIST-sized data through Clay Arrays means one boxed value per
 * number; an AnnData keeps the numbers in a libann tensor instead, and
 * train/accuracy/evaluate use it directly. */

static ClayPluginObject data_proto;

static void data_free(void *data) { if (data) tensor_free((PTensor)data); }

/* an AnnData object wrapping t (takes ownership of t) */
static ClayPluginValue data_wrap(PTensor t) {
    if (!t) { api->raise("ann: out of memory", "RuntimeError"); return 0; }
    ClayPluginObject obj = api->clone(data_proto);
    api->set_data(obj, t);
    return (ClayPluginValue)obj;
}

/* the tensor inside v if it is an AnnData, else NULL (v borrowed, not copied) */
static PTensor data_of(ClayPluginValue v) {
    if (api->is_int(v) || api->is_float(v) || api->is_nil(v) || api->is_array(v) || api->is_string(v))
        return NULL;
    ClayPluginObject o = (ClayPluginObject)v;
    if (api->is_nil(api->get_slot(o, "_annData"))) return NULL;
    return (PTensor)api->get_data(o);
}

/* training data as a tensor: an AnnData is used as is (*owned = 0), an Array
 * of rows is converted to a new tensor the caller frees (*owned = 1) */
static PTensor tensor_arg(ClayPluginValue v, const char *what, int *owned) {
    PTensor t = data_of(v);
    if (t) { *owned = 0; return t; }
    *owned = 1;
    return tensor_from(v, what);
}

static int input_size(PNetwork n)  { return n->layer_count > 0 ? n->layers[0].node_count : 0; }
static int output_size(PNetwork n) { return n->layer_count > 0 ? n->layers[n->layer_count - 1].node_count : 0; }

/* --- AnnNet ------------------------------------------------------------ */

/* AnnNet create(optimizer, loss) -> a new, empty network */
static ClayPluginValue net_create(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    const char *opt = str_arg(argc, argv, 0, "adam");
    const char *loss = str_arg(argc, argv, 1, "mse");
    Optimizer_type o;
    if (str_eq(opt, "sgd")) o = OPT_SGD;
    else if (str_eq(opt, "momentum")) o = OPT_MOMENTUM;
    else if (str_eq(opt, "rmsprop")) o = OPT_RMSPROP;
    else if (str_eq(opt, "adagrad")) o = OPT_ADAGRAD;
    else if (str_eq(opt, "adam")) o = OPT_ADAM;
    else { api->raise("optimizer must be sgd, momentum, rmsprop, adagrad or adam", "ArgumentError"); return 0; }
    Loss_type l;
    if (str_eq(loss, "mse")) l = LOSS_MSE;
    else if (str_eq(loss, "cross_entropy") || str_eq(loss, "categorical_cross_entropy") ||
             str_eq(loss, "binary_cross_entropy")) l = LOSS_CATEGORICAL_CROSS_ENTROPY;
    else { api->raise("loss must be mse or cross_entropy", "ArgumentError"); return 0; }

    PNetwork n = ann_make_network(o, l);
    if (!n) { api->raise("ann: cannot create a network", "RuntimeError"); return 0; }
    ClayPluginObject obj = api->clone(self);
    api->set_data(obj, n);
    return (ClayPluginValue)obj;
}

/* net addLayer(nodes, "input" | "hidden" | "output", activation) */
static ClayPluginValue net_add_layer(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "addLayer");
    if (argc < 1 || !api->is_int(argv[0]) || api->to_int(argv[0]) <= 0) {
        api->raise("addLayer needs a node count > 0", "ArgumentError");
        return 0;
    }
    const char *kind = str_arg(argc, argv, 1, "hidden");
    if (!str_eq(kind, "input") && !str_eq(kind, "output") && !str_eq(kind, "hidden")) {
        api->raise("layer kind must be input, hidden or output", "ArgumentError");
        return 0;
    }
    Layer_type lt = str_eq(kind, "input") ? LAYER_INPUT : str_eq(kind, "output") ? LAYER_OUTPUT : LAYER_HIDDEN;
    Activation_type act;
    if (!activation_of(str_arg(argc, argv, 2, lt == LAYER_INPUT ? "none" : "sigmoid"), &act)) {
        api->raise("activation must be none, sigmoid, relu, leaky_relu, tanh, softsign, softmax or gelu",
                   "ArgumentError");
        return 0;
    }
    if (ann_add_layer(n, (int)api->to_int(argv[0]), lt, act) != ERR_OK) {
        api->raise("ann: cannot add the layer", "RuntimeError");
        return 0;
    }
    return (ClayPluginValue)self;
}

/* Keras-style shorthands: net input(n), net dense(n, act), net output(n, act) */
static ClayPluginValue layer_as(ClayPluginObject self, int argc, ClayPluginValue *argv, const char *kind) {
    ClayPluginValue args[3];
    args[0] = argc > 0 ? argv[0] : api->make_nil();
    args[1] = api->make_string(kind);
    args[2] = argc > 1 ? argv[1] : api->make_nil();
    ClayPluginValue r = net_add_layer(self, argc > 1 ? 3 : 2, args);
    api->release(args[1]);
    return r;
}
static ClayPluginValue net_input(ClayPluginObject s, int c, ClayPluginValue *v)  { return layer_as(s, c, v, "input"); }
static ClayPluginValue net_dense(ClayPluginObject s, int c, ClayPluginValue *v)  { return layer_as(s, c, v, "hidden"); }
static ClayPluginValue net_output(ClayPluginObject s, int c, ClayPluginValue *v) { return layer_as(s, c, v, "output"); }

/* net train(inputs, targets) -> the final loss. Each is an Array of rows or an AnnData */
static ClayPluginValue net_train(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "train");
    if (argc < 2) { api->raise("train needs inputs and targets", "ArgumentError"); return 0; }
    int xo, yo;
    PTensor x = tensor_arg(argv[0], "inputs", &xo);
    PTensor y = tensor_arg(argv[1], "targets", &yo);
    char msg[160] = "";
    if (x->rows != y->rows)
        snprintf(msg, sizeof(msg), "train: inputs and targets need the same number of rows");
    else if (x->cols != input_size(n) || y->cols != output_size(n))
        snprintf(msg, sizeof(msg), "train: rows must have %d inputs and %d targets (got %d and %d)",
                 input_size(n), output_size(n), x->cols, y->cols);
    real loss = msg[0] ? 0 : ann_train_network(n, x, y, x->rows);
    if (xo) tensor_free(x);
    if (yo) tensor_free(y);
    if (msg[0]) { api->raise(msg, "ArgumentError"); return 0; }
    return api->make_float(loss);
}

/* net predict(row) -> the outputs, an Array */
static ClayPluginValue net_predict(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "predict");
    int in = input_size(n), out = output_size(n);
    if (argc < 1 || (int)api->to_floats(argv[0], NULL, 0) != in) {
        char msg[96];
        snprintf(msg, sizeof(msg), "predict needs %d inputs", in);
        api->raise(msg, "ArgumentError");
        return 0;
    }
    real *x = (real *)malloc((size_t)in * sizeof(real));
    real *y = (real *)malloc((size_t)out * sizeof(real));
    api->to_floats(argv[0], x, (size_t)in);
    ann_predict(n, x, y);
    ClayPluginValue result = api->from_floats(y, (size_t)out, 0);
    free(x);
    free(y);
    return result;
}

/* net classify(row) -> the index of the largest output */
static ClayPluginValue net_classify(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "classify");
    int in = input_size(n), out = output_size(n);
    if (argc < 1 || (int)api->to_floats(argv[0], NULL, 0) != in) {
        api->raise("classify: wrong number of inputs", "ArgumentError");
        return 0;
    }
    real *x = (real *)malloc((size_t)in * sizeof(real));
    real *y = (real *)malloc((size_t)out * sizeof(real));
    api->to_floats(argv[0], x, (size_t)in);
    ann_predict(n, x, y);
    int k = ann_class_prediction(y, out);
    free(x);
    free(y);
    return api->make_int(k);
}

/* net accuracy(inputs, targets) / net evaluate(inputs, targets) (the loss) */
static ClayPluginValue net_score(ClayPluginObject self, int argc, ClayPluginValue *argv, int loss) {
    PNetwork n = net_of(self, loss ? "evaluate" : "accuracy");
    if (argc < 2) { api->raise("needs inputs and targets", "ArgumentError"); return 0; }
    int xo, yo;
    PTensor x = tensor_arg(argv[0], "inputs", &xo);
    PTensor y = tensor_arg(argv[1], "targets", &yo);
    real v = loss ? ann_evaluate_loss(n, x, y) : ann_evaluate_accuracy(n, x, y);
    if (xo) tensor_free(x);
    if (yo) tensor_free(y);
    return api->make_float(v);
}
static ClayPluginValue net_accuracy(ClayPluginObject s, int c, ClayPluginValue *v) { return net_score(s, c, v, 0); }
static ClayPluginValue net_evaluate(ClayPluginObject s, int c, ClayPluginValue *v) { return net_score(s, c, v, 1); }

/* settings: each returns the network, so they chain */
static ClayPluginValue net_set_rate(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    if (argc > 0) ann_set_learning_rate(net_of(self, "learningRate"), api->to_float(argv[0]));
    return (ClayPluginValue)self;
}
static ClayPluginValue net_set_epochs(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    if (argc > 0) ann_set_epoch_limit(net_of(self, "epochs"), (unsigned)api->to_int(argv[0]));
    return (ClayPluginValue)self;
}
static ClayPluginValue net_set_batch(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    if (argc > 0) ann_set_batch_size(net_of(self, "batchSize"), (unsigned)api->to_int(argv[0]));
    return (ClayPluginValue)self;
}
static ClayPluginValue net_set_convergence(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    if (argc > 0) ann_set_convergence(net_of(self, "convergence"), api->to_float(argv[0]));
    return (ClayPluginValue)self;
}
static ClayPluginValue net_quiet(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    (void)argc; (void)argv;
    ann_set_print_function(net_of(self, "quiet"), NULL);
    return (ClayPluginValue)self;
}

static ClayPluginValue net_sizes(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    (void)argc; (void)argv;
    PNetwork n = net_of(self, "sizes");
    ClayPluginValue arr = api->array_new();
    for (int i = 0; i < n->layer_count; i++)
        api->array_push(arr, api->make_int(n->layers[i].node_count));
    return arr;
}

/* net save(path) / AnnNet load(path) (libann's text .nna format) */
static ClayPluginValue net_save(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "save");
    const char *path = str_arg(argc, argv, 0, NULL);
    if (!path) { api->raise("save needs a file path", "ArgumentError"); return 0; }
    if (ann_save_network(n, path) != ERR_OK) { api->raise("ann: cannot save the network", "RuntimeError"); return 0; }
    return (ClayPluginValue)self;
}

static ClayPluginValue net_load(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    const char *path = str_arg(argc, argv, 0, NULL);
    if (!path) { api->raise("load needs a file path", "ArgumentError"); return 0; }
    PNetwork n = ann_load_network(path);
    if (!n) { api->raise("ann: cannot load the network", "RuntimeError"); return 0; }
    ClayPluginObject obj = api->clone(self);
    api->set_data(obj, n);
    return (ClayPluginValue)obj;
}

/* AnnNet seed(n): make weight initialization repeatable */
static ClayPluginValue net_seed(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    if (argc > 0) ann_set_seed((unsigned)api->to_int(argv[0]));
    return (ClayPluginValue)self;
}

static void net_free(void *data) { ann_free_network((PNetwork)data); }

/* libann's errors become Clay errors (raised by the methods above), so it
 * shouldn't also print them */
static void quiet_errors(int code, const char *msg, const char *func) {
    (void)code; (void)msg; (void)func;
}

/* --- AnnData methods ---------------------------------------------------- */

static char csv_error[256];
static void capture_error(int code, const char *msg, const char *func) {
    (void)code; (void)func;
    snprintf(csv_error, sizeof(csv_error), "%s", msg ? msg : "unknown error");
}

static PTensor self_tensor(ClayPluginObject self, const char *who) {
    PTensor t = (PTensor)api->get_data(self);
    if (!t) {
        char msg[96];
        snprintf(msg, sizeof(msg), "%s: not a data table (make one with Ann loadCsv)", who);
        api->raise(msg, "RuntimeError");
    }
    return t;
}

/* AnnData loadCsv(path, header) -> a table of the file's numbers. Every value
 * must be a number; header (default false) skips the first line. */
static ClayPluginValue data_load_csv(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    (void)self;
    const char *path = str_arg(argc, argv, 0, NULL);
    if (!path) { api->raise("loadCsv needs a file path", "ArgumentError"); return 0; }
    int header = argc > 1 && !api->is_nil(argv[1]) && argv[1] != api->make_bool(0);
    real *buf = NULL;
    int rows = 0, stride = 0;
    csv_error[0] = 0;
    ann_set_error_log_callback(capture_error);
    int rc = ann_load_csv(path, header ? CSV_HAS_HEADER : CSV_NO_HEADER, &buf, &rows, &stride);
    ann_set_error_log_callback(quiet_errors);
    if (rc != ERR_OK || !buf || rows <= 0 || stride <= 0) {
        char msg[320];
        snprintf(msg, sizeof(msg), "loadCsv: %s", csv_error[0] ? csv_error : "cannot read the file");
        if (buf) ann_free_csv(buf);
        api->raise(msg, rc == ERR_INVALID ? "ArgumentError" : "RuntimeError");
        return 0;
    }
    PTensor t = tensor_create_from_array(rows, stride, buf);
    ann_free_csv(buf);
    return data_wrap(t);
}

static ClayPluginValue data_rows(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    (void)argc; (void)argv;
    PTensor t = self_tensor(self, "rows");
    return t ? api->make_int(t->rows) : 0;
}
static ClayPluginValue data_cols(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    (void)argc; (void)argv;
    PTensor t = self_tensor(self, "cols");
    return t ? api->make_int(t->cols) : 0;
}

/* a range argument pair: from..to, to exclusive (to defaults to from + 1 if
 * one_default, else to the end) */
static int range_args(const char *who, int argc, ClayPluginValue *argv, int limit,
                      int one_default, int *from, int *to) {
    if (argc < 1 || !api->is_int(argv[0])) {
        char msg[96];
        snprintf(msg, sizeof(msg), "%s needs integer positions", who);
        api->raise(msg, "ArgumentError");
        return 0;
    }
    *from = (int)api->to_int(argv[0]);
    if (argc > 1 && api->is_int(argv[1])) *to = (int)api->to_int(argv[1]);
    else *to = one_default ? *from + 1 : limit;
    if (*from < 0 || *to > limit || *from >= *to) {
        char msg[128];
        snprintf(msg, sizeof(msg), "%s: %d..%d is outside 0..%d", who, *from, *to, limit);
        api->raise(msg, "ArgumentError");
        return 0;
    }
    return 1;
}

/* a copy of rows [from, to) and columns [c0, c1) */
static PTensor sub_table(PTensor t, int r0, int r1, int c0, int c1) {
    int rows = r1 - r0, cols = c1 - c0;
    PTensor r = tensor_create(rows, cols);
    if (!r) return NULL;
    for (int i = 0; i < rows; i++)
        memcpy(r->values + (size_t)i * cols, t->values + (size_t)(r0 + i) * t->cols + c0,
               (size_t)cols * sizeof(real));
    return r;
}

/* data slice(from, to) -> rows from..to-1 (to defaults to the end) */
static ClayPluginValue data_slice(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PTensor t = self_tensor(self, "slice");
    int a, b;
    if (!t || !range_args("slice", argc, argv, t->rows, 0, &a, &b)) return 0;
    return data_wrap(sub_table(t, a, b, 0, t->cols));
}

/* data columns(from, to) -> columns from..to-1 (to defaults to the end) */
static ClayPluginValue data_columns(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PTensor t = self_tensor(self, "columns");
    int a, b;
    if (!t || !range_args("columns", argc, argv, t->cols, 0, &a, &b)) return 0;
    return data_wrap(sub_table(t, 0, t->rows, a, b));
}

/* data column(n) -> column n, as a one-column table */
static ClayPluginValue data_column(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PTensor t = self_tensor(self, "column");
    int a, b;
    if (!t || !range_args("column", argc, argv, t->cols, 1, &a, &b)) return 0;
    return data_wrap(sub_table(t, 0, t->rows, a, a + 1));
}

/* data oneHot(classes) -> rows x classes, from a one-column table of labels 0..classes-1 */
static ClayPluginValue data_one_hot(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PTensor t = self_tensor(self, "oneHot");
    if (!t) return 0;
    if (argc < 1 || !api->is_int(argv[0]) || api->to_int(argv[0]) <= 0) {
        api->raise("oneHot needs the number of classes", "ArgumentError");
        return 0;
    }
    int classes = (int)api->to_int(argv[0]);
    if (t->cols != 1) {
        api->raise("oneHot needs a one-column table (use column(n) first)", "ArgumentError");
        return 0;
    }
    for (int i = 0; i < t->rows; i++) {
        real v = t->values[i];
        if (v < 0 || v >= (real)classes || v != (real)(int)v) {
            char msg[128];
            snprintf(msg, sizeof(msg), "oneHot: row %d has label %g, not a whole number below %d",
                     i, (double)v, classes);
            api->raise(msg, "ArgumentError");
            return 0;
        }
    }
    return data_wrap(tensor_onehot(t, classes));
}

/* data scale(divisor) -> a copy with every value divided by divisor, e.g. 255 for pixels */
static ClayPluginValue data_scale(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PTensor t = self_tensor(self, "scale");
    if (!t) return 0;
    if (argc < 1 || !(api->is_int(argv[0]) || api->is_float(argv[0])) || api->to_float(argv[0]) == 0) {
        api->raise("scale needs a non-zero number to divide by", "ArgumentError");
        return 0;
    }
    real d = api->to_float(argv[0]);
    PTensor r = tensor_copy(t);
    if (r) {
        size_t n = (size_t)r->rows * (size_t)r->cols;
        for (size_t i = 0; i < n; i++) r->values[i] /= d;
    }
    return data_wrap(r);
}

/* data row(i) -> row i as an Array */
static ClayPluginValue data_row(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PTensor t = self_tensor(self, "row");
    if (!t) return 0;
    if (argc < 1 || !api->is_int(argv[0]) || api->to_int(argv[0]) < 0 || api->to_int(argv[0]) >= t->rows) {
        api->raise("row: position out of range", "ArgumentError");
        return 0;
    }
    return api->from_floats(t->values + (size_t)api->to_int(argv[0]) * t->cols, (size_t)t->cols, 0);
}

/* data toArray -> an Array of rows (boxes every value: avoid on big tables) */
static ClayPluginValue data_to_array(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    (void)argc; (void)argv;
    PTensor t = self_tensor(self, "toArray");
    if (!t) return 0;
    return api->from_floats(t->values, (size_t)t->rows * (size_t)t->cols, (size_t)t->cols);
}

static ClayPluginMethod data_methods[] = {
    { "loadCsv", data_load_csv },   { "rows", data_rows },
    { "cols", data_cols },          { "slice", data_slice },
    { "columns", data_columns },    { "column", data_column },
    { "oneHot", data_one_hot },     { "scale", data_scale },
    { "row", data_row },            { "toArray", data_to_array },
    { NULL, NULL }
};

static ClayPluginMethod methods[] = {
    { "create", net_create },       { "addLayer", net_add_layer },
    { "input", net_input },         { "dense", net_dense },
    { "output", net_output },
    { "train", net_train },         { "predict", net_predict },
    { "classify", net_classify },   { "accuracy", net_accuracy },
    { "evaluate", net_evaluate },   { "learningRate", net_set_rate },
    { "epochs", net_set_epochs },   { "batchSize", net_set_batch },
    { "convergence", net_set_convergence }, { "quiet", net_quiet },
    { "sizes", net_sizes },         { "save", net_save },
    { "load", net_load },           { "seed", net_seed },
    { NULL, NULL }
};

CLAY_PLUGIN_EXPORT void ann_plugin_init(const ClayPluginAPI *clay_api) {
    api = clay_api;
    if (api->api_version < 2) return;      /* needs arrays and float buffers */
    ann_set_error_log_callback(quiet_errors);
    ClayPluginType def = { "AnnNet", methods, net_free, sizeof(void *) };
    api->register_type(&def);

    ClayPluginType data_def = { "AnnData", data_methods, data_free, sizeof(void *) };
    data_proto = api->register_type(&data_def);
    api->set_slot(data_proto, "_annData", api->make_bool(1));
}
