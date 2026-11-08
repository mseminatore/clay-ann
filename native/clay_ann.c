/*
 * clay_ann.c - the Clay plugin for libann (native/libann, a git submodule)
 *
 * Registers the AnnNet prototype: each network object holds a libann
 * PNetwork, freed with the object. Training data crosses into C as float
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

/* net train(inputs, targets) -> the final loss */
static ClayPluginValue net_train(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "train");
    if (argc < 2) { api->raise("train needs inputs and targets", "ArgumentError"); return 0; }
    PTensor x = tensor_from(argv[0], "inputs");
    PTensor y = tensor_from(argv[1], "targets");
    if (x->rows != y->rows) {
        tensor_free(x); tensor_free(y);
        api->raise("train: inputs and targets need the same number of rows", "ArgumentError");
        return 0;
    }
    if (x->cols != input_size(n) || y->cols != output_size(n)) {
        char msg[160];
        snprintf(msg, sizeof(msg), "train: rows must have %d inputs and %d targets (got %d and %d)",
                 input_size(n), output_size(n), x->cols, y->cols);
        tensor_free(x); tensor_free(y);
        api->raise(msg, "ArgumentError");
        return 0;
    }
    real loss = ann_train_network(n, x, y, x->rows);
    tensor_free(x);
    tensor_free(y);
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
    PTensor x = tensor_from(argv[0], "inputs");
    PTensor y = tensor_from(argv[1], "targets");
    real v = loss ? ann_evaluate_loss(n, x, y) : ann_evaluate_accuracy(n, x, y);
    tensor_free(x);
    tensor_free(y);
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
}
