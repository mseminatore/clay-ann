/*
 * clay_ann.c - the Clay plugin for libann (native/libann, a git submodule)
 *
 * Registers the AnnNet prototype: each network object holds a libann
 * PNetwork, freed with the object; AnnData objects hold a table of numbers
 * (a libann tensor) for loading and handing big data sets to a network. Training data crosses into C as floa
 * buffers (Clay floats and libann's `real` are both 32-bit floats). The Clay
 * side (src/main.clay) adds the friendlier `Ann` API on top.
 */
#include "clay_plugin.h"
#include "libann/ann.h"
#include "libann/tensor.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//---------------------------------------------------------------------------------
// hang onto the API objec
//---------------------------------------------------------------------------------
static const ClayPluginAPI *api;

//---------------------------------------------------------------------------------
// --- helpers ------------------------------------------------------------
//---------------------------------------------------------------------------------

//---------------------------------------------------------------------------------
// What an AnnNet object holds: the libann network, plus what libann only
// borrows and so must outlive training (the validation set, scheduler settings)
//---------------------------------------------------------------------------------
typedef struct {
    PNetwork net;
    PTensor val_x, val_y;
    real *snap;             /* the weights kept by snapshot, or NULL */
    size_t snap_count;
    union {
        LRStepParams step;
        LRExponentialParams expo;
        LRCosineParams cosine;
    } sched;
} NetData;

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static NetData *nd_of(ClayPluginObject self, const char *who) {
    NetData *d = (NetData *)api->get_data(self);

    if (!d) {
        char msg[128];
        snprintf(msg, sizeof(msg), "%s: not a network (make one with Ann network)", who);

        api->raise(msg, "RuntimeError");
    }

    return d;
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static PNetwork net_of(ClayPluginObject self, const char *who) {
    NetData *d = nd_of(self, who);

    return d ? d->net : NULL;
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static int str_eq(const char *a, const char *b) {
    return strcmp(a, b) == 0;
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static const char *str_arg(int argc, ClayPluginValue *argv, int i, const char *dflt) {
    if (i >= argc || !api->is_string(argv[i]))
        return dflt;

    return api->to_string(argv[i]);
}

//---------------------------------------------------------------------------------
// raise a Clay error for a libann error code from a file or export function
//---------------------------------------------------------------------------------
static void raise_code(const char *who, int code, const char *what) {
    char msg[256];
    snprintf(msg, sizeof(msg), "%s: %s (%s)", who, what, ann_strerror(code));

    api->raise(msg, code == ERR_INVALID || code == ERR_NULL_PTR ? "ArgumentError" : "RuntimeError");
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
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

//---------------------------------------------------------------------------------
// rows of numbers -> a rows x cols tensor (cols from the data)
//---------------------------------------------------------------------------------
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

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static void data_free(void *data) {
    if (data)
        tensor_free((PTensor)data);
}

//---------------------------------------------------------------------------------
// an AnnData object wrapping t (takes ownership of t)
//---------------------------------------------------------------------------------
static ClayPluginValue data_wrap(PTensor t) {
    if (!t) {
        api->raise("ann: out of memory", "RuntimeError");

        return CLAY_NIL;
    }

    ClayPluginObject obj = api->clone(data_proto);

    api->set_data(obj, t);

    return (ClayPluginValue)obj;
}

//---------------------------------------------------------------------------------
// the tensor inside v if it is an AnnData, else NULL (v borrowed, not copied)
//---------------------------------------------------------------------------------
static PTensor data_of(ClayPluginValue v) {
    if (api->is_int(v) || api->is_float(v) || api->is_nil(v) || api->is_array(v) || api->is_string(v))
        return NULL;

    ClayPluginObject o = (ClayPluginObject)v;

    if (api->is_nil(api->get_slot(o, "_annData")))
        return NULL;

    return (PTensor)api->get_data(o);
}

//---------------------------------------------------------------------------------
// training data as a tensor: an AnnData is used as is (*owned = 0), an Array
// of rows is converted to a new tensor the caller frees (*owned = 1)
//---------------------------------------------------------------------------------
static PTensor tensor_arg(ClayPluginValue v, const char *what, int *owned) {
    PTensor t = data_of(v);

    if (t) { *owned = 0;
        return t;
    }

    *owned = 1;

    return tensor_from(v, what);
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static int input_size(PNetwork n)  {
    return n->layer_count > 0 ? n->layers[0].node_count : 0;
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static int output_size(PNetwork n) {
    return n->layer_count > 0 ? n->layers[n->layer_count - 1].node_count : 0;
}

//---------------------------------------------------------------------------------
// --- AnnNet ------------------------------------------------------------
//---------------------------------------------------------------------------------

//---------------------------------------------------------------------------------
// libann prints its training report with fputs(stdout), which a debug
// session loses; send it where Clay's own output goes instead
//---------------------------------------------------------------------------------
static void net_print(const char *text) {
    if (api->api_version >= 3)
        api->write_output(text);
    else
        fputs(text, stdout);
}

//---------------------------------------------------------------------------------
// a new AnnNet object (a clone of proto) owning network n
//---------------------------------------------------------------------------------
static ClayPluginValue net_wrap(ClayPluginObject proto, PNetwork n) {
    NetData *d = (NetData *)calloc(1, sizeof(NetData));

    if (!d) {
        ann_free_network(n);
        api->raise("ann: out of memory", "RuntimeError");
        return CLAY_NIL;
    }

    d->net = n;

    ClayPluginObject obj = api->clone(proto);

    api->set_data(obj, d);

    return (ClayPluginValue)obj;
}

//---------------------------------------------------------------------------------
// AnnNet create(optimizer, loss) -> a new, empty network
//---------------------------------------------------------------------------------
static ClayPluginValue net_create(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    const char *opt = str_arg(argc, argv, 0, "adam");
    const char *loss = str_arg(argc, argv, 1, "mse");

    Optimizer_type o;

    if (str_eq(opt, "sgd"))
        o = OPT_SGD;
    else if (str_eq(opt, "momentum"))
        o = OPT_MOMENTUM;
    else if (str_eq(opt, "rmsprop"))
        o = OPT_RMSPROP;
    else if (str_eq(opt, "adagrad"))
        o = OPT_ADAGRAD;
    else if (str_eq(opt, "adam"))
        o = OPT_ADAM;
    else {
        api->raise("optimizer must be sgd, momentum, rmsprop, adagrad or adam", "ArgumentError");
        return CLAY_NIL;
    }

    Loss_type l;

    if (str_eq(loss, "mse"))
        l = LOSS_MSE;
    else if (str_eq(loss, "cross_entropy") || str_eq(loss, "categorical_cross_entropy") ||
             str_eq(loss, "binary_cross_entropy"))
        l = LOSS_CATEGORICAL_CROSS_ENTROPY;
    else {
        api->raise("loss must be mse or cross_entropy", "ArgumentError");
        return CLAY_NIL;
    }

    PNetwork n = ann_make_network(o, l);

    if (!n) {
        api->raise("ann: cannot create a network", "RuntimeError");
        return CLAY_NIL;
    }

    ann_set_print_function(n, net_print);

    return net_wrap(self, n);
}

//---------------------------------------------------------------------------------
// net addLayer(nodes, "input" | "hidden" | "output", activation)
//---------------------------------------------------------------------------------
static ClayPluginValue net_add_layer(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "addLayer");

    if (argc < 1 || !api->is_int(argv[0]) || api->to_int(argv[0]) <= 0) {
        api->raise("addLayer needs a node count > 0", "ArgumentError");
        return CLAY_NIL;
    }

    const char *kind = str_arg(argc, argv, 1, "hidden");

    if (!str_eq(kind, "input") && !str_eq(kind, "output") && !str_eq(kind, "hidden")) {
        api->raise("layer kind must be input, hidden or output", "ArgumentError");
        return CLAY_NIL;
    }

    Layer_type lt = str_eq(kind, "input") ? LAYER_INPUT : str_eq(kind, "output") ? LAYER_OUTPUT : LAYER_HIDDEN;
    Activation_type act;

    if (!activation_of(str_arg(argc, argv, 2, lt == LAYER_INPUT ? "none" : "sigmoid"), &act)) {
        api->raise("activation must be none, sigmoid, relu, leaky_relu, tanh, softsign, softmax or gelu",
                   "ArgumentError");
        return CLAY_NIL;
    }

    if (ann_add_layer(n, (int)api->to_int(argv[0]), lt, act) != ERR_OK) {
        api->raise("ann: cannot add the layer", "RuntimeError");
        return CLAY_NIL;
    }

    return (ClayPluginValue)self;
}

//---------------------------------------------------------------------------------
// Keras-style shorthands: net input(n), net dense(n, act), net output(n, act)
//---------------------------------------------------------------------------------
static ClayPluginValue layer_as(ClayPluginObject self, int argc, ClayPluginValue *argv, const char *kind) {
    ClayPluginValue args[3];

    args[0] = argc > 0 ? argv[0] : api->make_nil();
    args[1] = api->make_string(kind);
    args[2] = argc > 1 ? argv[1] : api->make_nil();

    ClayPluginValue r = net_add_layer(self, argc > 1 ? 3 : 2, args);

    api->release(args[1]);

    return r;
}

//---------------------------------------------------------------------------------
// net input(n): n values per sample. net input(channels, height, width): an
// image, its values channel by channel, row by row (what convolution and
// pooling layers need)
//---------------------------------------------------------------------------------
static ClayPluginValue net_input(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    if (argc >= 3 && api->is_int(argv[0]) && api->is_int(argv[1]) && api->is_int(argv[2])) {
        PNetwork n = net_of(self, "input");

        int rc = ann_add_input(n, (int)api->to_int(argv[0]), (int)api->to_int(argv[1]), (int)api->to_int(argv[2]));

        if (rc != ERR_OK) {
            raise_code("input", rc, "bad shape, or the network already has layers");
            return CLAY_NIL;
        }

        return (ClayPluginValue)self;
    }

    return layer_as(self, argc, argv, "input");
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static ClayPluginValue net_dense(ClayPluginObject s, int c, ClayPluginValue *v)  {
    return layer_as(s, c, v, "hidden");
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static ClayPluginValue net_output(ClayPluginObject s, int c, ClayPluginValue *v) {
    return layer_as(s, c, v, "output");
}

//---------------------------------------------------------------------------------
// --- image, sequence and other layers ------------------------------------
//---------------------------------------------------------------------------------

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static const char *activation_name(Activation_type a) {
    switch (a) {
    case ACTIVATION_SIGMOID: return "sigmoid";
    case ACTIVATION_RELU: return "relu";
    case ACTIVATION_LEAKY_RELU: return "leaky_relu";
    case ACTIVATION_TANH: return "tanh";
    case ACTIVATION_SOFTSIGN: return "softsign";
    case ACTIVATION_SOFTMAX: return "softmax";
    case ACTIVATION_GELU: return "gelu";
    default: return "none";
    }
}

/* argument i as an activation name (default dflt when absent or nil) */
static int act_arg(const char *who, int argc, ClayPluginValue *argv, int i, const char *dflt, Activation_type *out) {
    const char *name = dflt;
    if (i < argc && !api->is_nil(argv[i])) {
        if (!api->is_string(argv[i])) { name = ""; }
        else name = api->to_string(argv[i]);
    }
    if (!activation_of(name, out)) {
        char msg[160];
        snprintf(msg, sizeof(msg), "%s: activation must be none, sigmoid, relu, leaky_relu, tanh, softsign or gelu", who);
        api->raise(msg, "ArgumentError");
        return 0;
    }
    return 1;
}

/* argument i as a whole number >= min (a default when absent or nil, if has_default) */
static int int_arg(const char *who, int argc, ClayPluginValue *argv, int i, int has_default, int dflt, int min, int *out) {
    if (i >= argc || api->is_nil(argv[i])) {
        if (has_default) { *out = dflt; return 1; }
    } else if (api->is_int(argv[i]) && api->to_int(argv[i]) >= min) {
        *out = (int)api->to_int(argv[i]);
        return 1;
    }
    char msg[128];
    snprintf(msg, sizeof(msg), "%s: argument %d must be a whole number of at least %d", who, i + 1, min);
    api->raise(msg, "ArgumentError");
    return 0;
}

/* argument i as a size: one number (height and width alike) or [height, width] */
static int size_arg(const char *who, int argc, ClayPluginValue *argv, int i, int has_default, int dh, int dw, int *h, int *w) {
    if (i >= argc || api->is_nil(argv[i])) {
        if (has_default) { *h = dh; *w = dw; return 1; }
    } else if (api->is_int(argv[i]) && api->to_int(argv[i]) >= 1) {
        *h = *w = (int)api->to_int(argv[i]);
        return 1;
    } else if (api->is_array(argv[i]) && api->array_size(argv[i]) == 2) {
        ClayPluginValue a = api->array_get(argv[i], 0), b = api->array_get(argv[i], 1);
        if (api->is_int(a) && api->is_int(b) && api->to_int(a) >= 1 && api->to_int(b) >= 1) {
            *h = (int)api->to_int(a);
            *w = (int)api->to_int(b);
            return 1;
        }
    }
    char msg[128];
    snprintf(msg, sizeof(msg), "%s: argument %d must be a size (a number, or [height, width])", who, i + 1);
    api->raise(msg, "ArgumentError");
    return 0;
}

/* argument i as padding: a number or [height, width] of zeros per side, "valid" (none, the
 * default) or "same" (keeps the size at stride 1) */
static int padding_arg(const char *who, int argc, ClayPluginValue *argv, int i, int kh, int kw, int *ph, int *pw) {
    if (i >= argc || api->is_nil(argv[i])) {
        *ph = *pw = 0;
        return 1;
    }

    if (api->is_string(argv[i])) {
        const char *k = api->to_string(argv[i]);

        if (str_eq(k, "valid")) {
            *ph = *pw = 0;
            return 1;
        }

        if (str_eq(k, "same")) {
            *ph = (kh - 1) / 2;
            *pw = (kw - 1) / 2;
            return 1;
        }

        char msg[128];
        snprintf(msg, sizeof(msg), "%s: padding must be a number, \"valid\" or \"same\"", who);

        api->raise(msg, "ArgumentError");
        return 0;
    }

    if (api->is_int(argv[i]) && api->to_int(argv[i]) >= 0) { *ph = *pw = (int)api->to_int(argv[i]); return 1; }

    if (api->is_array(argv[i]) && api->array_size(argv[i]) == 2) {
        ClayPluginValue a = api->array_get(argv[i], 0), b = api->array_get(argv[i], 1);
        if (api->is_int(a) && api->is_int(b) && api->to_int(a) >= 0 && api->to_int(b) >= 0) {
            *ph = (int)api->to_int(a);
            *pw = (int)api->to_int(b);
            return 1;
        }
    }

    char msg[128];
    snprintf(msg, sizeof(msg), "%s: padding must be a number, [height, width], \"valid\" or \"same\"", who);

    api->raise(msg, "ArgumentError");

    return 0;
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static ClayPluginValue layer_added(ClayPluginObject self, const char *who, int rc) {
    if (rc != ERR_OK) {
        raise_code(who, rc, "can't add the layer: bad settings, no input layer yet, or a window that doesn't fit its input");

        return CLAY_NIL;
    }

    return (ClayPluginValue)self;
}

/* net conv2d(filters, kernel, activation = "none", stride = 1, padding = "valid", groups = 1)
 * kernel, stride and padding are a number or [height, width]; padding can be "same" */
static ClayPluginValue net_conv2d(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "conv2d");

    int filters, kh, kw, sh, sw, ph, pw, groups;
    Activation_type act;

    if (!int_arg("conv2d", argc, argv, 0, 0, 0, 1, &filters) ||
        !size_arg("conv2d", argc, argv, 1, 0, 0, 0, &kh, &kw) ||
        !act_arg("conv2d", argc, argv, 2, "none", &act) ||
        !size_arg("conv2d", argc, argv, 3, 1, 1, 1, &sh, &sw) ||
        !padding_arg("conv2d", argc, argv, 4, kh, kw, &ph, &pw) ||
        !int_arg("conv2d", argc, argv, 5, 1, 1, 1, &groups))
        return CLAY_NIL;

    return layer_added(self, "conv2d", ann_add_conv2d_ex(n, filters, kh, kw, sh, sw, ph, pw, groups, act));
}

/* net depthwiseConv2d(multiplier = 1, kernel, activation = "none", stride = 1, padding = "valid"):
 * each input channel gets its own filters, channels aren't mixed (follow with conv2d(n, 1) to mix) */
static ClayPluginValue net_depthwise(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "depthwiseConv2d");

    int mult, kh, kw, sh, sw, ph, pw, c = 0, h = 0, w = 0;
    Activation_type act;

    if (!int_arg("depthwiseConv2d", argc, argv, 0, 1, 1, 1, &mult) ||
        !size_arg("depthwiseConv2d", argc, argv, 1, 0, 0, 0, &kh, &kw) ||
        !act_arg("depthwiseConv2d", argc, argv, 2, "none", &act) ||
        !size_arg("depthwiseConv2d", argc, argv, 3, 1, 1, 1, &sh, &sw) ||
        !padding_arg("depthwiseConv2d", argc, argv, 4, kh, kw, &ph, &pw))
        return CLAY_NIL;

    if (n->layer_count < 1 || ann_get_layer_shape(n, n->layer_count - 1, &c, &h, &w) != ERR_OK || c < 1) {
        api->raise("depthwiseConv2d: add an input layer first", "ArgumentError");
        return CLAY_NIL;
    }

    return layer_added(self, "depthwiseConv2d", ann_add_conv2d_ex(n, c * mult, kh, kw, sh, sw, ph, pw, c, act));
}

/* net conv1d(filters, kernel, activation = "none", stride = 1, padding = "valid"): over
 * sequences, an input(features, 1, length) or the output of an embedding */
static ClayPluginValue net_conv1d(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "conv1d");

    int filters, k, stride, ph, pw;
    Activation_type act;

    if (!int_arg("conv1d", argc, argv, 0, 0, 0, 1, &filters) ||
        !int_arg("conv1d", argc, argv, 1, 0, 0, 1, &k) ||
        !act_arg("conv1d", argc, argv, 2, "none", &act) ||
        !int_arg("conv1d", argc, argv, 3, 1, 1, 1, &stride) ||
        !padding_arg("conv1d", argc, argv, 4, 1, k, &ph, &pw))
        return CLAY_NIL;

    return layer_added(self, "conv1d", ann_add_conv1d(n, filters, k, stride, pw, act));
}

/* net maxPool2d(size = 2, stride = size): size and stride are a number or [height, width] */
static ClayPluginValue net_maxpool2d(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "maxPool2d");
    int ph, pw, sh, sw;

    if (!size_arg("maxPool2d", argc, argv, 0, 1, 2, 2, &ph, &pw) ||
        !size_arg("maxPool2d", argc, argv, 1, 1, ph, pw, &sh, &sw)) return CLAY_NIL;

    return layer_added(self, "maxPool2d", ann_add_maxpool2d_ex(n, ph, pw, sh, sw));
}

/* net maxPool1d(size = 2, stride = size) */
static ClayPluginValue net_maxpool1d(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "maxPool1d");
    int size, stride;

    if (!int_arg("maxPool1d", argc, argv, 0, 1, 2, 1, &size) ||
        !int_arg("maxPool1d", argc, argv, 1, 1, size, 1, &stride)) return CLAY_NIL;

    return layer_added(self, "maxPool1d", ann_add_maxpool1d(n, size, stride));
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static ClayPluginValue net_flatten(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    CLAY_UNUSED(argc); CLAY_UNUSED(argv);

    return layer_added(self, "flatten", ann_add_flatten(net_of(self, "flatten")));
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static ClayPluginValue net_global_avg(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    CLAY_UNUSED(argc); CLAY_UNUSED(argv);

    return layer_added(self, "globalAvgPool", ann_add_global_avgpool2d(net_of(self, "globalAvgPool")));
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static ClayPluginValue net_global_max(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    CLAY_UNUSED(argc); CLAY_UNUSED(argv);

    return layer_added(self, "globalMaxPool", ann_add_global_maxpool2d(net_of(self, "globalMaxPool")));
}

/* net upsample2d(scale = 2): each value repeated in a scale x scale block */
static ClayPluginValue net_upsample(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "upsample2d");
    int scale;

    if (!int_arg("upsample2d", argc, argv, 0, 1, 2, 1, &scale))
        return CLAY_NIL;

    return layer_added(self, "upsample2d", ann_add_upsample2d(n, scale));
}

/* net reshape(channels, height, width): give the previous layer's values an image shape */
static ClayPluginValue net_reshape(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "reshape");
    int c, h, w;

    if (!int_arg("reshape", argc, argv, 0, 0, 0, 1, &c) || !int_arg("reshape", argc, argv, 1, 0, 0, 1, &h) ||
        !int_arg("reshape", argc, argv, 2, 0, 0, 1, &w))
        return CLAY_NIL;

    return layer_added(self, "reshape", ann_add_reshape(n, c, h, w));
}

/* net batchNorm(activation = "none"): normalize each channel over the batch, then the activation.
 * Put it between a layer and its activation: conv2d(16, 3, "none") batchNorm("relu") */
static ClayPluginValue net_batchnorm(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "batchNorm");
    Activation_type act;
    if (!act_arg("batchNorm", argc, argv, 0, "none", &act)) return CLAY_NIL;
    return layer_added(self, "batchNorm", ann_add_batchnorm(n, act));
}

/* net embedding(vocabulary, dimension, paddingId = -1, masked = false): token ids (the layer
 * below holds a sequence of them) to learned vectors; with masked true, steps holding
 * paddingId are left out of the layers above */
static ClayPluginValue net_embedding(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "embedding");
    int vocab, dim, pad = -1;
    if (!int_arg("embedding", argc, argv, 0, 0, 0, 1, &vocab) || !int_arg("embedding", argc, argv, 1, 0, 0, 1, &dim)) return CLAY_NIL;
    if (argc > 2 && !api->is_nil(argv[2])) {
        if (!api->is_int(argv[2]) || api->to_int(argv[2]) < -1) {
            api->raise("embedding: paddingId must be a token id, or -1 for none", "ArgumentError");
            return CLAY_NIL;
        }
        pad = (int)api->to_int(argv[2]);
    }
    int masked = argc > 3 && !api->is_nil(argv[3]) && argv[3] != api->make_bool(0);
    return layer_added(self, "embedding",
                       masked ? ann_add_embedding_masked(n, vocab, dim, pad) : ann_add_embedding(n, vocab, dim, pad));
}

//---------------------------------------------------------------------------------
// net layerInfo -> one row per layer: [kind, channels, height, width,
// activation, parameters]
//---------------------------------------------------------------------------------
static ClayPluginValue net_layer_info(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    CLAY_UNUSED(argc); CLAY_UNUSED(argv);
    PNetwork n = net_of(self, "layerInfo");
    static const char *kinds[] = { "dense", "conv2d", "maxpool2d", "flatten", "upsample2d", "reshape",
                                   "global_avgpool2d", "batchnorm", "embedding", "global_maxpool2d" };

    ClayPluginValue all = api->array_new();

    for (int i = 0; i < n->layer_count; i++) {
        int c = 0, h = 0, w = 0;

        ann_get_layer_shape(n, i, &c, &h, &w);

        int kind = (int)ann_get_layer_kind(n, i);

        const char *kname = i == 0 ? "input" : (kind >= 0 && kind < (int)(sizeof(kinds) / sizeof(kinds[0])) ? kinds[kind] : "layer");
        long params = 0;

        for (int p = 0, pc = ann_get_layer_param_count(n, i); p < pc; p++) {
            int r = 0, cc = 0;

            if (ann_get_layer_param_shape(n, i, p, &r, &cc) == ERR_OK)
                params += (long)r * cc;
        }

        ClayPluginValue row = api->array_new();
        ClayPluginValue v[6];

        v[0] = api->make_string(kname);
        v[1] = api->make_int(c);
        v[2] = api->make_int(h);
        v[3] = api->make_int(w);
        v[4] = api->make_string(i == 0 ? "none" : activation_name(ann_get_layer_activation(n, i)));
        v[5] = api->make_int((intptr_t)params);

        for (int k = 0; k < 6; k++) {
            api->array_push(row, v[k]);
            api->release(v[k]);
        }

        api->array_push(all, row);
        api->release(row);
    }

    return all;
}

//---------------------------------------------------------------------------------
// net layerCount -> the number of layers, the input layer (0) included, as in `layers`
//---------------------------------------------------------------------------------
static ClayPluginValue net_layer_count(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    CLAY_UNUSED(argc); CLAY_UNUSED(argv);

    PNetwork n = net_of(self, "layerCount");

    return n ? api->make_int(ann_get_layer_count(n)) : CLAY_NIL;
}

//---------------------------------------------------------------------------------
// layer argument 0 of who, checked against the network's layers
//---------------------------------------------------------------------------------
static int layer_arg(PNetwork n, const char *who, int argc, ClayPluginValue *argv, int *layer) {
    int count = ann_get_layer_count(n);

    if (argc < 1 || !api->is_int(argv[0]) || api->to_int(argv[0]) < 0 || api->to_int(argv[0]) >= count) {
        char msg[128];

        snprintf(msg, sizeof(msg), "%s needs a layer index, 0 to %d", who, count - 1);

        api->raise(msg, "ArgumentError");

        return 0;
    }

    *layer = (int)api->to_int(argv[0]);

    return 1;
}

//---------------------------------------------------------------------------------
// parameter p of a layer, as rows (as_rows) or one flat Array; nil if the layer has no
// such parameter
//---------------------------------------------------------------------------------
static ClayPluginValue layer_param_value(PNetwork n, const char *who, int layer, int p, int as_rows) {
    if (p >= ann_get_layer_param_count(n, layer)) return CLAY_NIL;

    int rows = 0;
    int cols = 0;

    if (ann_get_layer_param_shape(n, layer, p, &rows, &cols) != ERR_OK) return CLAY_NIL;

    size_t count = (size_t)rows * (size_t)cols;
    real *buf = (real *)malloc((count ? count : 1) * sizeof(real));

    if (!buf) {
        api->raise("ann: out of memory", "RuntimeError");

        return CLAY_NIL;
    }

    if (ann_get_layer_param(n, layer, p, buf, (int)count) != ERR_OK) {
        char msg[96];

        free(buf);

        snprintf(msg, sizeof(msg), "%s: cannot read the layer's parameters", who);

        api->raise(msg, "RuntimeError");

        return CLAY_NIL;
    }

    ClayPluginValue r = api->from_floats(buf, count, as_rows ? (size_t)cols : 0);

    free(buf);

    return r;
}

//---------------------------------------------------------------------------------
// net weights(layer) -> the layer's weights as rows, or nil for a layer without any (the
// input, pooling, flatten, reshape, upsampling). Dense: a row per node, one value per node
// below; conv: a row per filter (channel by channel, then the kernel row by row); embedding:
// a row per token id; batchNorm: one row, its scale (gamma). Initialized first, as
// training would start, if nothing has set them yet.
//---------------------------------------------------------------------------------
static ClayPluginValue net_weights(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "weights");
    int layer;

    if (!n || !layer_arg(n, "weights", argc, argv, &layer)) return CLAY_NIL;

    return layer_param_value(n, "weights", layer, 0, 1);
}

//---------------------------------------------------------------------------------
// net biases(layer) -> the layer's biases, one per node or filter (batchNorm: its shift,
// beta), or nil for a layer without them
//---------------------------------------------------------------------------------
static ClayPluginValue net_biases(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "biases");
    int layer;

    if (!n || !layer_arg(n, "biases", argc, argv, &layer)) return CLAY_NIL;

    return layer_param_value(n, "biases", layer, 1, 0);
}

//---------------------------------------------------------------------------------
// net train(inputs, targets) -> the final loss. Each is an Array of rows
// or an AnnData
//---------------------------------------------------------------------------------
static ClayPluginValue net_train(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "train");

    if (argc < 2) {
        api->raise("train needs inputs and targets", "ArgumentError");
        return CLAY_NIL;
    }

    int xo, yo;

    PTensor x = tensor_arg(argv[0], "inputs", &xo);
    PTensor y = tensor_arg(argv[1], "targets", &yo);

    char msg[160] = "";

    if (n->loss_type == LOSS_CATEGORICAL_CROSS_ENTROPY && n->layer_count > 0 &&
        n->layers[n->layer_count - 1].activation != ACTIVATION_SOFTMAX &&
        n->layers[n->layer_count - 1].activation != ACTIVATION_SIGMOID)
        snprintf(msg, sizeof(msg), "train: cross_entropy needs a softmax or sigmoid output layer");
    else if (x->rows != y->rows)
        snprintf(msg, sizeof(msg), "train: inputs and targets need the same number of rows");
    else if (x->cols != input_size(n) || y->cols != output_size(n))
        snprintf(msg, sizeof(msg), "train: rows must have %d inputs and %d targets (got %d and %d)",
                 input_size(n), output_size(n), x->cols, y->cols);

    real loss = msg[0] ? 0 : ann_train_network(n, x, y, x->rows);

    if (xo) tensor_free(x);
    if (yo) tensor_free(y);

    if (msg[0]) {
        api->raise(msg, "ArgumentError");
        return CLAY_NIL;
    }

    return api->make_float(loss);
}

//---------------------------------------------------------------------------------
// net predict(row) -> the outputs, an Array
//---------------------------------------------------------------------------------
static ClayPluginValue net_predict(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "predict");

    int in = input_size(n), out = output_size(n);

    if (argc < 1 || (int)api->to_floats(argv[0], NULL, 0) != in) {
        char msg[96];
        snprintf(msg, sizeof(msg), "predict needs %d inputs", in);
        api->raise(msg, "ArgumentError");
        return CLAY_NIL;
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

//---------------------------------------------------------------------------------
// net classify(row) -> the index of the largest outpu
//---------------------------------------------------------------------------------
static ClayPluginValue net_classify(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "classify");

    int in = input_size(n), out = output_size(n);

    if (argc < 1 || (int)api->to_floats(argv[0], NULL, 0) != in) {
        api->raise("classify: wrong number of inputs", "ArgumentError");
        return CLAY_NIL;
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

//---------------------------------------------------------------------------------
// net accuracy(inputs, targets) / net evaluate(inputs, targets) (the loss)
//---------------------------------------------------------------------------------
static ClayPluginValue net_score(ClayPluginObject self, int argc, ClayPluginValue *argv, int loss) {
    PNetwork n = net_of(self, loss ? "evaluate" : "accuracy");

    if (argc < 2) {
        api->raise("needs inputs and targets", "ArgumentError");
        return CLAY_NIL;
    }

    int xo, yo;
    PTensor x = tensor_arg(argv[0], "inputs", &xo);
    PTensor y = tensor_arg(argv[1], "targets", &yo);

    real v = loss ? ann_evaluate_loss(n, x, y) : ann_evaluate_accuracy(n, x, y);

    if (xo) tensor_free(x);
    if (yo) tensor_free(y);

    return api->make_float(v);
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static ClayPluginValue net_accuracy(ClayPluginObject s, int c, ClayPluginValue *v) {
    return net_score(s, c, v, 0);
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static ClayPluginValue net_evaluate(ClayPluginObject s, int c, ClayPluginValue *v) {
    return net_score(s, c, v, 1);
}

//---------------------------------------------------------------------------------
// settings: each returns the network, so they chain
//---------------------------------------------------------------------------------

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static ClayPluginValue net_set_epochs(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    if (argc > 0)
        ann_set_epoch_limit(net_of(self, "epochs"), (unsigned)api->to_int(argv[0]));

    return (ClayPluginValue)self;
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static ClayPluginValue net_set_batch(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    if (argc > 0)
        ann_set_batch_size(net_of(self, "batchSize"), (unsigned)api->to_int(argv[0]));

    return (ClayPluginValue)self;
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static ClayPluginValue net_set_convergence(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    if (argc > 0)
        ann_set_convergence(net_of(self, "convergence"), api->to_float(argv[0]));

    return (ClayPluginValue)self;
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static ClayPluginValue net_quiet(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    CLAY_UNUSED(argc); CLAY_UNUSED(argv);

    ann_set_print_function(net_of(self, "quiet"), NULL);

    return (ClayPluginValue)self;
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static ClayPluginValue net_sizes(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    CLAY_UNUSED(argc); CLAY_UNUSED(argv);

    PNetwork n = net_of(self, "sizes");
    ClayPluginValue arr = api->array_new();

    for (int i = 0; i < n->layer_count; i++)
        api->array_push(arr, api->make_int(n->layers[i].node_count));

    return arr;
}

//---------------------------------------------------------------------------------
// net save(path) / AnnNet load(path) (libann's text .nna format)
//---------------------------------------------------------------------------------
static ClayPluginValue net_save(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "save");

    const char *path = str_arg(argc, argv, 0, NULL);

    if (!path) {
        api->raise("save needs a file path", "ArgumentError");
        return CLAY_NIL;
    }

    if (ann_save_network(n, path) != ERR_OK) {
        api->raise("ann: cannot save the network", "RuntimeError");
        return CLAY_NIL;
    }

    return (ClayPluginValue)self;
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static ClayPluginValue net_load(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    const char *path = str_arg(argc, argv, 0, NULL);

    if (!path) {
        api->raise("load needs a file path", "ArgumentError");
        return CLAY_NIL;
    }

    PNetwork n = ann_load_network(path);

    if (!n) {
        api->raise("ann: cannot load the network", "RuntimeError");
        return CLAY_NIL;
    }

    ann_set_print_function(n, net_print);

    return net_wrap(self, n);
}

//---------------------------------------------------------------------------------
// AnnNet seed(n): make weight initialization repeatable
//---------------------------------------------------------------------------------
static ClayPluginValue net_seed(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    if (argc > 0)
        ann_set_seed((unsigned)api->to_int(argv[0]));

    return (ClayPluginValue)self;
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static void net_free(void *data) {
    NetData *d = (NetData *)data;

    if (!d) return;

    ann_free_network(d->net);          /* before the tensors it borrows */

    if (d->val_x) tensor_free(d->val_x);
    if (d->val_y) tensor_free(d->val_y);

    free(d->snap);
    free(d);
}

//---------------------------------------------------------------------------------
// libann's errors become Clay errors (raised by the methods above), so i
// shouldn't also print them */
//---------------------------------------------------------------------------------
static void quiet_errors(int code, const char *msg, const char *func) {
    CLAY_UNUSED(code); CLAY_UNUSED(msg); CLAY_UNUSED(func);
}

//---------------------------------------------------------------------------------
// --- training controls ----------------------------------------------------
//---------------------------------------------------------------------------------

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static real real_arg(const char *who, int argc, ClayPluginValue *argv, int i) {
    if (i >= argc || !(api->is_int(argv[i]) || api->is_float(argv[i]))) {
        char msg[96];
        snprintf(msg, sizeof(msg), "%s needs a number", who);
        api->raise(msg, "ArgumentError");

        return 0;
    }

    return api->to_float(argv[i]);
}

//---------------------------------------------------------------------------------
// net learningRate(r) sets it (and returns the network); net learningRate reads i
//---------------------------------------------------------------------------------
static ClayPluginValue net_set_rate(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "learningRate");
    if (argc == 0) return api->make_float(n->learning_rate);
    ann_set_learning_rate(n, real_arg("learningRate", argc, argv, 0));
    return (ClayPluginValue)self;
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static ClayPluginValue net_set_weight_decay(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "weightDecay");

    ann_set_weight_decay(n, real_arg("weightDecay", argc, argv, 0));

    return (ClayPluginValue)self;
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static ClayPluginValue net_set_l1(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "l1");

    ann_set_l1_regularization(n, real_arg("l1", argc, argv, 0));

    return (ClayPluginValue)self;
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static ClayPluginValue net_set_clip(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "gradientClip");

    ann_set_gradient_clip(n, real_arg("gradientClip", argc, argv, 0));

    return (ClayPluginValue)self;
}

//---------------------------------------------------------------------------------
// dropout(rate) for every hidden layer, or dropout(layer, rate) for one
//---------------------------------------------------------------------------------
static ClayPluginValue net_set_dropout(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "dropout");

    real rate = real_arg("dropout", argc, argv, argc > 1 ? 1 : 0);

    if (rate < 0 || rate >= 1) {
        api->raise("dropout: the rate must be from 0 up to (not including) 1", "ArgumentError");
        return CLAY_NIL;
    }

    if (argc > 1) {
        if (!api->is_int(argv[0]) || api->to_int(argv[0]) < 0 || api->to_int(argv[0]) >= n->layer_count) {
            api->raise("dropout: no such layer", "ArgumentError");
            return CLAY_NIL;
        }

        ann_set_layer_dropout(n, (int)api->to_int(argv[0]), rate);
    } else {
        ann_set_dropout(n, rate);
    }

    return (ClayPluginValue)self;
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static ClayPluginValue net_set_weight_init(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "weightInit");
    const char *k = str_arg(argc, argv, 0, "");
    Weight_init_type t;
    if (str_eq(k, "auto")) t = WEIGHT_INIT_AUTO;
    else if (str_eq(k, "uniform")) t = WEIGHT_INIT_UNIFORM;
    else if (str_eq(k, "xavier")) t = WEIGHT_INIT_XAVIER;
    else if (str_eq(k, "he")) t = WEIGHT_INIT_HE;
    else { api->raise("weightInit must be auto, uniform, xavier or he", "ArgumentError"); return CLAY_NIL; }
    ann_set_weight_init(n, t);
    return (ClayPluginValue)self;
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static ClayPluginValue net_set_loss_reduction(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "lossReduction");

    const char *k = str_arg(argc, argv, 0, "");

    if (!str_eq(k, "sum") && !str_eq(k, "mean")) {
        api->raise("lossReduction must be sum or mean", "ArgumentError");
        return CLAY_NIL;
    }

    ann_set_loss_reduction(n, str_eq(k, "sum") ? LOSS_REDUCTION_SUM : LOSS_REDUCTION_MEAN);

    return (ClayPluginValue)self;
}

//---------------------------------------------------------------------------------
// inputNoise(kind, amount, low, high): kind is none, gaussian, masking or salt_pepper
//---------------------------------------------------------------------------------
static ClayPluginValue net_set_noise(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "inputNoise");

    const char *k = str_arg(argc, argv, 0, "");
    Noise_type t;

    if (str_eq(k, "none"))
        t = NOISE_NONE;
    else if (str_eq(k, "gaussian"))
        t = NOISE_GAUSSIAN;
    else if (str_eq(k, "masking"))
        t = NOISE_MASKING;
    else if (str_eq(k, "salt_pepper"))
        t = NOISE_SALT_PEPPER;
    else {
        api->raise("inputNoise kind must be none, gaussian, masking or salt_pepper", "ArgumentError");
        return CLAY_NIL;
    }

    real amount = argc > 1 ? real_arg("inputNoise", argc, argv, 1) : 0;
    real low = argc > 2 ? real_arg("inputNoise", argc, argv, 2) : 0;
    real high = argc > 3 ? real_arg("inputNoise", argc, argv, 3) : 0;

    if (ann_set_input_noise(n, t, amount, low, high) != ERR_OK) {
        api->raise("inputNoise: bad amount or range", "ArgumentError");
        return CLAY_NIL;
    }

    return (ClayPluginValue)self;
}

//---------------------------------------------------------------------------------
// lrSchedule("step", stepSize, gamma) | ("exponential", gamma) |
// ("cosine", tMax, minLr) | ("none")
//---------------------------------------------------------------------------------
static ClayPluginValue net_set_schedule(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    NetData *d = nd_of(self, "lrSchedule");
    const char *k = str_arg(argc, argv, 0, "");

    if (str_eq(k, "none")) {
        ann_set_lr_scheduler(d->net, NULL, NULL);
    } else if (str_eq(k, "step")) {
        d->sched.step.step_size = (unsigned)(real_arg("lrSchedule", argc, argv, 1));
        d->sched.step.gamma = real_arg("lrSchedule", argc, argv, 2);
        ann_set_lr_scheduler(d->net, ann_lr_scheduler_step, &d->sched.step);
    } else if (str_eq(k, "exponential")) {
        d->sched.expo.gamma = real_arg("lrSchedule", argc, argv, 1);
        ann_set_lr_scheduler(d->net, ann_lr_scheduler_exponential, &d->sched.expo);
    } else if (str_eq(k, "cosine")) {
        d->sched.cosine.T_max = (unsigned)(real_arg("lrSchedule", argc, argv, 1));
        d->sched.cosine.min_lr = argc > 2 ? real_arg("lrSchedule", argc, argv, 2) : 0;
        ann_set_lr_scheduler(d->net, ann_lr_scheduler_cosine, &d->sched.cosine);
    } else {
        api->raise("lrSchedule kind must be step, exponential, cosine or none", "ArgumentError");
        return CLAY_NIL;
    }

    return (ClayPluginValue)self;
}

/* validation(inputs, targets): held-out data whose loss is reported each epoch and
 * watched by earlyStopping. Copied, so later changes to the tables don't matter. */
static ClayPluginValue net_set_validation(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    NetData *d = nd_of(self, "validation");

    if (argc < 2) {
        api->raise("validation needs inputs and targets", "ArgumentError");
        return CLAY_NIL;
    }

    int xo, yo;
    PTensor x = tensor_arg(argv[0], "inputs", &xo);
    PTensor y = tensor_arg(argv[1], "targets", &yo);
    PTensor cx = xo ? x : tensor_copy(x);
    PTensor cy = yo ? y : tensor_copy(y);

    if (!cx || !cy || ann_set_validation_data(d->net, cx, cy) != ERR_OK) {
        if (cx) tensor_free(cx);
        if (cy) tensor_free(cy);
        api->raise("validation: inputs and targets need the same, non-zero number of rows", "ArgumentError");
        return CLAY_NIL;
    }

    if (d->val_x) tensor_free(d->val_x);
    if (d->val_y) tensor_free(d->val_y);

    d->val_x = cx;
    d->val_y = cy;

    return (ClayPluginValue)self;
}

//---------------------------------------------------------------------------------
// earlyStopping(patience, minDelta = 0, restoreBest = true); patience 0 turns it off
//---------------------------------------------------------------------------------
static ClayPluginValue net_set_early(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "earlyStopping");
    real patience = real_arg("earlyStopping", argc, argv, 0);
    real delta = argc > 1 && !api->is_nil(argv[1]) ? real_arg("earlyStopping", argc, argv, 1) : 0;
    int restore = !(argc > 2 && !api->is_nil(argv[2]) && argv[2] == api->make_bool(0));
    if (patience < 0) { api->raise("earlyStopping: patience can't be negative", "ArgumentError"); return CLAY_NIL; }
    ann_set_early_stopping(n, (unsigned)patience, delta, restore);
    return (ClayPluginValue)self;
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static ClayPluginValue net_epochs_trained(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    CLAY_UNUSED(argc); CLAY_UNUSED(argv);

    return api->make_int((intptr_t)ann_get_epochs_trained(net_of(self, "epochsTrained")));
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static ClayPluginValue net_best_epoch(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    CLAY_UNUSED(argc); CLAY_UNUSED(argv);

    return api->make_int((intptr_t)ann_get_best_epoch(net_of(self, "bestEpoch")));
}

/* recordEpoch(loss, learningRate, validationLoss = -1): add an epoch trained outside train
 * (with trainEpoch or trainStep) to the history, so history and learningCurve include it */
static ClayPluginValue net_record_epoch(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "recordEpoch");
    if (argc < 2) { api->raise("recordEpoch needs a loss and a learning rate", "ArgumentError"); return CLAY_NIL; }
    real loss = real_arg("recordEpoch", argc, argv, 0);
    real rate = real_arg("recordEpoch", argc, argv, 1);
    real val = argc > 2 && !api->is_nil(argv[2]) ? real_arg("recordEpoch", argc, argv, 2) : -1;
    int rc = ann_record_epoch(n, loss, rate, val);
    if (rc == ERR_INVALID) { api->raise("recordEpoch: the history is full (clearHistory first)", "RuntimeError"); return CLAY_NIL; }
    if (rc != ERR_OK) { api->raise("ann: out of memory", "RuntimeError"); return CLAY_NIL; }
    return (ClayPluginValue)self;
}

/* trainingRun(epochs, bestEpoch = 0): set what epochsTrained and bestEpoch report, for a run
 * made outside train */
static ClayPluginValue net_training_run(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "trainingRun");
    int epochs, best;
    if (!int_arg("trainingRun", argc, argv, 0, 0, 0, 0, &epochs) ||
        !int_arg("trainingRun", argc, argv, 1, 1, 0, 0, &best)) return CLAY_NIL;
    if (best > epochs) { api->raise("trainingRun: bestEpoch can't be after the last epoch", "ArgumentError"); return CLAY_NIL; }
    ann_set_training_run(n, (unsigned)epochs, (unsigned)best);
    return (ClayPluginValue)self;
}

/* Every parameter of every layer, in order: counted (buf NULL), copied out to buf, or
 * (restore) copied back from it. Returns the number of values, or -1 if libann refuses. */
static long params_copy(PNetwork n, real *buf, int restore) {
    long total = 0;
    int layers = ann_get_layer_count(n);

    for (int layer = 1; layer < layers; layer++) {
        int params = ann_get_layer_param_count(n, layer);
        for (int p = 0; p < params; p++) {
            int rows = 0, cols = 0;
            if (ann_get_layer_param_shape(n, layer, p, &rows, &cols) != ERR_OK) return -1;
            int count = rows * cols;
            if (buf) {
                int rc = restore ? ann_set_layer_param(n, layer, p, buf + total, count)
                                 : ann_get_layer_param(n, layer, p, buf + total, count);
                if (rc != ERR_OK) return -1;
            }
            total += count;
        }
    }

    return total;
}

/* net snapshot: remember the weights as they are now (replacing any earlier snapshot),
 * for `restore` to put back: the best epoch of a training loop, say */
static ClayPluginValue net_snapshot(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    CLAY_UNUSED(argc); CLAY_UNUSED(argv);

    NetData *d = nd_of(self, "snapshot");
    if (!d) return CLAY_NIL;

    long count = params_copy(d->net, NULL, 0);

    if (count <= 0) {
        api->raise("snapshot: the network has no weights (add layers first)", "RuntimeError");
        return CLAY_NIL;
    }

    if ((size_t)count != d->snap_count || !d->snap) {
        real *buf = (real *)realloc(d->snap, (size_t)count * sizeof(real));

        if (!buf) {
            api->raise("snapshot: out of memory", "RuntimeError");
            return CLAY_NIL;
        }

        d->snap = buf;
        d->snap_count = (size_t)count;
    }

    if (params_copy(d->net, d->snap, 0) != count) {
        free(d->snap); d->snap = NULL; d->snap_count = 0;
        api->raise("snapshot: cannot read the weights", "RuntimeError");

        return CLAY_NIL;
    }

    return (ClayPluginValue)self;
}

//---------------------------------------------------------------------------------
// net restore: put back the weights kept by `snapshot` (which stays, so i
// can be restored again)
//---------------------------------------------------------------------------------
static ClayPluginValue net_restore(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    CLAY_UNUSED(argc); CLAY_UNUSED(argv);

    NetData *d = nd_of(self, "restore");

    if (!d) return CLAY_NIL;

    if (!d->snap) {
        api->raise("restore: nothing to restore (call snapshot first)", "RuntimeError");
        return CLAY_NIL;
    }

    if (params_copy(d->net, NULL, 0) != (long)d->snap_count) {
        api->raise("restore: the network's layers changed since the snapshot", "RuntimeError");
        return CLAY_NIL;
    }

    if (params_copy(d->net, d->snap, 1) != (long)d->snap_count) {
        api->raise("restore: cannot set the weights", "RuntimeError");
        return CLAY_NIL;
    }

    return (ClayPluginValue)self;
}

//---------------------------------------------------------------------------------
// history -> one row per epoch trained: [loss, learningRate, validationLoss]
// (-1 without validation data)
//---------------------------------------------------------------------------------
static ClayPluginValue net_history(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    CLAY_UNUSED(argc); CLAY_UNUSED(argv);

    PNetwork n = net_of(self, "history");
    unsigned count = ann_get_history_count(n);

    if (count == 0)
        return api->array_new();

    real *buf = (real *)malloc((size_t)count * 3 * sizeof(real));
    real *a = (real *)malloc((size_t)count * sizeof(real));
    real *b = (real *)malloc((size_t)count * sizeof(real));
    real *c = (real *)malloc((size_t)count * sizeof(real));

    if (!buf || !a || !b || !c) {
        free(buf);
        free(a);
        free(b);
        free(c);

        api->raise("ann: out of memory", "RuntimeError");
        return CLAY_NIL;
    }

    ann_get_history(n, a, b, c, count);

    for (unsigned i = 0; i < count; i++) {
        buf[3 * i] = a[i];
        buf[3 * i + 1] = b[i];
        buf[3 * i + 2] = c[i];
    }

    ClayPluginValue r = api->from_floats(buf, (size_t)count * 3, 3);

    free(buf);
    free(a);
    free(b);
    free(c);

    return r;
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static ClayPluginValue net_clear_history(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    CLAY_UNUSED(argc); CLAY_UNUSED(argv);

    ann_clear_history(net_of(self, "clearHistory"));

    return (ClayPluginValue)self;
}

//---------------------------------------------------------------------------------
// learningCurve(path): the history as a CSV file
//---------------------------------------------------------------------------------
static ClayPluginValue net_learning_curve(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "learningCurve");

    const char *path = str_arg(argc, argv, 0, NULL);

    if (!path) {
        api->raise("learningCurve needs a file path", "ArgumentError");
        return CLAY_NIL;
    }

    if (ann_export_learning_curve(n, path) != ERR_OK) {
        api->raise("ann: cannot write the learning curve (train first, and check the path)", "RuntimeError");
        return CLAY_NIL;
    }

    return (ClayPluginValue)self;
}

//[]---------------------------------------------------------------------------------[]
// --- training loops driven from Clay --------------------------------------
//[]---------------------------------------------------------------------------------[]

//---------------------------------------------------------------------------------
// the rows of a batch (x and y) as a checked pair of tensors; on a mismatch frees what i
// made, raises, and returns 0
//---------------------------------------------------------------------------------
static int pair_args(PNetwork n, const char *who, int argc, ClayPluginValue *argv,
                     PTensor *x, PTensor *y, int *xo, int *yo) {
    if (argc < 2) {
        char msg[96];
        snprintf(msg, sizeof(msg), "%s needs inputs and targets", who);

        api->raise(msg, "ArgumentError");
        return 0;
    }

    *x = tensor_arg(argv[0], "inputs", xo);
    *y = tensor_arg(argv[1], "targets", yo);
    char msg[160] = "";

    if ((*x)->rows != (*y)->rows)
        snprintf(msg, sizeof(msg), "%s: inputs and targets need the same number of rows", who);
    else if ((*x)->cols != input_size(n) || (*y)->cols != output_size(n))
        snprintf(msg, sizeof(msg), "%s: rows must have %d inputs and %d targets (got %d and %d)",
                 who, input_size(n), output_size(n), (*x)->cols, (*y)->cols);

                 if (msg[0]) {

        if (*xo) tensor_free(*x);
        if (*yo) tensor_free(*y);

        api->raise(msg, "ArgumentError");
        return 0;
    }

    return 1;
}

//---------------------------------------------------------------------------------
// net trainBegin: start an online session (dropout on, optimizer state kept)
//---------------------------------------------------------------------------------
static ClayPluginValue net_train_begin(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    CLAY_UNUSED(argc); CLAY_UNUSED(argv);

    PNetwork n = net_of(self, "trainBegin");
    int rc = ann_train_begin(n);

    if (rc != ERR_OK) {
        raise_code("trainBegin", rc, "cannot start (add layers first; cross_entropy needs a softmax or sigmoid output layer)");
        return CLAY_NIL;
    }

    return (ClayPluginValue)self;
}

//---------------------------------------------------------------------------------
// net trainStep(inputs, targets) -> the loss of this one mini-batch (any number of
// rows), between trainBegin and trainEnd
//---------------------------------------------------------------------------------
static ClayPluginValue net_train_step(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "trainStep");
    PTensor x, y;
    int xo, yo;

    if (!pair_args(n, "trainStep", argc, argv, &x, &y, &xo, &yo))
        return CLAY_NIL;

    real loss = ann_train_step(n, x->values, y->values, x->rows);

    if (xo) tensor_free(x);
    if (yo) tensor_free(y);

    return api->make_float(loss);
}

//---------------------------------------------------------------------------------
// net trainEnd: end the online session (dropout off); the network is ready for predict */
//---------------------------------------------------------------------------------
static ClayPluginValue net_train_end(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    CLAY_UNUSED(argc); CLAY_UNUSED(argv);

    ann_train_end(net_of(self, "trainEnd"));

    return (ClayPluginValue)self;
}

//---------------------------------------------------------------------------------
// net trainEpoch(inputs, targets, batchSize = the network's, shuffle = true) -> the mean
// loss over one pass through the data in mini-batches, in a random order unless shuffle
// is false (repeatable after Ann seed)
//---------------------------------------------------------------------------------
static ClayPluginValue net_train_epoch(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "trainEpoch");
    PTensor x, y;
    int xo, yo;

    if (!pair_args(n, "trainEpoch", argc, argv, &x, &y, &xo, &yo))
        return CLAY_NIL;

    int batch = n->batchSize > 0 ? (int)n->batchSize : 32;

    if (argc > 2 && !api->is_nil(argv[2])) {
        if (!api->is_int(argv[2]) || api->to_int(argv[2]) < 1) {
            if (xo) tensor_free(x);
            if (yo) tensor_free(y);
            api->raise("trainEpoch: batchSize must be at least 1", "ArgumentError");

            return CLAY_NIL;
        }

        batch = (int)api->to_int(argv[2]);
    }

    int shuffle = !(argc > 3 && !api->is_nil(argv[3]) && argv[3] == api->make_bool(0));

    if (batch > x->rows)
        batch = x->rows;

    int in = x->cols, out = y->cols, rows = x->rows;
    int *order = (int *)malloc((size_t)rows * sizeof(int));
    real *bx = (real *)malloc((size_t)batch * in * sizeof(real));
    real *by = (real *)malloc((size_t)batch * out * sizeof(real));
    int rc = ERR_ALLOC;
    double total = 0;
    int batches = 0;

    if (order && bx && by) {
        for (int i = 0; i < rows; i++)
            order[i] = i;

        for (int i = rows - 1; shuffle && i > 0; i--) {
            int j = rand() % (i + 1);
            int t = order[i]; order[i] = order[j]; order[j] = t;
        }

        rc = ann_train_begin(n);

        for (int start = 0; rc == ERR_OK && start < rows; start += batch) {
            int m = rows - start < batch ? rows - start : batch;

            for (int k = 0; k < m; k++) {
                memcpy(bx + (size_t)k * in, x->values + (size_t)order[start + k] * in, (size_t)in * sizeof(real));
                memcpy(by + (size_t)k * out, y->values + (size_t)order[start + k] * out, (size_t)out * sizeof(real));
            }

            total += ann_train_step(n, bx, by, m);
            batches++;
        }

        if (rc == ERR_OK)
            ann_train_end(n);
    }

    free(order);
    free(bx);
    free(by);

    if (xo) tensor_free(x);
    if (yo) tensor_free(y);

    if (rc != ERR_OK) {
        raise_code("trainEpoch", rc, "cannot train (cross_entropy needs a softmax or sigmoid output layer)");
        return CLAY_NIL;
    }

    return api->make_float((float)(total / batches));
}

/* --- evaluation ---------------------------------------------------------- */

//---------------------------------------------------------------------------------
// the outputs for every row of x (an AnnData or an Array of rows): a rows x
// outputs tensor
//---------------------------------------------------------------------------------
static PTensor run_batch(PNetwork n, PTensor x, const char *who) {
    if (x->cols != input_size(n)) {
        char msg[128];
        snprintf(msg, sizeof(msg), "%s: rows must have %d inputs (got %d)", who, input_size(n), x->cols);
        api->raise(msg, "ArgumentError");

        return NULL;
    }

    PTensor out = tensor_create(x->rows, output_size(n));

    if (!out) {
        api->raise("ann: out of memory", "RuntimeError");
        return NULL;
    }

    if (ann_predict_batch(n, x->values, out->values, x->rows) != ERR_OK) {
        tensor_free(out);
        api->raise("ann: prediction failed", "RuntimeError");

        return NULL;
    }

    return out;
}

//---------------------------------------------------------------------------------
// net predictBatch(inputs) -> the outputs of every row: an AnnData for an
// AnnData, else an Array of rows
//---------------------------------------------------------------------------------
static ClayPluginValue net_predict_batch(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "predictBatch");

    if (argc < 1) {
        api->raise("predictBatch needs inputs", "ArgumentError");
        return CLAY_NIL;
    }

    int xo;

    PTensor x = tensor_arg(argv[0], "inputs", &xo);
    PTensor out = run_batch(n, x, "predictBatch");

    ClayPluginValue r = 0;

    if (out) {
        if (xo) {
            r = api->from_floats(out->values, (size_t)out->rows * out->cols, (size_t)out->cols);
            tensor_free(out);
        } else {
            r = data_wrap(out);
        }
    }

    if (xo) tensor_free(x);

    return r;
}

//---------------------------------------------------------------------------------
// net activations(inputs, layer) -> layer's outputs for every row (after its activation): an
// AnnData for an AnnData, else an Array of rows. Each row is the layer's shape flattened:
// channel by channel, each channel row by row. layer counts from 0 (the inputs) as in
// `layers`, or from the end when negative (-1 is the output layer, -2 the one below). The
// forward pass stops at the layer (libann's ann_predict_layer).
//---------------------------------------------------------------------------------
static ClayPluginValue net_activations(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "activations");

    if (!n) return CLAY_NIL;

    int count = ann_get_layer_count(n);

    if (argc < 2 || !api->is_int(argv[1]) || api->to_int(argv[1]) >= count || api->to_int(argv[1]) < -count) {
        char msg[128];

        snprintf(msg, sizeof(msg), "activations needs inputs and a layer index, 0 to %d (or -1 to -%d from the end)",
                 count - 1, count);

        api->raise(msg, "ArgumentError");

        return CLAY_NIL;
    }

    int layer = (int)api->to_int(argv[1]);

    if (layer < 0) layer += count;

    int xo;
    PTensor x = tensor_arg(argv[0], "inputs", &xo);

    if (!x) return CLAY_NIL;

    ClayPluginValue r = CLAY_NIL;

    if (x->cols != input_size(n)) {
        char msg[128];

        snprintf(msg, sizeof(msg), "activations: rows must have %d inputs (got %d)", input_size(n), x->cols);

        api->raise(msg, "ArgumentError");
    } else {
        PTensor out = tensor_create(x->rows, n->layers[layer].node_count);

        if (!out) {
            api->raise("ann: out of memory", "RuntimeError");
        } else if (ann_predict_layer(n, x->values, layer, out->values, x->rows) != ERR_OK) {
            tensor_free(out);

            api->raise("activations: the forward pass failed", "RuntimeError");
        } else if (xo) {
            r = api->from_floats(out->values, (size_t)out->rows * out->cols, (size_t)out->cols);

            tensor_free(out);
        } else {
            r = data_wrap(out);
        }
    }

    if (xo) tensor_free(x);

    return r;
}

//---------------------------------------------------------------------------------
// net classifyBatch(inputs) -> the index of the largest output for every row:
// a one-column AnnData for an AnnData, else an Array of integers
//---------------------------------------------------------------------------------
static ClayPluginValue net_classify_batch(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "classifyBatch");

    if (argc < 1) {
        api->raise("classifyBatch needs inputs", "ArgumentError");
        return CLAY_NIL;
    }

    int xo;

    PTensor x = tensor_arg(argv[0], "inputs", &xo);
    PTensor out = run_batch(n, x, "classifyBatch");

    ClayPluginValue r = 0;

    if (out) {
        int k = out->cols;

        if (xo) {
            r = api->array_new();

            for (int i = 0; i < out->rows; i++)
                api->array_push(r, api->make_int(ann_class_prediction(out->values + (size_t)i * k, k)));
        } else {
            PTensor labels = tensor_create(out->rows, 1);

            if (labels)
                for (int i = 0; i < out->rows; i++)
                    labels->values[i] = (real)ann_class_prediction(out->values + (size_t)i * k, k);

            r = data_wrap(labels);
        }

        tensor_free(out);
    }

    if (xo) tensor_free(x);

    return r;
}

/* net confusionMatrix(inputs, targets) -> rows of counts: matrix[actual][predicted].
 * targets are one-hot rows, or a single column of class numbers. */
static ClayPluginValue net_confusion(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "confusionMatrix");
    if (argc < 2) { api->raise("confusionMatrix needs inputs and targets", "ArgumentError"); return CLAY_NIL; }
    int xo, yo;
    PTensor x = tensor_arg(argv[0], "inputs", &xo);
    PTensor y = tensor_arg(argv[1], "targets", &yo);
    int k = output_size(n);
    ClayPluginValue r = 0;
    const char *err = NULL;
    PTensor out = NULL;
    if (x->rows != y->rows) err = "confusionMatrix: inputs and targets need the same number of rows";
    else if (y->cols != k && y->cols != 1) err = "confusionMatrix: targets must be one-hot rows or one column of class numbers";
    else out = run_batch(n, x, "confusionMatrix");
    if (out) {
        long *counts = (long *)calloc((size_t)k * k, sizeof(long));
        if (!counts) err = "ann: out of memory";
        for (int i = 0; counts && i < out->rows; i++) {
            int actual = y->cols == 1 ? (int)y->values[i] : ann_class_prediction(y->values + (size_t)i * k, k);
            int pred = ann_class_prediction(out->values + (size_t)i * k, k);
            if (actual < 0 || actual >= k) { err = "confusionMatrix: a target class is out of range"; break; }
            counts[(size_t)actual * k + pred]++;
        }
        if (counts && !err) {
            r = api->array_new();
            for (int a = 0; a < k; a++) {
                ClayPluginValue row = api->array_new();
                for (int p = 0; p < k; p++) {
                    ClayPluginValue v = api->make_int((intptr_t)counts[(size_t)a * k + p]);
                    api->array_push(row, v);
                    api->release(v);
                }
                api->array_push(r, row);
                api->release(row);
            }
        }
        free(counts);
        tensor_free(out);
    }

    if (xo) tensor_free(x);
    if (yo) tensor_free(y);

    if (err) {
        api->raise(err, "ArgumentError");
        return CLAY_NIL;
    }

    return r;
}

//---------------------------------------------------------------------------------
// AnnNet backendInfo -> [blas, gpu, gpuActive, threads, bytesPerNumber, optimized]
//---------------------------------------------------------------------------------
static ClayPluginValue net_backend_info(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    CLAY_UNUSED(self); CLAY_UNUSED(argc); CLAY_UNUSED(argv);

    Backend_info info;
    memset(&info, 0, sizeof(info));

    ann_get_backend_info(&info);

    ClayPluginValue r = api->array_new();
    ClayPluginValue v[6];

    v[0] = api->make_string(info.blas ? info.blas : "none");
    v[1] = api->make_string(info.gpu ? info.gpu : "none");
    v[2] = api->make_int(info.gpu_active);
    v[3] = api->make_int(info.threads);
    v[4] = api->make_int(info.real_bytes);
    v[5] = api->make_int(info.optimized);

    for (int i = 0; i < 6; i++) {
        api->array_push(r, v[i]);
        api->release(v[i]);
    }

    return r;
}

//---------------------------------------------------------------------------------
// AnnNet threads(n): the CPU threads libann's own loops use; with no
// argument returns how many
//---------------------------------------------------------------------------------
static ClayPluginValue net_threads(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    CLAY_UNUSED(self);

    if (argc > 0) {
        if (!api->is_int(argv[0]) || api->to_int(argv[0]) < 1) {
            api->raise("threads needs a count of at least 1", "ArgumentError");
            return CLAY_NIL;
        }

        ann_set_num_threads((int)api->to_int(argv[0]));
    }

    return api->make_int(ann_get_num_threads());
}

//---------------------------------------------------------------------------------
// AnnNet version -> libann's version, e.g. "0.10.0"
//---------------------------------------------------------------------------------
static ClayPluginValue net_version(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    CLAY_UNUSED(self);
    CLAY_UNUSED(argc);
    CLAY_UNUSED(argv);

    return api->make_string(ann_get_version());
}

/* --- saving, exporting and quantizing ------------------------------------ */

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static const char *path_arg(const char *who, int argc, ClayPluginValue *argv) {
    const char *path = str_arg(argc, argv, 0, NULL);

    if (!path) {
        char msg[96];

        snprintf(msg, sizeof(msg), "%s needs a file path", who);

        api->raise(msg, "ArgumentError");
    }

    return path;
}

//---------------------------------------------------------------------------------
// net saveBinary(path) / AnnNet loadBinary(path): libann's compact binary forma
//---------------------------------------------------------------------------------
static ClayPluginValue net_save_binary(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "saveBinary");

    const char *path = path_arg("saveBinary", argc, argv);
    int rc = ann_save_network_binary(n, path);

    if (rc != ERR_OK) {
        raise_code("saveBinary", rc, "cannot write the network");
        return CLAY_NIL;
    }

    return (ClayPluginValue)self;
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static ClayPluginValue net_load_binary(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    const char *path = path_arg("loadBinary", argc, argv);
    PNetwork n = ann_load_network_binary(path);

    if (!n) {
        api->raise("loadBinary: cannot load the network (is it a binary network file?)", "RuntimeError");
        return CLAY_NIL;
    }

    ann_set_print_function(n, net_print);

    return net_wrap(self, n);
}

//---------------------------------------------------------------------------------
// net exportOnnx(path) / AnnNet importOnnx(path): ONNX JSON, for dense networks
//---------------------------------------------------------------------------------
static ClayPluginValue net_export_onnx(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "exportOnnx");
    const char *path = path_arg("exportOnnx", argc, argv);

    int rc = ann_export_onnx(n, path);

    if (rc != ERR_OK) {
        raise_code("exportOnnx", rc, "cannot export (dense layers with common activations only)");
        return CLAY_NIL;
    }

    return (ClayPluginValue)self;
}

//---------------------------------------------------------------------------------
// ANN import an ONNX file
//---------------------------------------------------------------------------------
static ClayPluginValue net_import_onnx(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    const char *path = path_arg("importOnnx", argc, argv);

    PNetwork n = ann_import_onnx(path);

    if (!n) {
        api->raise("importOnnx: cannot import (an ONNX JSON file from exportOnnx, dense layers only)", "RuntimeError");
        return CLAY_NIL;
    }

    ann_set_print_function(n, net_print);

    return net_wrap(self, n);
}

//---------------------------------------------------------------------------------
// net exportC(path, prefix = "model"): the float32 network as a C header for
// embedded targets
//---------------------------------------------------------------------------------
static ClayPluginValue net_export_c(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "exportC");

    const char *path = path_arg("exportC", argc, argv);

    int rc = ann_export_c_header(n, NULL, path, str_arg(argc, argv, 1, "model"));

    if (rc != ERR_OK) {
        raise_code("exportC", rc, "cannot export (the prefix must be a C identifier)");
        return CLAY_NIL;
    }

    return (ClayPluginValue)self;
}

//---------------------------------------------------------------------------------
// net exportJs(path): the float32 network as a JavaScript module
//---------------------------------------------------------------------------------
static ClayPluginValue net_export_js(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "exportJs");

    const char *path = path_arg("exportJs", argc, argv);
    int rc = ann_export_js(n, path);

    if (rc != ERR_OK) {
        raise_code("exportJs", rc, "cannot export");
        return CLAY_NIL;
    }

    return (ClayPluginValue)self;
}

//---------------------------------------------------------------------------------
// net diagram(path): the layers as a pikchr diagram (render with the pikchr tool)
//---------------------------------------------------------------------------------
static ClayPluginValue net_diagram(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "diagram");

    const char *path = path_arg("diagram", argc, argv);
    int rc = ann_export_pikchr(n, path);

    if (rc != ERR_OK) {
        raise_code("diagram", rc, "cannot write the diagram");
        return CLAY_NIL;
    }

    return (ClayPluginValue)self;
}

//---------------------------------------------------------------------------------
// AnnQuant: an int8 copy of a network (see net quantize)
//---------------------------------------------------------------------------------
typedef struct { PQuantizedNetwork q; int in, out; } QuantData;

static ClayPluginObject quant_proto;

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static void quant_free(void *data) {
    QuantData *d = (QuantData *)data;
    if (!d) return;

    ann_free_quantized(d->q);

    free(d);
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static QuantData *quant_of(ClayPluginObject self, const char *who) {
    QuantData *d = (QuantData *)api->get_data(self);
    if (!d) {
        char msg[112];
        snprintf(msg, sizeof(msg), "%s: not an int8 network (make one with net quantize)", who);
        api->raise(msg, "RuntimeError");
    }
    return d;
}

//---------------------------------------------------------------------------------
// net quantize(calibrationInputs) -> an int8 copy, its ranges found by running these rows
// (a few hundred training rows are usually enough)
//---------------------------------------------------------------------------------
static ClayPluginValue net_quantize(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PNetwork n = net_of(self, "quantize");
    if (argc < 1) { api->raise("quantize needs calibration inputs", "ArgumentError"); return CLAY_NIL; }
    int xo;

    PTensor x = tensor_arg(argv[0], "calibration inputs", &xo);

    PQuantizedNetwork q = NULL;

    if (x->cols != input_size(n)) {
        if (xo) tensor_free(x);
        api->raise("quantize: calibration rows must have the network's number of inputs", "ArgumentError");
        return CLAY_NIL;
    }

    q = ann_quantize(n, x);
    if (xo)
        tensor_free(x);

    if (!q) {
        api->raise("quantize: this network can't be quantized (see libann's docs/C_EXPORT.md)", "ArgumentError");
        return CLAY_NIL;
    }

    QuantData *d = (QuantData *)calloc(1, sizeof(QuantData));

    if (!d) {
        ann_free_quantized(q);
        api->raise("ann: out of memory", "RuntimeError");

        return CLAY_NIL;
    }

    d->q = q;
    d->in = input_size(n);
    d->out = output_size(n);

    ClayPluginObject obj = api->clone(quant_proto);

    api->set_data(obj, d);

    return (ClayPluginValue)obj;
}

//---------------------------------------------------------------------------------
// q predict(row) -> the outputs, an Array
//---------------------------------------------------------------------------------
static ClayPluginValue quant_predict(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    QuantData *d = quant_of(self, "predict");

    if (argc < 1 || (int)api->to_floats(argv[0], NULL, 0) != d->in) {
        char msg[96];
        snprintf(msg, sizeof(msg), "predict needs %d inputs", d->in);

        api->raise(msg, "ArgumentError");

        return CLAY_NIL;
    }

    real *x = (real *)malloc((size_t)d->in * sizeof(real));
    real *y = (real *)malloc((size_t)d->out * sizeof(real));
    api->to_floats(argv[0], x, (size_t)d->in);

    int rc = ann_quantized_predict(d->q, x, y);

    ClayPluginValue r = rc == ERR_OK ? api->from_floats(y, (size_t)d->out, 0) : 0;

    free(x);
    free(y);

    if (rc != ERR_OK) {
        api->raise("predict failed", "RuntimeError");
        return CLAY_NIL;
    }

    return r;
}

//---------------------------------------------------------------------------------
// q classify(row) -> the index of the largest outpu
//---------------------------------------------------------------------------------
static ClayPluginValue quant_classify(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    QuantData *d = quant_of(self, "classify");

    if (argc < 1 || (int)api->to_floats(argv[0], NULL, 0) != d->in) {
        api->raise("classify: wrong number of inputs", "ArgumentError");
        return CLAY_NIL;
    }

    real *x = (real *)malloc((size_t)d->in * sizeof(real));
    real *y = (real *)malloc((size_t)d->out * sizeof(real));

    api->to_floats(argv[0], x, (size_t)d->in);

    int rc = ann_quantized_predict(d->q, x, y);
    int k = rc == ERR_OK ? ann_class_prediction(y, d->out) : -1;

    free(x);
    free(y);

    if (rc != ERR_OK) {
        api->raise("classify failed", "RuntimeError");
        return CLAY_NIL;
    }

    return api->make_int(k);
}

//---------------------------------------------------------------------------------
// q accuracy(inputs, targets) -> the fraction classified correctly, 0 to 1
//---------------------------------------------------------------------------------
static ClayPluginValue quant_accuracy(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    QuantData *d = quant_of(self, "accuracy");

    if (argc < 2) {
        api->raise("accuracy needs inputs and targets", "ArgumentError");
        return CLAY_NIL;
    }

    int xo, yo;

    PTensor x = tensor_arg(argv[0], "inputs", &xo);
    PTensor y = tensor_arg(argv[1], "targets", &yo);

    real v = (x->rows == y->rows && x->cols == d->in && y->cols == d->out)
                 ? ann_quantized_evaluate_accuracy(d->q, x, y) : -1;

    if (xo) tensor_free(x);
    if (yo) tensor_free(y);

    if (v < 0) {
        api->raise("accuracy: inputs and targets must match the network's inputs and outputs", "ArgumentError");
        return CLAY_NIL;
    }

    return api->make_float(v);
}

//---------------------------------------------------------------------------------
// q exportC(path, prefix = "model"): the int8 network as a C header
// (needs ann_rt.h beside it)
//---------------------------------------------------------------------------------
static ClayPluginValue quant_export_c(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    QuantData *d = quant_of(self, "exportC");
    const char *path = path_arg("exportC", argc, argv);
    int rc = ann_export_c_header(NULL, d->q, path, str_arg(argc, argv, 1, "model"));
    if (rc != ERR_OK) { raise_code("exportC", rc, "cannot export (the prefix must be a C identifier)"); return CLAY_NIL; }
    return (ClayPluginValue)self;
}

//---------------------------------------------------------------------------------
// q ioParams -> [inputScale, inputZeroPoint, outputScale, outputZeroPoint]:
// real = scale * (int8 - zeroPoint)
//---------------------------------------------------------------------------------
static ClayPluginValue quant_io(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    CLAY_UNUSED(argc); CLAY_UNUSED(argv);
    QuantData *d = quant_of(self, "ioParams");
    real is, os;
    int iz, oz;
    if (ann_quantized_get_io_params(d->q, &is, &iz, &os, &oz) != ERR_OK) {
        api->raise("ioParams failed", "RuntimeError");
        return CLAY_NIL;
    }
    float v[4] = { is, (float)iz, os, (float)oz };
    return api->from_floats(v, 4, 0);
}

//---------------------------------------------------------------------------------
// AnnQuant methods
//---------------------------------------------------------------------------------
static ClayPluginMethod quant_methods[] = {
    { "predict", quant_predict },   { "classify", quant_classify },
    { "accuracy", quant_accuracy }, { "exportC", quant_export_c },
    { "ioParams", quant_io },
    { NULL, NULL }
};

/* --- AnnData methods ---------------------------------------------------- */

static char csv_error[256];

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static void capture_error(int code, const char *msg, const char *func) {
    CLAY_UNUSED(code); CLAY_UNUSED(func);
    snprintf(csv_error, sizeof(csv_error), "%s", msg ? msg : "unknown error");
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static PTensor self_tensor(ClayPluginObject self, const char *who) {
    PTensor t = (PTensor)api->get_data(self);

    if (!t) {
        char msg[96];
        snprintf(msg, sizeof(msg), "%s: not a data table (make one with Ann loadCsv)", who);
        api->raise(msg, "RuntimeError");
    }

    return t;
}

//---------------------------------------------------------------------------------
// AnnData loadCsv(path, header) -> a table of the file's numbers. Every value
// must be a number; header (default false) skips the first line.
//---------------------------------------------------------------------------------
static ClayPluginValue data_load_csv(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    CLAY_UNUSED(self);

    const char *path = str_arg(argc, argv, 0, NULL);

    if (!path) {
        api->raise("loadCsv needs a file path", "ArgumentError");
        return CLAY_NIL;
    }

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

        if (buf)
            ann_free_csv(buf);

        api->raise(msg, rc == ERR_INVALID ? "ArgumentError" : "RuntimeError");

        return CLAY_NIL;
    }

    PTensor t = tensor_create_from_array(rows, stride, buf);
    ann_free_csv(buf);

    return data_wrap(t);
}

//---------------------------------------------------------------------------------
// AnnData rows
//---------------------------------------------------------------------------------
static ClayPluginValue data_rows(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    CLAY_UNUSED(argc); CLAY_UNUSED(argv);

    PTensor t = self_tensor(self, "rows");

    return t ? api->make_int(t->rows) : 0;
}

//---------------------------------------------------------------------------------
// AnnData cols
//---------------------------------------------------------------------------------
static ClayPluginValue data_cols(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    CLAY_UNUSED(argc); CLAY_UNUSED(argv);
    PTensor t = self_tensor(self, "cols");
    return t ? api->make_int(t->cols) : 0;
}

//---------------------------------------------------------------------------------
// a range argument pair: from..to, to exclusive (to defaults to from + 1 if
// one_default, else to the end)
//---------------------------------------------------------------------------------
static int range_args(const char *who, int argc, ClayPluginValue *argv, int limit,
                      int one_default, int *from, int *to) {
    if (argc < 1 || !api->is_int(argv[0])) {
        char msg[96];
        snprintf(msg, sizeof(msg), "%s needs integer positions", who);

        api->raise(msg, "ArgumentError");

        return 0;
    }

    *from = (int)api->to_int(argv[0]);

    if (argc > 1 && api->is_int(argv[1]))
        *to = (int)api->to_int(argv[1]);

    else *to = one_default ? *from + 1 : limit;

    if (*from < 0 || *to > limit || *from >= *to) {
        char msg[128];
        snprintf(msg, sizeof(msg), "%s: %d..%d is outside 0..%d", who, *from, *to, limit);
        api->raise(msg, "ArgumentError");

        return 0;
    }

    return 1;
}

//---------------------------------------------------------------------------------
// a copy of rows [from, to) and columns [c0, c1)
//---------------------------------------------------------------------------------
static PTensor sub_table(PTensor t, int r0, int r1, int c0, int c1) {
    int rows = r1 - r0, cols = c1 - c0;

    PTensor r = tensor_create(rows, cols);

    if (!r) return NULL;

    for (int i = 0; i < rows; i++)
        memcpy(r->values + (size_t)i * cols, t->values + (size_t)(r0 + i) * t->cols + c0,
               (size_t)cols * sizeof(real));

    return r;
}

//---------------------------------------------------------------------------------
// data slice(from, to) -> rows from..to-1 (to defaults to the end)
//---------------------------------------------------------------------------------
static ClayPluginValue data_slice(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PTensor t = self_tensor(self, "slice");

    int a, b;

    if (!t || !range_args("slice", argc, argv, t->rows, 0, &a, &b))
        return CLAY_NIL;

    return data_wrap(sub_table(t, a, b, 0, t->cols));
}

//---------------------------------------------------------------------------------
// data columns(from, to) -> columns from..to-1 (to defaults to the end)
//---------------------------------------------------------------------------------
static ClayPluginValue data_columns(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PTensor t = self_tensor(self, "columns");

    int a, b;

    if (!t || !range_args("columns", argc, argv, t->cols, 0, &a, &b))
        return CLAY_NIL;

    return data_wrap(sub_table(t, 0, t->rows, a, b));
}

//---------------------------------------------------------------------------------
// data column(n) -> column n, as a one-column table
//---------------------------------------------------------------------------------
static ClayPluginValue data_column(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PTensor t = self_tensor(self, "column");

    int a, b;

    if (!t || !range_args("column", argc, argv, t->cols, 1, &a, &b))
        return CLAY_NIL;

    return data_wrap(sub_table(t, 0, t->rows, a, a + 1));
}

//---------------------------------------------------------------------------------
// data oneHot(classes) -> rows x classes, from a one-column table of labels
// 0..classes-1
//---------------------------------------------------------------------------------
static ClayPluginValue data_one_hot(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PTensor t = self_tensor(self, "oneHot");

    if (!t) return CLAY_NIL;

    if (argc < 1 || !api->is_int(argv[0]) || api->to_int(argv[0]) <= 0) {
        api->raise("oneHot needs the number of classes", "ArgumentError");

        return CLAY_NIL;
    }

    int classes = (int)api->to_int(argv[0]);

    if (t->cols != 1) {
        api->raise("oneHot needs a one-column table (use column(n) first)", "ArgumentError");

        return CLAY_NIL;
    }

    for (int i = 0; i < t->rows; i++) {
        real v = t->values[i];

        if (v < 0 || v >= (real)classes || v != (real)(int)v) {
            char msg[128];
            snprintf(msg, sizeof(msg), "oneHot: row %d has label %g, not a whole number below %d",
                     i, (double)v, classes);
            api->raise(msg, "ArgumentError");

            return CLAY_NIL;
        }
    }

    return data_wrap(tensor_onehot(t, classes));
}

//---------------------------------------------------------------------------------
// data scale(divisor) -> a copy with every value divided by divisor, e.g. 255 for pixels.
// An Array of one divisor per column divides each column by its own (data scale(data max)):
// a column whose divisor is 0 is left as it is, since it can only be all zeros when the
// divisor is its maximum.
//---------------------------------------------------------------------------------
static ClayPluginValue data_scale(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PTensor t = self_tensor(self, "scale");

    if (!t) return CLAY_NIL;

    if (argc >= 1 && api->is_array(argv[0])) {
        size_t n = api->array_size(argv[0]);
        int numbers = 1;

        for (size_t c = 0; c < n; c++) {
            ClayPluginValue e = api->array_get(argv[0], c);

            if (!(api->is_int(e) || api->is_float(e)))
                numbers = 0;
        }

        if (n != (size_t)t->cols || !numbers) {
            char msg[128];

            snprintf(msg, sizeof(msg), "scale needs one number per column: %d columns, got %lu values%s",
                     t->cols, (unsigned long)n, numbers ? "" : " (not all numbers)");

            api->raise(msg, "ArgumentError");

            return CLAY_NIL;
        }

        real *d = (real *)malloc(n * sizeof(real));
        PTensor r = d ? tensor_copy(t) : NULL;

        if (!d || !r) {
            free(d);

            if (r)
                tensor_free(r);

            api->raise("ann: out of memory", "RuntimeError");

            return CLAY_NIL;
        }

        api->to_floats(argv[0], d, n);

        for (size_t c = 0; c < n; c++)
            if (d[c] == 0) d[c] = 1;

        for (int i = 0; i < r->rows; i++) {
            real *row = r->values + (size_t)i * r->cols;

            for (int c = 0; c < r->cols; c++)
                row[c] /= d[c];
        }

        free(d);

        return data_wrap(r);
    }

    if (argc < 1 || !(api->is_int(argv[0]) || api->is_float(argv[0])) || api->to_float(argv[0]) == 0) {
        api->raise("scale needs a non-zero number (or an Array of one per column) to divide by", "ArgumentError");

        return CLAY_NIL;
    }

    real d = api->to_float(argv[0]);
    PTensor r = tensor_copy(t);

    if (r) {
        size_t n = (size_t)r->rows * (size_t)r->cols;
        for (size_t i = 0; i < n; i++)
            r->values[i] /= d;
    }

    return data_wrap(r);
}

//---------------------------------------------------------------------------------
// data row(i) -> row i as an Array
//---------------------------------------------------------------------------------
static ClayPluginValue data_row(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PTensor t = self_tensor(self, "row");

    if (!t) return CLAY_NIL;

    if (argc < 1 || !api->is_int(argv[0]) || api->to_int(argv[0]) < 0 || api->to_int(argv[0]) >= t->rows) {
        api->raise("row: position out of range", "ArgumentError");
        return CLAY_NIL;
    }

    return api->from_floats(t->values + (size_t)api->to_int(argv[0]) * t->cols, (size_t)t->cols, 0);
}

//---------------------------------------------------------------------------------
// data toArray -> an Array of rows (boxes every value: avoid on big tables)
//---------------------------------------------------------------------------------
static ClayPluginValue data_to_array(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    CLAY_UNUSED(argc); CLAY_UNUSED(argv);

    PTensor t = self_tensor(self, "toArray");

    if (!t) return CLAY_NIL;

    return api->from_floats(t->values, (size_t)t->rows * (size_t)t->cols, (size_t)t->cols);
}

//---------------------------------------------------------------------------------
// AnnData fromArray(rows) -> a table from an Array of rows of numbers
//---------------------------------------------------------------------------------
static ClayPluginValue data_from_array(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    CLAY_UNUSED(self);

    if (argc < 1) {
        api->raise("fromArray needs an Array of rows", "ArgumentError");
        return CLAY_NIL;
    }

    return data_wrap(tensor_from(argv[0], "rows"));
}

//---------------------------------------------------------------------------------
// data shuffle -> the rows in random order (repeatable after Ann seed)
//---------------------------------------------------------------------------------
static ClayPluginValue data_shuffle(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    CLAY_UNUSED(argc); CLAY_UNUSED(argv);

    PTensor t = self_tensor(self, "shuffle");

    if (!t) return CLAY_NIL;

    PTensor r = tensor_copy(t);

    if (!r) {
        api->raise("ann: out of memory", "RuntimeError");
        return CLAY_NIL;
    }

    size_t cols = (size_t)r->cols;
    real *tmp = (real *)malloc(cols * sizeof(real));

    if (!tmp) {
        tensor_free(r);
        api->raise("ann: out of memory", "RuntimeError");
        return CLAY_NIL;
    }

    for (int i = r->rows - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        if (j == i) continue;
        memcpy(tmp, r->values + (size_t)i * cols, cols * sizeof(real));
        memcpy(r->values + (size_t)i * cols, r->values + (size_t)j * cols, cols * sizeof(real));
        memcpy(r->values + (size_t)j * cols, tmp, cols * sizeof(real));
    }

    free(tmp);

    return data_wrap(r);
}

//---------------------------------------------------------------------------------
// data split(fraction) -> [first, rest]: the first rows (fraction of them, e.g. 0.8)
// and the others
//---------------------------------------------------------------------------------
static ClayPluginValue data_split(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PTensor t = self_tensor(self, "split");

    if (!t) return CLAY_NIL;

    if (argc < 1 || !(api->is_int(argv[0]) || api->is_float(argv[0]))) {
        api->raise("split needs a fraction between 0 and 1", "ArgumentError");
        return CLAY_NIL;
    }

    double f = (double)api->to_float(argv[0]);
    int cut = (int)(f * t->rows + 0.5);

    if (!(f > 0 && f < 1) || cut < 1 || cut >= t->rows) {
        api->raise("split: the fraction must leave rows on both sides", "ArgumentError");
        return CLAY_NIL;
    }

    ClayPluginValue a = data_wrap(sub_table(t, 0, cut, 0, t->cols));
    ClayPluginValue b = data_wrap(sub_table(t, cut, t->rows, 0, t->cols));
    ClayPluginValue r = api->array_new();

    api->array_push(r, a); api->release(a);
    api->array_push(r, b); api->release(b);

    return r;
}

//---------------------------------------------------------------------------------
// the mean and standard deviation of every column
//---------------------------------------------------------------------------------
static void column_stats(PTensor t, double *mean, double *sd) {
    for (int c = 0; c < t->cols; c++) {
        double sum = 0, sq = 0;

        for (int i = 0; i < t->rows; i++) {
            double v = t->values[(size_t)i * t->cols + c];
            sum += v;
            sq += v * v;
        }

        mean[c] = sum / t->rows;
        double var = sq / t->rows - mean[c] * mean[c];
        sd[c] = var > 0 ? sqrt(var) : 0;
    }
}

//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
static ClayPluginValue data_stat(ClayPluginObject self, const char *who, int want_sd) {
    PTensor t = self_tensor(self, who);

    if (!t) return CLAY_NIL;

    double *mean = (double *)malloc((size_t)t->cols * sizeof(double));
    double *sd = (double *)malloc((size_t)t->cols * sizeof(double));
    real *out = (real *)malloc((size_t)t->cols * sizeof(real));

    if (!mean || !sd || !out) {
        free(mean);
        free(sd);
        free(out);
        api->raise("ann: out of memory", "RuntimeError");
        return CLAY_NIL;
    }

    column_stats(t, mean, sd);

    for (int c = 0; c < t->cols; c++)
        out[c] = (real)(want_sd ? sd[c] : mean[c]);

    ClayPluginValue r = api->from_floats(out, (size_t)t->cols, 0);

    free(mean);
    free(sd);
    free(out);

    return r;
}

//---------------------------------------------------------------------------------
// data mean -> the mean of each column, an Array
//---------------------------------------------------------------------------------
static ClayPluginValue data_mean(ClayPluginObject s, int c, ClayPluginValue *v) {
    CLAY_UNUSED(c); CLAY_UNUSED(v);

    return data_stat(s, "mean", 0);
}

//---------------------------------------------------------------------------------
// data std -> the standard deviation of each column, an Array
//---------------------------------------------------------------------------------
static ClayPluginValue data_std(ClayPluginObject s, int c, ClayPluginValue *v) {
    CLAY_UNUSED(c); CLAY_UNUSED(v);

    return data_stat(s, "std", 1);
}

//---------------------------------------------------------------------------------
// qsort's order for a column of values
//---------------------------------------------------------------------------------
static int real_cmp(const void *a, const void *b) {
    real x = *(const real *)a;
    real y = *(const real *)b;

    return (x > y) - (x < y);
}

//---------------------------------------------------------------------------------
// the smallest (kind 0), largest (1) or median (2) of every column, an Array; the median of
// an even number of rows is the mean of the middle two
//---------------------------------------------------------------------------------
static ClayPluginValue data_extreme(ClayPluginObject self, const char *who, int kind) {
    PTensor t = self_tensor(self, who);

    if (!t) return CLAY_NIL;

    size_t rows = (size_t)t->rows;
    size_t cols = (size_t)t->cols;
    real *out = (real *)malloc(cols * sizeof(real));
    real *col = (kind == 2) ? (real *)malloc(rows * sizeof(real)) : NULL;

    if (!out || (kind == 2 && !col)) {
        free(out);
        free(col);
        api->raise("ann: out of memory", "RuntimeError");

        return CLAY_NIL;
    }

    for (size_t c = 0; c < cols; c++) {
        if (kind == 2) {
            for (size_t i = 0; i < rows; i++)
                col[i] = t->values[i * cols + c];

            qsort(col, rows, sizeof(real), real_cmp);

            out[c] = (rows % 2) ? col[rows / 2] : (real)(((double)col[rows / 2 - 1] + (double)col[rows / 2]) / 2);
        } else {
            real m = t->values[c];

            for (size_t i = 1; i < rows; i++) {
                real v = t->values[i * cols + c];

                if (kind == 0 ? v < m : v > m)
                    m = v;
            }

            out[c] = m;
        }
    }

    ClayPluginValue r = api->from_floats(out, cols, 0);

    free(out);
    free(col);

    return r;
}

//---------------------------------------------------------------------------------
// data min -> the smallest value of each column, an Array
//---------------------------------------------------------------------------------
static ClayPluginValue data_min(ClayPluginObject s, int c, ClayPluginValue *v) {
    CLAY_UNUSED(c); CLAY_UNUSED(v);

    return data_extreme(s, "min", 0);
}

//---------------------------------------------------------------------------------
// data max -> the largest value of each column, an Array
//---------------------------------------------------------------------------------
static ClayPluginValue data_max(ClayPluginObject s, int c, ClayPluginValue *v) {
    CLAY_UNUSED(c); CLAY_UNUSED(v);

    return data_extreme(s, "max", 1);
}

//---------------------------------------------------------------------------------
// data median -> the middle value of each column, an Array
//---------------------------------------------------------------------------------
static ClayPluginValue data_median(ClayPluginObject s, int c, ClayPluginValue *v) {
    CLAY_UNUSED(c); CLAY_UNUSED(v);

    return data_extreme(s, "median", 2);
}

//---------------------------------------------------------------------------------
// data standardize -> a copy with each column shifted to mean 0 and scaled to standard
// deviation 1 (a column that never varies is only shifted)
//---------------------------------------------------------------------------------
static ClayPluginValue data_standardize(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    CLAY_UNUSED(argc); CLAY_UNUSED(argv);
    PTensor t = self_tensor(self, "standardize");
    if (!t) return CLAY_NIL;
    PTensor r = tensor_copy(t);
    double *mean = (double *)malloc((size_t)t->cols * sizeof(double));
    double *sd = (double *)malloc((size_t)t->cols * sizeof(double));
    if (!r || !mean || !sd) {
        if (r) tensor_free(r);
        free(mean); free(sd);
        api->raise("ann: out of memory", "RuntimeError");
        return CLAY_NIL;
    }
    column_stats(t, mean, sd);
    for (int i = 0; i < r->rows; i++)
        for (int c = 0; c < r->cols; c++) {
            real *v = &r->values[(size_t)i * r->cols + c];
            *v = (real)((*v - mean[c]) / (sd[c] > 0 ? sd[c] : 1));
        }
    free(mean); free(sd);
    return data_wrap(r);
}

//---------------------------------------------------------------------------------
// data argmax -> a one-column table: the index of the largest value in each row
//---------------------------------------------------------------------------------
static ClayPluginValue data_argmax(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    CLAY_UNUSED(argc); CLAY_UNUSED(argv);

    PTensor t = self_tensor(self, "argmax");
    if (!t) return CLAY_NIL;

    PTensor r = tensor_create(t->rows, 1);

    if (r)
        for (int i = 0; i < t->rows; i++)
            r->values[i] = (real)ann_class_prediction(t->values + (size_t)i * t->cols, t->cols);

    return data_wrap(r);
}

//---------------------------------------------------------------------------------
// principal components
//---------------------------------------------------------------------------------

//---------------------------------------------------------------------------------
// The eigenvalues and eigenvectors of the symmetric n x n matrix a (row-major,
// destroyed), by cyclic Jacobi rotations: vals[i] and column i of vecs, sorted
// largest first.
//---------------------------------------------------------------------------------
static void jacobi_eigen(double *a, int n, double *vals, double *vecs) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            vecs[i * n + j] = i == j;

    for (int sweep = 0; sweep < 100; sweep++) {
        double off = 0;
        double diag = 0;

        for (int i = 0; i < n; i++) {
            diag += a[i * n + i] * a[i * n + i];

            for (int j = i + 1; j < n; j++)
                off += a[i * n + j] * a[i * n + j];
        }

        if (off == 0 || off <= 1e-30 * diag) break;

        for (int p = 0; p < n; p++)
            for (int q = p + 1; q < n; q++) {
                double apq = a[p * n + q];

                if (apq == 0) continue;

                double theta = (a[q * n + q] - a[p * n + p]) / (2 * apq);
                double t = (theta >= 0 ? 1 : -1) / (fabs(theta) + sqrt(theta * theta + 1));
                double c = 1 / sqrt(t * t + 1);
                double s = t * c;

                // columns p and q
                for (int k = 0; k < n; k++) {
                    double akp = a[k * n + p];
                    double akq = a[k * n + q];

                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }

                // rows p and q
                for (int k = 0; k < n; k++) {
                    double apk = a[p * n + k];
                    double aqk = a[q * n + k];

                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }

                for (int k = 0; k < n; k++) {
                    double vkp = vecs[k * n + p];
                    double vkq = vecs[k * n + q];

                    vecs[k * n + p] = c * vkp - s * vkq;
                    vecs[k * n + q] = s * vkp + c * vkq;
                }
            }
    }

    for (int i = 0; i < n; i++)
        vals[i] = a[i * n + i];

    // selection sort, largest first
    for (int i = 0; i < n; i++) {
        int best = i;

        for (int j = i + 1; j < n; j++)
            if (vals[j] > vals[best]) best = j;

        if (best == i) continue;

        double t = vals[i];

        vals[i] = vals[best];
        vals[best] = t;

        for (int k = 0; k < n; k++) {
            t = vecs[k * n + i];
            vecs[k * n + i] = vecs[k * n + best];
            vecs[k * n + best] = t;
        }
    }
}

//---------------------------------------------------------------------------------
// a repeatable pseudo-random number in -0.5 to 0.5 (a linear congruential generator)
//---------------------------------------------------------------------------------
static double pca_random(unsigned *seed) {
    *seed = *seed * 1103515245u + 12345u;

    return (double)(*seed >> 8) / 16777216.0 - 0.5;
}

//---------------------------------------------------------------------------------
// orthonormalize the p columns of the d x p matrix q (modified Gram-Schmidt); a column
// that is (nearly) a mix of the earlier ones becomes a fresh unit vector
//---------------------------------------------------------------------------------
static void orthonormalize(double *q, int d, int p, unsigned *seed) {
    for (int j = 0; j < p; j++) {
        for (int tries = 0; tries < 3; tries++) {
            for (int i = 0; i < j; i++) {
                double dot = 0;

                for (int r = 0; r < d; r++)
                    dot += q[r * p + i] * q[r * p + j];

                for (int r = 0; r < d; r++)
                    q[r * p + j] -= dot * q[r * p + i];
            }

            double norm = 0;

            for (int r = 0; r < d; r++)
                norm += q[r * p + j] * q[r * p + j];

            norm = sqrt(norm);

            if (norm > 1e-12) {
                for (int r = 0; r < d; r++)
                    q[r * p + j] /= norm;

                break;
            }

            for (int r = 0; r < d; r++)
                q[r * p + j] = pca_random(seed);
        }
    }
}

//---------------------------------------------------------------------------------
// The top k eigenvectors (rows of comps, k x d) and eigenvalues of the symmetric d x d
// matrix cov. Up to 64 dimensions it is solved whole by Jacobi; bigger ones by subspace
// iteration on k + 8 vectors, solving the projected matrix by Jacobi (Rayleigh-Ritz).
//---------------------------------------------------------------------------------
static int top_eigen(const double *cov, int d, int k, double *vals, double *comps) {
    int p = d <= 64 || k + 8 >= d ? d : k + 8;
    double *q = (double *)malloc((size_t)d * p * sizeof(double));
    double *z = (double *)malloc((size_t)d * p * sizeof(double));
    double *t = (double *)malloc((size_t)p * p * sizeof(double));
    double *ev = (double *)malloc((size_t)p * sizeof(double));
    double *u = (double *)malloc((size_t)p * p * sizeof(double));
    double *prev = (double *)malloc((size_t)k * sizeof(double));

    if (!q || !z || !t || !ev || !u || !prev) {
        free(q);
        free(z);
        free(t);
        free(ev);
        free(u);
        free(prev);

        return 0;
    }

    if (p == d) {
        memcpy(t, cov, (size_t)d * d * sizeof(double));

        jacobi_eigen(t, d, ev, u);

        for (int c = 0; c < k; c++)
            for (int r = 0; r < d; r++)
                comps[(size_t)c * d + r] = u[r * d + c];

        memcpy(vals, ev, (size_t)k * sizeof(double));
    } else {
        unsigned seed = 12345;

        for (int i = 0; i < d * p; i++)
            q[i] = pca_random(&seed);

        orthonormalize(q, d, p, &seed);

        for (int c = 0; c < k; c++)
            prev[c] = -1;

        for (int it = 1; it <= 1000; it++) {
            // z = cov q
            for (int r = 0; r < d; r++) {
                const double *row = cov + (size_t)r * d;
                double *zr = z + (size_t)r * p;

                for (int j = 0; j < p; j++)
                    zr[j] = 0;

                for (int l = 0; l < d; l++) {
                    double a = row[l];

                    if (a == 0) continue;

                    const double *ql = q + (size_t)l * p;

                    for (int j = 0; j < p; j++)
                        zr[j] += a * ql[j];
                }
            }

            if (it % 5 == 0 || it == 1000) {
                // Rayleigh-Ritz: t = q^T cov q = q^T z, then the components are q u
                for (int i = 0; i < p; i++)
                    for (int j = 0; j < p; j++) {
                        double s = 0;

                        for (int r = 0; r < d; r++)
                            s += q[(size_t)r * p + i] * z[(size_t)r * p + j];

                        t[i * p + j] = s;
                    }

                // symmetric, despite rounding
                for (int i = 0; i < p; i++)
                    for (int j = i + 1; j < p; j++)
                        t[i * p + j] = t[j * p + i] = (t[i * p + j] + t[j * p + i]) / 2;

                jacobi_eigen(t, p, ev, u);

                int done = 1;
                double scale = fabs(ev[0]) > 0 ? fabs(ev[0]) : 1;

                for (int c = 0; c < k; c++) {
                    if (fabs(ev[c] - prev[c]) > 1e-10 * scale) done = 0;

                    prev[c] = ev[c];
                }

                if (done || it == 1000) {
                    for (int c = 0; c < k; c++)
                        for (int r = 0; r < d; r++) {
                            double s = 0;

                            for (int j = 0; j < p; j++)
                                s += q[(size_t)r * p + j] * u[j * p + c];

                            comps[(size_t)c * d + r] = s;
                        }

                    memcpy(vals, ev, (size_t)k * sizeof(double));

                    break;
                }
            }

            memcpy(q, z, (size_t)d * p * sizeof(double));

            orthonormalize(q, d, p, &seed);
        }
    }

    // a fixed sign: each component's largest entry positive
    for (int c = 0; c < k; c++) {
        double *v = comps + (size_t)c * d;
        int big = 0;

        for (int r = 1; r < d; r++)
            if (fabs(v[r]) > fabs(v[big])) big = r;

        if (v[big] < 0)
            for (int r = 0; r < d; r++)
                v[r] = -v[r];
    }

    free(q);
    free(z);
    free(t);
    free(ev);
    free(u);
    free(prev);

    return 1;
}

#define PCA_CHUNK 512   // rows centred at a time for the covariance

//---------------------------------------------------------------------------------
// data _pca(k, maxRows) -> [projected, components, explained, mean] (main.clay's pca makes
// them a Dict): the principal components of the rows. Fitted on maxRows of them at most
// (evenly spaced; all by default), every row projected. The covariance is summed a chunk of
// centred rows at a time with libann's gemm, so the memory is cols x cols plus a chunk.
//---------------------------------------------------------------------------------
static ClayPluginValue data_pca(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    PTensor t = self_tensor(self, "pca");
    int k;
    int most;

    if (!t || !int_arg("pca", argc, argv, 0, 1, 2, 1, &k) || !int_arg("pca", argc, argv, 1, 1, 0, 0, &most)) return CLAY_NIL;

    int n = t->rows;
    int d = t->cols;

    if (k > d) {
        char msg[128];

        snprintf(msg, sizeof(msg), "pca: %d components from %d columns", k, d);

        api->raise(msg, "ArgumentError");

        return CLAY_NIL;
    }

    int m = most > 0 && most < n ? most : n;

    if (m < 2) {
        api->raise("pca needs at least 2 rows", "ArgumentError");

        return CLAY_NIL;
    }

#define FIT_ROW(i) (m == n ? (i) : (int)(((long long)(i) * n) / m))

    size_t outn = (size_t)n * k > (size_t)d * k ? (size_t)n * k : (size_t)d * k;
    double *mean = (double *)calloc((size_t)d, sizeof(double));
    double *cov = (double *)malloc((size_t)d * d * sizeof(double));
    double *comps = (double *)malloc((size_t)k * d * sizeof(double));
    double *vals = (double *)malloc((size_t)k * sizeof(double));
    real *out = (real *)malloc((outn > (size_t)d ? outn : (size_t)d) * sizeof(real));
    int chunk = m < PCA_CHUNK ? m : PCA_CHUNK;
    PTensor c = tensor_zeros(d, d);
    PTensor x = tensor_create(chunk, d);
    PTensor tail = m % chunk ? tensor_create(m % chunk, d) : NULL;
    ClayPluginValue res = CLAY_NIL;

    if (mean && cov && comps && vals && out && c && x && (tail || m % chunk == 0)) {
        for (int i = 0; i < m; i++) {
            const real *row = t->values + (size_t)FIT_ROW(i) * d;

            for (int j = 0; j < d; j++)
                mean[j] += row[j];
        }

        for (int j = 0; j < d; j++)
            mean[j] /= m;

        for (int start = 0; start < m; start += chunk) {
            int rows = m - start < chunk ? m - start : chunk;
            PTensor b = rows == chunk ? x : tail;

            for (int i = 0; i < rows; i++) {
                const real *row = t->values + (size_t)FIT_ROW(start + i) * d;
                real *dst = b->values + (size_t)i * d;

                for (int j = 0; j < d; j++)
                    dst[j] = (real)(row[j] - mean[j]);
            }

            // c += b^T b
            tensor_gemm_transA(1, b, b, 1, c);
        }

        double total = 0;

        // symmetric, despite rounding
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++) {
                double v = ((double)c->values[(size_t)i * d + j] + c->values[(size_t)j * d + i]) / 2 / (m - 1);

                cov[(size_t)i * d + j] = v;

                if (i == j) total += v;
            }

        if (top_eigen(cov, d, k, vals, comps)) {
            ClayPluginValue v;

            res = api->array_new();

            // every row, projected
            for (int i = 0; i < n; i++) {
                const real *row = t->values + (size_t)i * d;

                for (int cc = 0; cc < k; cc++) {
                    const double *p = comps + (size_t)cc * d;
                    double s = 0;

                    for (int j = 0; j < d; j++)
                        s += (row[j] - mean[j]) * p[j];

                    out[(size_t)i * k + cc] = (real)s;
                }
            }

            v = api->from_floats(out, (size_t)n * k, (size_t)k);

            api->array_push(res, v);
            api->release(v);

            for (size_t i = 0; i < (size_t)k * d; i++)
                out[i] = (real)comps[i];

            v = api->from_floats(out, (size_t)k * d, (size_t)d);

            api->array_push(res, v);
            api->release(v);

            for (int i = 0; i < k; i++)
                out[i] = (real)(total > 0 && vals[i] > 0 ? vals[i] / total : 0);

            v = api->from_floats(out, (size_t)k, 0);

            api->array_push(res, v);
            api->release(v);

            for (int j = 0; j < d; j++)
                out[j] = (real)mean[j];

            v = api->from_floats(out, (size_t)d, 0);

            api->array_push(res, v);
            api->release(v);
        }
    }

#undef FIT_ROW

    free(mean);
    free(cov);
    free(comps);
    free(vals);
    free(out);

    if (c) tensor_free(c);
    if (x) tensor_free(x);
    if (tail) tensor_free(tail);

    if (!res) api->raise("ann: out of memory", "RuntimeError");

    return res;
}

//---------------------------------------------------------------------------------
// AnnData methods
//---------------------------------------------------------------------------------
static ClayPluginMethod data_methods[] = {
    { "loadCsv", data_load_csv },   { "rows", data_rows },
    { "cols", data_cols },          { "slice", data_slice },
    { "columns", data_columns },    { "column", data_column },
    { "oneHot", data_one_hot },     { "scale", data_scale },
    { "row", data_row },            { "toArray", data_to_array },
    { "fromArray", data_from_array },{ "shuffle", data_shuffle },
    { "split", data_split },        { "mean", data_mean },
    { "std", data_std },            { "standardize", data_standardize },
    { "argmax", data_argmax },      { "min", data_min },
    { "max", data_max },            { "median", data_median },
    { "_pca", data_pca },
    { NULL, NULL }
};

//---------------------------------------------------------------------------------
// Ann methods
//---------------------------------------------------------------------------------
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
    { "weightDecay", net_set_weight_decay }, { "l1", net_set_l1 },
    { "gradientClip", net_set_clip },        { "dropout", net_set_dropout },
    { "weightInit", net_set_weight_init },   { "lossReduction", net_set_loss_reduction },
    { "inputNoise", net_set_noise },         { "lrSchedule", net_set_schedule },
    { "validation", net_set_validation },    { "earlyStopping", net_set_early },
    { "epochsTrained", net_epochs_trained }, { "bestEpoch", net_best_epoch },
    { "recordEpoch", net_record_epoch },     { "trainingRun", net_training_run },
    { "snapshot", net_snapshot },            { "restore", net_restore },
    { "history", net_history },              { "clearHistory", net_clear_history },
    { "learningCurve", net_learning_curve },
    { "predictBatch", net_predict_batch },   { "classifyBatch", net_classify_batch },
    { "confusionMatrix", net_confusion },    { "backendInfo", net_backend_info },
    { "threads", net_threads },              { "version", net_version },
    { "conv2d", net_conv2d },                { "depthwiseConv2d", net_depthwise },
    { "conv1d", net_conv1d },                { "maxPool2d", net_maxpool2d },
    { "maxPool1d", net_maxpool1d },          { "flatten", net_flatten },
    { "globalAvgPool", net_global_avg },     { "globalMaxPool", net_global_max },
    { "upsample2d", net_upsample },          { "reshape", net_reshape },
    { "batchNorm", net_batchnorm },          { "embedding", net_embedding },
    { "layerInfo", net_layer_info },         { "layerCount", net_layer_count },
    { "weights", net_weights },              { "biases", net_biases },
    { "activations", net_activations },
    { "trainBegin", net_train_begin },       { "trainStep", net_train_step },
    { "trainEnd", net_train_end },           { "trainEpoch", net_train_epoch },
    { "saveBinary", net_save_binary },       { "loadBinary", net_load_binary },
    { "exportOnnx", net_export_onnx },       { "importOnnx", net_import_onnx },
    { "exportC", net_export_c },             { "exportJs", net_export_js },
    { "diagram", net_diagram },              { "quantize", net_quantize },
    { NULL, NULL }
};

//---------------------------------------------------------------------------------
// Plugin registration
//---------------------------------------------------------------------------------
CLAY_PLUGIN_EXPORT void ann_plugin_init(const ClayPluginAPI *clay_api) {
    // save the API struct ptr
    api = clay_api;

    if (api->api_version < 2) return;      // needs arrays and float buffers

    // register the Ann prototype
    ann_set_error_log_callback(quiet_errors);
    ClayPluginType def = { "AnnNet", methods, net_free, sizeof(void *) };
    api->register_type(&def);

    // register the AnnData prototype
    ClayPluginType data_def = { "AnnData", data_methods, data_free, sizeof(void *) };
    data_proto = api->register_type(&data_def);
    api->set_slot(data_proto, "_annData", api->make_bool(1));

    // register the AnnQuant prototype
    ClayPluginType quant_def = { "AnnQuant", quant_methods, quant_free, sizeof(void *) };
    quant_proto = api->register_type(&quant_def);
}
