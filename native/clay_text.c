/*
 * clay_text.c - text preprocessing for the ann brick: the AnnTokenizer prototype
 *
 * Turns texts into the rows of token ids an embedding layer reads, as pyann's
 * pyann.text does (Tokenizer, pad_sequences, strip_html, split_contractions,
 * load_word_vectors), so a tokenizer saved by one loads in the other and gives
 * the same ids. libann has no text support, so it all lives here, in C, since a
 * corpus such as IMDB's 50,000 reviews is millions of words.
 *
 * Texts are UTF-8. Python works on Unicode code points; this file does too where
 * it matters (character tokens, filters, whitespace), and lower-cases ASCII,
 * Latin-1, Latin Extended-A, Greek and Cyrillic letters (Python lower-cases every
 * script). strip_html decodes numeric entities and the common named ones (all of
 * HTML 4's), not all 2,231 of HTML5.
 *
 * Every method checks its arguments before it allocates, and scratch memory is
 * owned by the tokenizer object, so a Clay error (which unwinds past C frames)
 * can't leak it.
 */
#include "clay_plugin.h"
#include "libann/tensor.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const ClayPluginAPI *api;
static ClayPluginObject tok_proto;
static ClayPluginObject data_proto;     /* AnnData, from clay_ann.c */

//---------------------------------------------------------------------------------
// --- growable byte buffers ----------------------------------------------
//---------------------------------------------------------------------------------
typedef struct {
    char *p;
    size_t n, cap;
    int oom;
} Buf;

static int buf_reserve(Buf *b, size_t extra) {
    if (b->oom) return 0;
    if (b->n + extra + 1 <= b->cap) return 1;

    size_t cap = b->cap ? b->cap : 256;
    while (cap < b->n + extra + 1) cap *= 2;

    char *p = (char *)realloc(b->p, cap);
    if (!p) { b->oom = 1; return 0; }

    b->p = p;
    b->cap = cap;
    return 1;
}

static void buf_put(Buf *b, const char *s, size_t n) {
    if (!buf_reserve(b, n)) return;
    memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = 0;
}

static void buf_putc(Buf *b, char c) { buf_put(b, &c, 1); }

static void buf_puts(Buf *b, const char *s) { buf_put(b, s, strlen(s)); }

static void buf_free(Buf *b) {
    free(b->p);
    b->p = NULL;
    b->n = b->cap = 0;
    b->oom = 0;
}

//---------------------------------------------------------------------------------
// --- UTF-8 ---------------------------------------------------------------
//---------------------------------------------------------------------------------

// the code point at s[*i] (advancing *i past it); a byte that doesn't start a valid
// sequence counts as one code point of its own value
static uint32_t u8_next(const char *str, size_t n, size_t *i) {
    const unsigned char *s = (const unsigned char *)str;
    size_t k = *i;
    unsigned c = s[k];

    if (c < 0x80) { *i = k + 1; return c; }

    int len = (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 0;

    if (len && k + len <= n) {
        uint32_t cp = c & (len == 2 ? 0x1F : len == 3 ? 0x0F : 0x07);
        int ok = 1;

        for (int j = 1; j < len; j++) {
            if ((s[k + j] & 0xC0) != 0x80) { ok = 0; break; }
            cp = (cp << 6) | (s[k + j] & 0x3F);
        }

        if (ok && cp >= (len == 2 ? 0x80u : len == 3 ? 0x800u : 0x10000u) && cp <= 0x10FFFF) {
            *i = k + len;
            return cp;
        }
    }

    *i = k + 1;
    return c;
}

// the code point that ends just before s[i] (0 at the start)
static uint32_t u8_prev(const char *s, size_t i) {
    if (i == 0) return 0;

    size_t k = i - 1;
    int back = 0;

    while (k > 0 && back < 3 && (((unsigned char)s[k]) & 0xC0) == 0x80) { k--; back++; }

    size_t j = k;
    uint32_t cp = u8_next(s, i, &j);

    if (j == i) return cp;
    return (unsigned char)s[i - 1];     // not a whole sequence: the byte itself
}

// cp as UTF-8 in t: its length
static int u8_encode(uint32_t cp, char *t) {
    if (cp < 0x80) { t[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        t[0] = (char)(0xC0 | (cp >> 6)); t[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        t[0] = (char)(0xE0 | (cp >> 12)); t[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        t[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    t[0] = (char)(0xF0 | (cp >> 18)); t[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    t[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); t[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

static void buf_put_cp(Buf *b, uint32_t cp) {
    char t[4];
    buf_put(b, t, (size_t)u8_encode(cp, t));
}

// Python's str.isspace (and so str.split() and the regex \s)
static int is_space_cp(uint32_t c) {
    return c == ' ' || (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x1F) || c == 0x85 ||
           c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 ||
           c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

// the regex \w: letters, digits and underscore. Outside ASCII an approximation:
// everything but spaces, Latin-1's symbols and the punctuation blocks
static int is_word_cp(uint32_t c) {
    if (c < 0x80)
        return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
    if (c < 0xA0) return 0;
    if (c < 0xC0)
        return c == 0xAA || c == 0xB2 || c == 0xB3 || c == 0xB5 || c == 0xB9 || c == 0xBA ||
               c == 0xBC || c == 0xBD || c == 0xBE;
    if (c == 0xD7 || c == 0xF7) return 0;
    if (is_space_cp(c)) return 0;
    if (c >= 0x2000 && c <= 0x2BFF) return 0;       // punctuation, symbols, arrows, ...
    if (c >= 0x3000 && c <= 0x303F) return 0;       // CJK punctuation
    if (c >= 0xFE30 && c <= 0xFE4F) return 0;
    if (c == 0xFFFD) return 0;
    return 1;
}

static int word_at(const char *s, size_t n, size_t i) {
    if (i >= n) return 0;
    return is_word_cp(u8_next(s, n, &i));
}

static int word_before(const char *s, size_t i) {
    return i > 0 && is_word_cp(u8_prev(s, i));
}

// lower case for the letters this file knows (see the top); the result always
// takes as many UTF-8 bytes as the letter, so lowering works in place
static uint32_t lower_cp(uint32_t c) {
    if (c < 0x80) return (c >= 'A' && c <= 'Z') ? c + 32 : c;
    if (c >= 0xC0 && c <= 0xDE && c != 0xD7) return c + 32;
    if (c >= 0x100 && c <= 0x17F) {
        if (c == 0x130) return c;                   // İ lowers to two code points
        if (c == 0x178) return 0xFF;
        if ((c >= 0x139 && c <= 0x148) || (c >= 0x179 && c <= 0x17E)) return (c & 1) ? c + 1 : c;
        if (c == 0x138 || c == 0x149 || c == 0x17F) return c;
        return (c & 1) ? c : c + 1;
    }
    if (c >= 0x391 && c <= 0x3AB && c != 0x3A2) return c + 32;
    if (c == 0x386) return 0x3AC;
    if (c >= 0x388 && c <= 0x38A) return c + 37;
    if (c == 0x38C) return 0x3CC;
    if (c == 0x38E || c == 0x38F) return c + 63;
    if (c >= 0x410 && c <= 0x42F) return c + 32;
    if (c >= 0x400 && c <= 0x40F) return c + 80;
    return c;
}

static int u8_len(uint32_t cp) {
    return cp < 0x80 ? 1 : cp < 0x800 ? 2 : cp < 0x10000 ? 3 : 4;
}

static void lower_inplace(char *s, size_t n) {
    size_t i = 0;

    while (i < n) {
        size_t start = i;

        if ((unsigned char)s[i] < 0x80) {
            if (s[i] >= 'A' && s[i] <= 'Z') s[i] += 32;
            i++;
            continue;
        }

        uint32_t cp = u8_next(s, n, &i);
        uint32_t lo = lower_cp(cp);

        if (lo != cp && (size_t)u8_len(lo) == i - start)
            u8_encode(lo, s + start);
    }
}

static int ci_eq(const char *s, size_t n, size_t i, const char *lit) {
    size_t k = strlen(lit);

    if (i + k > n) return 0;
    for (size_t j = 0; j < k; j++) {
        char c = s[i + j];
        if (c >= 'A' && c <= 'Z') c += 32;
        if (c != lit[j]) return 0;
    }
    return 1;
}

static int is_alpha(char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }

//---------------------------------------------------------------------------------
// --- strip_html ------------------------------------------------------------
//---------------------------------------------------------------------------------

// <script ...>...</script> and <style ...>...</style> at s[i] (case-insensitive):
// its end, or 0
static size_t match_script(const char *s, size_t n, size_t i) {
    static const char *names[] = { "script", "style" };

    for (int w = 0; w < 2; w++) {
        const char *name = names[w];
        size_t len = strlen(name);

        if (!ci_eq(s, n, i + 1, name)) continue;

        size_t j = i + 1 + len;
        if (word_at(s, n, j)) continue;             // \b

        const char *gt = (const char *)memchr(s + j, '>', n - j);
        if (!gt) continue;

        // the first </name\s*> after it
        for (size_t k = (size_t)(gt - s) + 1; k + 2 + len <= n; k++) {
            if (s[k] != '<' || s[k + 1] != '/' || !ci_eq(s, n, k + 2, name)) continue;

            size_t m = k + 2 + len;
            while (m < n) {
                size_t m2 = m;
                if (!is_space_cp(u8_next(s, n, &m2))) break;
                m = m2;
            }
            if (m < n && s[m] == '>') return m + 1;
        }
    }

    return 0;
}

// <!-- ... --> at s[i]: its end, or 0
static size_t match_comment(const char *s, size_t n, size_t i) {
    if (i + 4 > n || memcmp(s + i, "<!--", 4) != 0) return 0;

    for (size_t k = i + 4; k + 3 <= n; k++)
        if (s[k] == '-' && s[k + 1] == '-' && s[k + 2] == '>') return k + 3;

    return 0;
}

// a tag (or <!DOCTYPE ...>) at s[i]: its end, or 0. Like
// <(?:/?[A-Za-z]|[!?][A-Za-z])[^>]*>, so "3 < 4 and 5 > 2" is no tag
static size_t match_tag(const char *s, size_t n, size_t i) {
    size_t j = i + 1;

    if (j < n && s[j] == '/' && j + 1 < n && is_alpha(s[j + 1])) j += 2;
    else if (j < n && is_alpha(s[j])) j += 1;
    else if (j < n && (s[j] == '!' || s[j] == '?') && j + 1 < n && is_alpha(s[j + 1])) j += 2;
    else return 0;

    const char *gt = (const char *)memchr(s + j, '>', n - j);
    return gt ? (size_t)(gt - s) + 1 : 0;
}

typedef size_t (*MatchFn)(const char *, size_t, size_t);

// replace every match (found at '<') with a space
static void replace_matches(const char *s, size_t n, Buf *out, MatchFn match) {
    size_t i = 0;

    out->n = 0;
    buf_reserve(out, n);

    while (i < n) {
        const char *lt = (const char *)memchr(s + i, '<', n - i);
        size_t at = lt ? (size_t)(lt - s) : n;

        buf_put(out, s + i, at - i);
        if (at >= n) break;

        size_t end = match(s, n, at);
        if (end) { buf_putc(out, ' '); i = end; }
        else { buf_putc(out, '<'); i = at + 1; }
    }
}

// HTML's named characters: HTML 4's (the Latin-1 ones also work without their ';', as
// in Python's html.unescape) and a few common others
typedef struct { const char *name; uint32_t cp; int legacy; } Entity;

static const char *latin1_names[96] = {
    "nbsp", "iexcl", "cent", "pound", "curren", "yen", "brvbar", "sect", "uml", "copy", "ordf",
    "laquo", "not", "shy", "reg", "macr", "deg", "plusmn", "sup2", "sup3", "acute", "micro",
    "para", "middot", "cedil", "sup1", "ordm", "raquo", "frac14", "frac12", "frac34", "iquest",
    "Agrave", "Aacute", "Acirc", "Atilde", "Auml", "Aring", "AElig", "Ccedil", "Egrave",
    "Eacute", "Ecirc", "Euml", "Igrave", "Iacute", "Icirc", "Iuml", "ETH", "Ntilde", "Ograve",
    "Oacute", "Ocirc", "Otilde", "Ouml", "times", "Oslash", "Ugrave", "Uacute", "Ucirc", "Uuml",
    "Yacute", "THORN", "szlig", "agrave", "aacute", "acirc", "atilde", "auml", "aring", "aelig",
    "ccedil", "egrave", "eacute", "ecirc", "euml", "igrave", "iacute", "icirc", "iuml", "eth",
    "ntilde", "ograve", "oacute", "ocirc", "otilde", "ouml", "divide", "oslash", "ugrave",
    "uacute", "ucirc", "uuml", "yacute", "thorn", "yuml"
};

static const Entity entities[] = {
    {"amp", 38, 1}, {"lt", 60, 1}, {"gt", 62, 1}, {"quot", 34, 1},
    {"AMP", 38, 1}, {"LT", 60, 1}, {"GT", 62, 1}, {"QUOT", 34, 1}, {"COPY", 169, 1}, {"REG", 174, 1},
    {"apos", 39, 0}, {"OElig", 338, 0}, {"oelig", 339, 0}, {"Scaron", 352, 0}, {"scaron", 353, 0},
    {"Yuml", 376, 0}, {"fnof", 402, 0}, {"circ", 710, 0}, {"tilde", 732, 0},
    {"ensp", 8194, 0}, {"emsp", 8195, 0}, {"thinsp", 8201, 0}, {"zwnj", 8204, 0},
    {"zwj", 8205, 0}, {"lrm", 8206, 0}, {"rlm", 8207, 0}, {"ndash", 8211, 0},
    {"mdash", 8212, 0}, {"lsquo", 8216, 0}, {"rsquo", 8217, 0}, {"sbquo", 8218, 0},
    {"ldquo", 8220, 0}, {"rdquo", 8221, 0}, {"bdquo", 8222, 0}, {"dagger", 8224, 0},
    {"Dagger", 8225, 0}, {"bull", 8226, 0}, {"hellip", 8230, 0}, {"permil", 8240, 0},
    {"prime", 8242, 0}, {"Prime", 8243, 0}, {"lsaquo", 8249, 0}, {"rsaquo", 8250, 0},
    {"oline", 8254, 0}, {"frasl", 8260, 0}, {"euro", 8364, 0}, {"image", 8465, 0},
    {"weierp", 8472, 0}, {"real", 8476, 0}, {"trade", 8482, 0}, {"alefsym", 8501, 0},
    {"larr", 8592, 0}, {"uarr", 8593, 0}, {"rarr", 8594, 0}, {"darr", 8595, 0},
    {"harr", 8596, 0}, {"crarr", 8629, 0}, {"lArr", 8656, 0}, {"uArr", 8657, 0},
    {"rArr", 8658, 0}, {"dArr", 8659, 0}, {"hArr", 8660, 0}, {"forall", 8704, 0},
    {"part", 8706, 0}, {"exist", 8707, 0}, {"empty", 8709, 0}, {"nabla", 8711, 0},
    {"isin", 8712, 0}, {"notin", 8713, 0}, {"ni", 8715, 0}, {"prod", 8719, 0},
    {"sum", 8721, 0}, {"minus", 8722, 0}, {"lowast", 8727, 0}, {"radic", 8730, 0},
    {"prop", 8733, 0}, {"infin", 8734, 0}, {"ang", 8736, 0}, {"and", 8743, 0},
    {"or", 8744, 0}, {"cap", 8745, 0}, {"cup", 8746, 0}, {"int", 8747, 0},
    {"there4", 8756, 0}, {"sim", 8764, 0}, {"cong", 8773, 0}, {"asymp", 8776, 0},
    {"ne", 8800, 0}, {"equiv", 8801, 0}, {"le", 8804, 0}, {"ge", 8805, 0},
    {"sub", 8834, 0}, {"sup", 8835, 0}, {"nsub", 8836, 0}, {"sube", 8838, 0},
    {"supe", 8839, 0}, {"oplus", 8853, 0}, {"otimes", 8855, 0}, {"perp", 8869, 0},
    {"sdot", 8901, 0}, {"lceil", 8968, 0}, {"rceil", 8969, 0}, {"lfloor", 8970, 0},
    {"rfloor", 8971, 0}, {"lang", 9001, 0}, {"rang", 9002, 0}, {"loz", 9674, 0},
    {"spades", 9824, 0}, {"clubs", 9827, 0}, {"hearts", 9829, 0}, {"diams", 9830, 0},
    {"Alpha", 913, 0}, {"Beta", 914, 0}, {"Gamma", 915, 0}, {"Delta", 916, 0},
    {"Epsilon", 917, 0}, {"Zeta", 918, 0}, {"Eta", 919, 0}, {"Theta", 920, 0},
    {"Iota", 921, 0}, {"Kappa", 922, 0}, {"Lambda", 923, 0}, {"Mu", 924, 0},
    {"Nu", 925, 0}, {"Xi", 926, 0}, {"Omicron", 927, 0}, {"Pi", 928, 0},
    {"Rho", 929, 0}, {"Sigma", 931, 0}, {"Tau", 932, 0}, {"Upsilon", 933, 0},
    {"Phi", 934, 0}, {"Chi", 935, 0}, {"Psi", 936, 0}, {"Omega", 937, 0},
    {"alpha", 945, 0}, {"beta", 946, 0}, {"gamma", 947, 0}, {"delta", 948, 0},
    {"epsilon", 949, 0}, {"zeta", 950, 0}, {"eta", 951, 0}, {"theta", 952, 0},
    {"iota", 953, 0}, {"kappa", 954, 0}, {"lambda", 955, 0}, {"mu", 956, 0},
    {"nu", 957, 0}, {"xi", 958, 0}, {"omicron", 959, 0}, {"pi", 960, 0},
    {"rho", 961, 0}, {"sigmaf", 962, 0}, {"sigma", 963, 0}, {"tau", 964, 0},
    {"upsilon", 965, 0}, {"phi", 966, 0}, {"chi", 967, 0}, {"psi", 968, 0},
    {"omega", 969, 0}, {"thetasym", 977, 0}, {"upsih", 978, 0}, {"piv", 982, 0},
};

// the code point of the entity name s[0..n) (with its ';' when semi), or 0
static uint32_t entity_cp(const char *s, size_t n, int semi) {
    for (int k = 0; k < 96; k++)
        if (strlen(latin1_names[k]) == n && memcmp(latin1_names[k], s, n) == 0) return 160 + (uint32_t)k;

    for (size_t k = 0; k < sizeof(entities) / sizeof(entities[0]); k++)
        if ((semi || entities[k].legacy) && strlen(entities[k].name) == n &&
            memcmp(entities[k].name, s, n) == 0)
            return entities[k].cp;

    return 0;
}

// a numeric character reference's text, as Python's html.unescape gives it
static void put_charref(Buf *out, uint32_t num, int too_big) {
    static const uint16_t cp1252[32] = {
        0x20AC, 0x81, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021, 0x02C6, 0x2030, 0x0160,
        0x2039, 0x0152, 0x8D, 0x017D, 0x8F, 0x90, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022,
        0x2013, 0x2014, 0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0x9D, 0x017E, 0x0178
    };

    if (too_big || (num >= 0xD800 && num <= 0xDFFF) || num > 0x10FFFF) { buf_put_cp(out, 0xFFFD); return; }
    if (num == 0) { buf_put_cp(out, 0xFFFD); return; }
    if (num == 0x0D) { buf_putc(out, '\r'); return; }
    if (num >= 0x80 && num <= 0x9F) { buf_put_cp(out, cp1252[num - 0x80]); return; }
    if ((num >= 0x1 && num <= 0x8) || num == 0xB || (num >= 0xE && num <= 0x1F) ||
        (num >= 0x7F && num <= 0x9F) || (num >= 0xFDD0 && num <= 0xFDEF) || (num & 0xFFFE) == 0xFFFE)
        return;     // invalid code points vanish

    buf_put_cp(out, num);
}

static int is_hex(char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

static int hex_val(char c) {
    return c <= '9' ? c - '0' : (c | 32) - 'a' + 10;
}

// Python's html.unescape
static void unescape(const char *s, size_t n, Buf *out) {
    size_t i = 0;

    out->n = 0;
    buf_reserve(out, n);

    while (i < n) {
        const char *amp = (const char *)memchr(s + i, '&', n - i);
        size_t at = amp ? (size_t)(amp - s) : n;

        buf_put(out, s + i, at - i);
        if (at >= n) break;

        size_t j = at + 1;

        if (j < n && s[j] == '#') {
            int hex = j + 2 < n && (s[j + 1] == 'x' || s[j + 1] == 'X') && is_hex(s[j + 2]);
            size_t k = hex ? j + 2 : j + 1;

            if (!hex && !(k < n && s[k] >= '0' && s[k] <= '9')) {
                buf_putc(out, '&');
                i = j;
                continue;
            }

            uint32_t num = 0;
            int big = 0;

            while (k < n && (hex ? is_hex(s[k]) : (s[k] >= '0' && s[k] <= '9'))) {
                if (!big) {
                    num = num * (hex ? 16 : 10) + (uint32_t)(hex ? hex_val(s[k]) : s[k] - '0');
                    if (num > 0x10FFFF) big = 1;
                }
                k++;
            }
            if (k < n && s[k] == ';') k++;

            put_charref(out, num, big);
            i = k;
            continue;
        }

        // a name: up to 32 characters that aren't \t \n \f space < & # ;
        size_t k = j;
        int chars = 0;

        while (k < n && chars < 32) {
            char c = s[k];
            if (c == '\t' || c == '\n' || c == '\f' || c == ' ' || c == '<' || c == '&' || c == '#' || c == ';')
                break;
            u8_next(s, n, &k);
            chars++;
        }

        if (k == j) { buf_putc(out, '&'); i = j; continue; }

        int semi = k < n && s[k] == ';';
        size_t name_len = k - j;
        size_t end = k + (semi ? 1 : 0);
        uint32_t cp = entity_cp(s + j, name_len, semi);

        if (cp) { buf_put_cp(out, cp); i = end; continue; }

        // else the longest prefix that is an entity without its ';'
        size_t whole = end - j, x;
        for (x = whole - 1; x >= 2; x--) {
            cp = entity_cp(s + j, x, 0);
            if (cp) break;
        }

        if (x >= 2 && cp) {
            buf_put_cp(out, cp);
            buf_put(out, s + j + x, whole - x);
        }
        else {
            buf_putc(out, '&');
            buf_put(out, s + j, whole);
        }
        i = end;
    }
}

// strip_html: scripts and styles, comments and tags become spaces, then entities are decoded
static void strip_html(const char *s, size_t n, Buf *out, Buf *t1, Buf *t2) {
    replace_matches(s, n, t1, match_script);
    replace_matches(t1->p ? t1->p : "", t1->n, t2, match_comment);
    replace_matches(t2->p ? t2->p : "", t2->n, t1, match_tag);
    unescape(t1->p ? t1->p : "", t1->n, out);
    if (t1->oom || t2->oom) out->oom = 1;
}

//---------------------------------------------------------------------------------
// --- split_contractions ------------------------------------------------------
//---------------------------------------------------------------------------------

// an apostrophe (' or ’) at s[i]: its length in bytes, or 0
static size_t quote_at(const char *s, size_t n, size_t i) {
    if (i < n && s[i] == '\'') return 1;
    if (i + 3 <= n && (unsigned char)s[i] == 0xE2 && (unsigned char)s[i + 1] == 0x80 &&
        (unsigned char)s[i + 2] == 0x99) return 3;
    return 0;
}

// (?<!\w)['’]+|['’]+(?!\w) -> a space
static void edge_quotes(const char *s, size_t n, Buf *out) {
    size_t i = 0;

    out->n = 0;
    buf_reserve(out, n);

    while (i < n) {
        size_t q = quote_at(s, n, i);

        if (!q) {
            size_t j = i;
            u8_next(s, n, &j);
            buf_put(out, s + i, j - i);
            i = j;
            continue;
        }

        // the run of quotes, and where its last one starts
        size_t k = i, last = i, count = 0;
        while ((q = quote_at(s, n, k)) != 0) { last = k; k += q; count++; }

        if (!word_before(s, i) || !word_at(s, n, k)) { buf_putc(out, ' '); i = k; }
        else if (count >= 2) { buf_putc(out, ' '); i = last; }     // all but the last
        else { buf_put(out, s + i, k - i); i = k; }
    }
}

// (?<=[A-Za-z])n['’]t\b -> " n't" (any case)
static void split_nt(const char *s, size_t n, Buf *out) {
    size_t i = 0;

    out->n = 0;
    buf_reserve(out, n + n / 4);

    while (i < n) {
        if ((s[i] == 'n' || s[i] == 'N') && i > 0 && is_alpha(s[i - 1])) {
            size_t q = quote_at(s, n, i + 1);
            size_t t = i + 1 + q;

            if (q && t < n && (s[t] == 't' || s[t] == 'T') && !word_at(s, n, t + 1)) {
                buf_puts(out, " n't");
                i = t + 1;
                continue;
            }
        }
        buf_putc(out, s[i]);
        i++;
    }
}

// (?<=\w)['’](s|re|ve|ll|d|m)\b -> " '" + the clitic (any case)
static void split_clitics(const char *s, size_t n, Buf *out) {
    static const char *clitics[] = { "s", "re", "ve", "ll", "d", "m" };
    size_t i = 0;

    out->n = 0;
    buf_reserve(out, n + n / 4);

    while (i < n) {
        size_t q = quote_at(s, n, i);

        if (q && word_before(s, i)) {
            size_t c = i + q;
            int found = 0;

            for (int k = 0; k < 6 && !found; k++) {
                size_t len = strlen(clitics[k]);

                if (ci_eq(s, n, c, clitics[k]) && !word_at(s, n, c + len)) {
                    buf_puts(out, " '");
                    buf_put(out, s + c, len);
                    i = c + len;
                    found = 1;
                }
            }
            if (found) continue;
        }

        if (q) { buf_put(out, s + i, q); i += q; }
        else { buf_putc(out, s[i]); i++; }
    }
}

static void split_contractions(const char *s, size_t n, Buf *out, Buf *t1, Buf *t2) {
    edge_quotes(s, n, t1);
    split_nt(t1->p ? t1->p : "", t1->n, t2);
    split_clitics(t2->p ? t2->p : "", t2->n, out);
    if (t1->oom || t2->oom) out->oom = 1;
}

//---------------------------------------------------------------------------------
// --- the tokenizer -----------------------------------------------------------
//---------------------------------------------------------------------------------

enum { STEP_HTML = 1, STEP_CONTRACTIONS = 2 };
#define MAX_STEPS 16

typedef struct {
    char *word;             // NUL-terminated copy
    uint32_t len;
    int64_t count;
    int32_t id;             // 0: not in the vocabulary
    int32_t counted;        // seen in fitted texts (so in word_counts)
} TokEntry;

typedef struct {
    // settings
    int num_words;          // 0: no limit
    char *oov;              // NULL: none
    int lower, char_level;
    char *filters;
    size_t filters_len;
    unsigned char ascii_filter[128];
    uint32_t *wide_filters;
    int n_wide;
    char *split;            // NULL: any whitespace
    size_t split_len;
    int steps[MAX_STEPS], n_steps;
    int custom;             // a preprocess of the caller's (Clay applies it)
    int single;             // the preprocess was one name, not a list

    // the vocabulary
    TokEntry *e;
    size_t n, cap;
    int32_t *slots;         // hash table: entry index, or -1
    size_t nslots;
    size_t *order;          // counted entries in the order first seen
    size_t n_order, cap_order;
    int32_t *by_id;         // id -> entry index, or -1
    size_t n_by_id;
    size_t n_ids;           // len(word_index)
    int32_t oov_id;

    // scratch
    Buf work, t1, t2, t3;
    int32_t *ids;
    size_t n_ids_buf, cap_ids_buf;
    size_t *offs;
    size_t cap_offs;
    float *floats;
    size_t cap_floats;
    int oom;
} Tok;

static void tok_clear_vocab(Tok *t) {
    for (size_t i = 0; i < t->n; i++) free(t->e[i].word);
    free(t->e); free(t->slots); free(t->order); free(t->by_id);
    t->e = NULL; t->slots = NULL; t->order = NULL; t->by_id = NULL;
    t->n = t->cap = t->nslots = t->n_order = t->cap_order = t->n_by_id = t->n_ids = 0;
    t->oov_id = 0;
}

static void tok_free(void *data) {
    Tok *t = (Tok *)data;
    if (!t) return;

    tok_clear_vocab(t);
    free(t->oov); free(t->filters); free(t->wide_filters); free(t->split);
    buf_free(&t->work); buf_free(&t->t1); buf_free(&t->t2); buf_free(&t->t3);
    free(t->ids); free(t->offs); free(t->floats);
    free(t);
}

static uint64_t hash_word(const char *s, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 1099511628211ULL; }
    return h;
}

static int64_t tok_find(const Tok *t, const char *w, size_t len) {
    if (!t->nslots) return -1;

    size_t mask = t->nslots - 1, h = (size_t)hash_word(w, len) & mask;

    while (t->slots[h] >= 0) {
        const TokEntry *e = &t->e[t->slots[h]];
        if (e->len == len && memcmp(e->word, w, len) == 0) return t->slots[h];
        h = (h + 1) & mask;
    }
    return -1;
}

static int tok_grow_slots(Tok *t) {
    size_t ns = t->nslots ? t->nslots * 2 : 1024;
    int32_t *slots = (int32_t *)malloc(ns * sizeof(int32_t));
    if (!slots) return 0;

    for (size_t i = 0; i < ns; i++) slots[i] = -1;
    for (size_t i = 0; i < t->n; i++) {
        size_t h = (size_t)hash_word(t->e[i].word, t->e[i].len) & (ns - 1);
        while (slots[h] >= 0) h = (h + 1) & (ns - 1);
        slots[h] = (int32_t)i;
    }

    free(t->slots);
    t->slots = slots;
    t->nslots = ns;
    return 1;
}

// the entry for w, added (with count 0) when new; -1 when out of memory
static int64_t tok_insert(Tok *t, const char *w, size_t len) {
    int64_t f = tok_find(t, w, len);
    if (f >= 0) return f;

    if (len > 0xFFFFFFFFu || t->n >= 0x7FFFFFFF) return -1;
    if ((t->n + 1) * 2 > t->nslots && !tok_grow_slots(t)) return -1;
    if (t->n == t->cap) {
        size_t cap = t->cap ? t->cap * 2 : 1024;
        TokEntry *e = (TokEntry *)realloc(t->e, cap * sizeof(TokEntry));
        if (!e) return -1;
        t->e = e;
        t->cap = cap;
    }

    char *copy = (char *)malloc(len + 1);
    if (!copy) return -1;
    memcpy(copy, w, len);
    copy[len] = 0;

    TokEntry *e = &t->e[t->n];
    e->word = copy; e->len = (uint32_t)len; e->count = 0; e->id = 0; e->counted = 0;

    size_t mask = t->nslots - 1, h = (size_t)hash_word(w, len) & mask;
    while (t->slots[h] >= 0) h = (h + 1) & mask;
    t->slots[h] = (int32_t)t->n;

    return (int64_t)t->n++;
}

static int tok_mark_counted(Tok *t, size_t idx) {
    if (t->e[idx].counted) return 1;
    if (t->n_order == t->cap_order) {
        size_t cap = t->cap_order ? t->cap_order * 2 : 1024;
        size_t *o = (size_t *)realloc(t->order, cap * sizeof(size_t));
        if (!o) return 0;
        t->order = o;
        t->cap_order = cap;
    }
    t->order[t->n_order++] = idx;
    t->e[idx].counted = 1;
    return 1;
}

// by_id from the entries' ids
static int tok_index_ids(Tok *t) {
    int32_t max = 0;
    for (size_t i = 0; i < t->n; i++) if (t->e[i].id > max) max = t->e[i].id;

    int32_t *b = (int32_t *)malloc(((size_t)max + 1) * sizeof(int32_t));
    if (!b) return 0;

    for (int32_t i = 0; i <= max; i++) b[i] = -1;
    t->n_ids = 0;
    for (size_t i = 0; i < t->n; i++)
        if (t->e[i].id > 0) { b[t->e[i].id] = (int32_t)i; t->n_ids++; }

    free(t->by_id);
    t->by_id = b;
    t->n_by_id = (size_t)max + 1;
    return 1;
}

typedef struct { int64_t count; size_t pos; } Rank;

static int rank_cmp(const void *a, const void *b) {
    const Rank *x = (const Rank *)a, *y = (const Rank *)b;
    if (x->count != y->count) return x->count > y->count ? -1 : 1;
    return x->pos < y->pos ? -1 : x->pos > y->pos;
}

// word_index from the counts: the most frequent first (ties in the order first
// seen), from 1, or from 2 after the oov token
static int tok_rebuild(Tok *t) {
    int64_t oov_idx = -1;

    if (t->oov) {
        oov_idx = tok_insert(t, t->oov, strlen(t->oov));
        if (oov_idx < 0) return 0;
    }

    Rank *r = (Rank *)malloc((t->n_order ? t->n_order : 1) * sizeof(Rank));
    if (!r) return 0;

    for (size_t p = 0; p < t->n_order; p++) { r[p].count = t->e[t->order[p]].count; r[p].pos = p; }
    qsort(r, t->n_order, sizeof(Rank), rank_cmp);

    for (size_t i = 0; i < t->n; i++) t->e[i].id = 0;

    int32_t next = 1;
    if (oov_idx >= 0) { t->e[oov_idx].id = 1; next = 2; }

    for (size_t p = 0; p < t->n_order; p++) {
        size_t idx = t->order[r[p].pos];
        if ((int64_t)idx == oov_idx) continue;
        t->e[idx].id = next++;
    }
    free(r);

    t->oov_id = oov_idx >= 0 ? 1 : 0;
    return tok_index_ids(t);
}

// a word's id as texts_to_sequences gives it: 0 drops the word
static int32_t tok_id(const Tok *t, const char *w, size_t len) {
    int64_t idx = tok_find(t, w, len);

    if (idx >= 0) {
        int32_t id = t->e[idx].id;
        if (id > 0 && (t->num_words == 0 || id < t->num_words)) return id;
    }
    return t->oov_id;
}

static size_t tok_vocab_size(const Tok *t) {
    size_t size = t->n_ids + 1;
    return (t->num_words && (size_t)t->num_words < size) ? (size_t)t->num_words : size;
}

// t->work = the text after the preprocess steps and lower-casing
static void tok_prepare(Tok *t, const char *text, size_t len) {
    t->work.n = 0;
    buf_put(&t->work, text, len);

    for (int k = 0; k < t->n_steps; k++) {
        if (t->steps[k] == STEP_HTML)
            strip_html(t->work.p ? t->work.p : "", t->work.n, &t->t3, &t->t1, &t->t2);
        else
            split_contractions(t->work.p ? t->work.p : "", t->work.n, &t->t3, &t->t1, &t->t2);

        Buf tmp = t->work; t->work = t->t3; t->t3 = tmp;
    }

    if (t->work.oom || t->t1.oom || t->t2.oom || t->t3.oom) t->oom = 1;
    if (!t->lower || !t->work.p) return;

    lower_inplace(t->work.p, t->work.n);

    // İ (U+0130) lowers to two code points, i and a combining dot, as in Python
    if (memchr(t->work.p, 0xC4, t->work.n)) {
        const char *s = t->work.p;
        size_t n = t->work.n, i = 0;
        int any = 0;

        for (size_t k = 0; k + 1 < n; k++)
            if ((unsigned char)s[k] == 0xC4 && (unsigned char)s[k + 1] == 0xB0) { any = 1; break; }
        if (!any) return;

        t->t3.n = 0;
        while (i < n) {
            if (i + 1 < n && (unsigned char)s[i] == 0xC4 && (unsigned char)s[i + 1] == 0xB0) {
                buf_put(&t->t3, "i\xCC\x87", 3);
                i += 2;
            }
            else buf_putc(&t->t3, s[i++]);
        }
        if (t->t3.oom) { t->oom = 1; return; }

        Buf tmp = t->work; t->work = t->t3; t->t3 = tmp;
    }
}

static int is_filter(const Tok *t, uint32_t cp) {
    if (cp < 128) return t->ascii_filter[cp];
    for (int k = 0; k < t->n_wide; k++) if (t->wide_filters[k] == cp) return 1;
    return 0;
}

typedef void (*TokenFn)(Tok *t, const char *w, size_t len, void *ctx);

// every token of a text, in order
static void tok_each(Tok *t, const char *text, size_t len, TokenFn fn, void *ctx) {
    tok_prepare(t, text, len);
    if (t->oom) return;

    char *s = t->work.p ? t->work.p : (char *)"";
    size_t n = t->work.n, i = 0;

    if (t->char_level) {
        while (i < n) {
            size_t j = i;
            u8_next(s, n, &j);
            fn(t, s + i, j - i, ctx);
            i = j;
        }
        return;
    }

    // filter characters become spaces (in place: a space is never longer)
    if (t->filters_len) {
        size_t w = 0;
        while (i < n) {
            size_t j = i;
            uint32_t cp = u8_next(s, n, &j);
            if (is_filter(t, cp)) s[w++] = ' ';
            else { memmove(s + w, s + i, j - i); w += j - i; }
            i = j;
        }
        n = w;
        t->work.n = n;
        i = 0;
    }

    if (t->split) {
        while (i <= n) {
            size_t end = n;
            for (size_t k = i; k + t->split_len <= n; k++)
                if (memcmp(s + k, t->split, t->split_len) == 0) { end = k; break; }
            if (end > i) fn(t, s + i, end - i, ctx);
            if (end >= n) break;
            i = end + t->split_len;
        }
        return;
    }

    while (i < n) {
        size_t j = i;
        uint32_t cp = u8_next(s, n, &j);
        if (is_space_cp(cp)) { i = j; continue; }

        size_t start = i;
        i = j;
        while (i < n) {
            j = i;
            cp = u8_next(s, n, &j);
            if (is_space_cp(cp)) break;
            i = j;
        }
        fn(t, s + start, i - start, ctx);
    }
}

static void count_token(Tok *t, const char *w, size_t len, void *ctx) {
    (void)ctx;
    if (t->oom) return;
    int64_t idx = tok_insert(t, w, len);
    if (idx < 0 || !tok_mark_counted(t, (size_t)idx)) { t->oom = 1; return; }
    t->e[idx].count++;
}

static void push_id(Tok *t, int32_t id) {
    if (t->oom) return;
    if (t->n_ids_buf == t->cap_ids_buf) {
        size_t cap = t->cap_ids_buf ? t->cap_ids_buf * 2 : 4096;
        int32_t *p = (int32_t *)realloc(t->ids, cap * sizeof(int32_t));
        if (!p) { t->oom = 1; return; }
        t->ids = p;
        t->cap_ids_buf = cap;
    }
    t->ids[t->n_ids_buf++] = id;
}

static void id_token(Tok *t, const char *w, size_t len, void *ctx) {
    (void)ctx;
    int32_t id = tok_id(t, w, len);
    if (id > 0) push_id(t, id);
}

//---------------------------------------------------------------------------------
// --- Clay helpers ------------------------------------------------------------
//---------------------------------------------------------------------------------

static ClayPluginValue raise_v(const char *msg, const char *type) {
    api->raise(msg, type);
    return CLAY_NIL;
}

static Tok *tok_of(ClayPluginObject self, const char *who) {
    Tok *t = (Tok *)api->get_data(self);
    if (!t) {
        char msg[128];
        snprintf(msg, sizeof(msg), "%s: not a tokenizer (make one with Ann tokenizer)", who);
        api->raise(msg, "RuntimeError");
    }
    return t;
}

static int is_number(ClayPluginValue v) { return api->is_int(v) || api->is_float(v); }

// true itself (booleans are compared with the ones the API makes)
static int is_true(ClayPluginValue v) {
    return !api->is_nil(v) && v == api->make_bool(1);
}

static ClayPluginValue str_val(const char *s, size_t n) {
    return api->make_bytes(s, n);
}

static char *dup_bytes(const char *s, size_t n) {
    char *p = (char *)malloc(n + 1);
    if (p) { memcpy(p, s, n); p[n] = 0; }
    return p;
}

// texts must be an Array of Strings
static size_t check_texts(ClayPluginValue texts, const char *who) {
    char msg[160];

    if (api->is_string(texts)) {
        snprintf(msg, sizeof(msg), "%s takes an Array of texts, not one String", who);
        api->raise(msg, "TypeError");
    }
    if (!api->is_array(texts)) {
        snprintf(msg, sizeof(msg), "%s takes an Array of texts (Strings)", who);
        api->raise(msg, "TypeError");
    }

    size_t n = api->array_size(texts);
    for (size_t i = 0; i < n; i++)
        if (!api->is_string(api->array_get(texts, i))) {
            snprintf(msg, sizeof(msg), "%s: texts must be Strings (item %zu isn't)", who, i);
            api->raise(msg, "TypeError");
        }
    return n;
}

static void raise_oom(Tok *t) {
    if (t) t->oom = 0;
    api->raise("ann: out of memory", "RuntimeError");
}

static ClayPluginValue wrap_tensor(PTensor m) {
    if (!m) return raise_v("ann: out of memory", "RuntimeError");
    ClayPluginObject obj = api->clone(data_proto);
    api->set_data(obj, m);
    return (ClayPluginValue)obj;
}

// "post" -> 0, "pre" -> 1
static int side_arg(int argc, ClayPluginValue *argv, int i, const char *what) {
    if (i >= argc || api->is_nil(argv[i])) return 0;

    const char *s = api->is_string(argv[i]) ? api->to_string(argv[i]) : "";
    if (strcmp(s, "post") == 0) return 0;
    if (strcmp(s, "pre") == 0) return 1;

    char msg[96];
    snprintf(msg, sizeof(msg), "%s must be \"post\" or \"pre\"", what);
    api->raise(msg, "ArgumentError");
    return 0;
}

// maxlen: nil (the longest) or a positive integer
static long maxlen_arg(int argc, ClayPluginValue *argv, int i, size_t longest) {
    if (i >= argc || api->is_nil(argv[i])) {
        if (longest == 0) api->raise("maxlen must be a positive integer (every sequence is empty)", "ArgumentError");
        if (longest > 0x7FFFFFFF) api->raise("maxlen is too large", "ArgumentError");
        return (long)longest;
    }
    if (!api->is_int(argv[i]) || api->to_int(argv[i]) < 1 || api->to_int(argv[i]) > 0x7FFFFFFF)
        api->raise("maxlen must be a positive integer", "ArgumentError");
    return (long)api->to_int(argv[i]);
}

// a preprocess step's name (Clay's or pyann's): STEP_*, or 0
static int step_of(const char *s) {
    if (strcmp(s, "stripHtml") == 0 || strcmp(s, "strip_html") == 0) return STEP_HTML;
    if (strcmp(s, "splitContractions") == 0 || strcmp(s, "split_contractions") == 0) return STEP_CONTRACTIONS;
    return 0;
}

static const char *step_name(int step) {
    return step == STEP_HTML ? "strip_html" : "split_contractions";
}

// set the steps from an Array: Strings name this file's steps, anything else
// (a Block) is the caller's, which Clay applies (then C applies none)
static void check_steps(ClayPluginValue steps, int *out, int *n_out, int *custom) {
    *n_out = 0;
    *custom = 0;
    if (api->is_nil(steps)) return;
    if (!api->is_array(steps)) api->raise("preprocess must be a name, a Block or an Array of them", "ArgumentError");

    size_t n = api->array_size(steps);
    if (n > MAX_STEPS) api->raise("preprocess: too many steps", "ArgumentError");

    for (size_t i = 0; i < n; i++) {
        ClayPluginValue v = api->array_get(steps, i);

        if (api->is_string(v)) {
            int s = step_of(api->to_string(v));
            if (!s) {
                char msg[160];
                snprintf(msg, sizeof(msg), "preprocess: unknown step \"%.60s\" (stripHtml, splitContractions or a Block)",
                         api->to_string(v));
                api->raise(msg, "ArgumentError");
            }
            out[(*n_out)++] = s;
        }
        else if (api->is_nil(v) || is_number(v) || api->is_array(v) || api->is_dict(v))
            api->raise("preprocess: each step must be a name or a Block", "ArgumentError");
        else
            *custom = 1;
    }
    if (*custom) *n_out = 0;
}

// filters: the characters to turn into spaces
static int set_filters(Tok *t, const char *f, size_t n) {
    free(t->filters); free(t->wide_filters);
    t->filters = NULL; t->wide_filters = NULL; t->n_wide = 0;
    memset(t->ascii_filter, 0, sizeof(t->ascii_filter));

    t->filters = dup_bytes(f, n);
    t->filters_len = n;
    t->wide_filters = (uint32_t *)malloc((n ? n : 1) * sizeof(uint32_t));
    if (!t->filters || !t->wide_filters) return 0;

    size_t i = 0;
    while (i < n) {
        uint32_t cp = u8_next(f, n, &i);
        if (cp < 128) t->ascii_filter[cp] = 1;
        else t->wide_filters[t->n_wide++] = cp;
    }
    return 1;
}

#define DEFAULT_FILTERS "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n"

// the settings shared by create and fromJson, checked before anything is allocated
typedef struct {
    long num_words;
    const char *oov; size_t oov_len; int has_oov;
    int lower, char_level;
    const char *filters; size_t filters_len;
    const char *split; size_t split_len; int has_split;
    int steps[MAX_STEPS], n_steps, custom, single;
} TokSettings;

static ClayPluginValue tok_make(const TokSettings *s) {
    Tok *t = (Tok *)calloc(1, sizeof(Tok));
    if (!t) return raise_v("ann: out of memory", "RuntimeError");

    t->num_words = (int)s->num_words;
    t->lower = s->lower;
    t->char_level = s->char_level;
    t->n_steps = s->n_steps;
    memcpy(t->steps, s->steps, sizeof(t->steps));
    t->custom = s->custom;
    t->single = s->single;

    int ok = set_filters(t, s->filters, s->filters_len);
    if (s->has_oov) ok = ok && (t->oov = dup_bytes(s->oov, s->oov_len)) != NULL;
    if (s->has_split) {
        t->split = dup_bytes(s->split, s->split_len);
        t->split_len = s->split_len;
        ok = ok && t->split != NULL;
    }
    if (!ok) { tok_free(t); return raise_v("ann: out of memory", "RuntimeError"); }

    ClayPluginObject obj = api->clone(tok_proto);
    api->set_data(obj, t);
    return (ClayPluginValue)obj;
}

//---------------------------------------------------------------------------------
// AnnTokenizer create(numWords, oovToken, lower, filters, split, charLevel, steps, single)
//---------------------------------------------------------------------------------
static ClayPluginValue tok_create(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    (void)self;
    TokSettings s;
    memset(&s, 0, sizeof(s));
    ClayPluginValue nil = CLAY_NIL;
#define ARG(i) ((i) < argc ? argv[i] : nil)

    if (!api->is_nil(ARG(0))) {
        if (!api->is_int(ARG(0)) || api->to_int(ARG(0)) < 1 || api->to_int(ARG(0)) > 0x7FFFFFFF)
            return raise_v("numWords must be nil or a positive integer", "ArgumentError");
        s.num_words = (long)api->to_int(ARG(0));
    }
    if (!api->is_nil(ARG(1))) {
        if (!api->is_string(ARG(1))) return raise_v("oovToken must be nil or a String", "ArgumentError");
        s.oov = api->to_bytes(ARG(1), &s.oov_len);
        s.has_oov = 1;
    }
    s.lower = api->is_nil(ARG(2)) ? 1 : is_true(ARG(2));
    if (api->is_nil(ARG(3))) { s.filters = DEFAULT_FILTERS; s.filters_len = strlen(DEFAULT_FILTERS); }
    else if (api->is_string(ARG(3))) s.filters = api->to_bytes(ARG(3), &s.filters_len);
    else return raise_v("filters must be a String", "ArgumentError");
    if (!api->is_nil(ARG(4))) {
        if (!api->is_string(ARG(4))) return raise_v("split must be nil (any whitespace) or a String", "ArgumentError");
        s.split = api->to_bytes(ARG(4), &s.split_len);
        if (s.split_len == 0) return raise_v("split must not be empty", "ArgumentError");
        s.has_split = 1;
    }
    s.char_level = is_true(ARG(5));
    check_steps(ARG(6), s.steps, &s.n_steps, &s.custom);
    s.single = is_true(ARG(7));
#undef ARG

    return tok_make(&s);
}

//---------------------------------------------------------------------------------
// tok _fit(texts): count the words of texts and rebuild the vocabulary -> tok
//---------------------------------------------------------------------------------
static ClayPluginValue tok_fit(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    Tok *t = tok_of(self, "fit");
    if (argc < 1) return raise_v("fit needs an Array of texts", "ArgumentError");

    size_t n = check_texts(argv[0], "fit");

    for (size_t i = 0; i < n && !t->oom; i++) {
        size_t len;
        const char *s = api->to_bytes(api->array_get(argv[0], i), &len);
        tok_each(t, s, len, count_token, NULL);
    }

    if (t->oom || !tok_rebuild(t)) raise_oom(t);
    return (ClayPluginValue)self;
}

// encode every text into t->ids, with t->offs[i] where text i starts (n + 1 offsets)
static void encode_all(Tok *t, ClayPluginValue texts, size_t n) {
    if (n + 1 > t->cap_offs) {
        size_t *o = (size_t *)realloc(t->offs, (n + 1) * sizeof(size_t));
        if (!o) { t->oom = 1; return; }
        t->offs = o;
        t->cap_offs = n + 1;
    }

    t->n_ids_buf = 0;
    for (size_t i = 0; i < n && !t->oom; i++) {
        size_t len;
        const char *s = api->to_bytes(api->array_get(texts, i), &len);
        t->offs[i] = t->n_ids_buf;
        tok_each(t, s, len, id_token, NULL);
    }
    t->offs[n] = t->n_ids_buf;
}

static void need_vocab(Tok *t, const char *who) {
    if (t->n_ids == 0) {
        char msg[128];
        snprintf(msg, sizeof(msg), "%s: the tokenizer has no vocabulary yet (call fit first)", who);
        api->raise(msg, "RuntimeError");
    }
}

//---------------------------------------------------------------------------------
// tok _encode(texts) -> an Array of token ids per text
//---------------------------------------------------------------------------------
static ClayPluginValue tok_encode(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    Tok *t = tok_of(self, "encode");
    if (argc < 1) return raise_v("encode needs an Array of texts", "ArgumentError");

    size_t n = check_texts(argv[0], "encode");
    need_vocab(t, "encode");

    encode_all(t, argv[0], n);
    if (t->oom) raise_oom(t);

    ClayPluginValue out = api->array_new();
    for (size_t i = 0; i < n; i++) {
        ClayPluginValue row = api->array_new();
        for (size_t k = t->offs[i]; k < t->offs[i + 1]; k++)
            api->array_push(row, api->make_int(t->ids[k]));
        api->array_push(out, row);
        api->release(row);
    }
    return out;
}

// rows x maxlen ids from rows of lengths lens (ids at src + offs), padded with value
static void fill_padded(PTensor m, size_t rows, long maxlen, const size_t *offs, const int32_t *ids,
                        const float *fids, int pad_pre, int trunc_pre, real value) {
    for (size_t r = 0; r < rows; r++) {
        size_t start = offs[r], len = offs[r + 1] - offs[r];
        real *dst = m->values + r * (size_t)maxlen;

        if (len > (size_t)maxlen) {
            if (trunc_pre) start += len - (size_t)maxlen;
            len = (size_t)maxlen;
        }

        size_t pad = (size_t)maxlen - len;
        real *at = pad_pre ? dst + pad : dst;

        for (size_t k = 0; k < (size_t)maxlen; k++) dst[k] = value;
        for (size_t k = 0; k < len; k++) at[k] = ids ? (real)ids[start + k] : (real)fids[start + k];
    }
}

//---------------------------------------------------------------------------------
// tok _encodePadded(texts, maxlen, padding, truncating) -> an AnnData of ids, a row
// per text (encode, then Ann padSequences, without the Arrays in between)
//---------------------------------------------------------------------------------
static ClayPluginValue tok_encode_padded(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    Tok *t = tok_of(self, "encodePadded");
    if (argc < 1) return raise_v("encodePadded needs an Array of texts", "ArgumentError");

    size_t n = check_texts(argv[0], "encodePadded");
    int pad_pre = side_arg(argc, argv, 2, "padding");
    int trunc_pre = side_arg(argc, argv, 3, "truncating");
    need_vocab(t, "encodePadded");
    if (n == 0) return raise_v("encodePadded: no texts", "ArgumentError");
    if (argc > 1 && !api->is_nil(argv[1])) maxlen_arg(argc, argv, 1, 1);

    encode_all(t, argv[0], n);
    if (t->oom) raise_oom(t);

    size_t longest = 0;
    for (size_t i = 0; i < n; i++)
        if (t->offs[i + 1] - t->offs[i] > longest) longest = t->offs[i + 1] - t->offs[i];

    long maxlen = maxlen_arg(argc, argv, 1, longest);
    if (n > 0x7FFFFFFF) return raise_v("encodePadded: too many texts", "ArgumentError");

    PTensor m = tensor_create((int)n, (int)maxlen);
    if (!m) raise_oom(t);
    fill_padded(m, n, maxlen, t->offs, t->ids, NULL, pad_pre, trunc_pre, 0);
    return wrap_tensor(m);
}

static void token_to_array(Tok *t, const char *w, size_t len, void *ctx) {
    (void)t;
    ClayPluginValue arr = *(ClayPluginValue *)ctx;
    ClayPluginValue s = str_val(w, len);
    api->array_push(arr, s);
    api->release(s);
}

//---------------------------------------------------------------------------------
// tok _tokenize(text) -> the text's tokens (Strings), before they become ids
//---------------------------------------------------------------------------------
static ClayPluginValue tok_tokenize(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    Tok *t = tok_of(self, "tokenize");
    if (argc < 1 || !api->is_string(argv[0])) return raise_v("tokenize needs a String", "TypeError");

    size_t len;
    const char *s = api->to_bytes(argv[0], &len);

    ClayPluginValue out = api->array_new();
    tok_each(t, s, len, token_to_array, &out);
    if (t->oom) { api->release(out); raise_oom(t); }
    return out;
}

// a row of numbers (an Array) into t->floats; its length
static size_t row_floats(Tok *t, ClayPluginValue row, const char *who) {
    if (!api->is_array(row)) {
        char msg[96];
        snprintf(msg, sizeof(msg), "%s takes an Array of sequences (Arrays of ids) or an AnnData", who);
        api->raise(msg, "TypeError");
    }

    size_t n = api->to_floats(row, NULL, 0);
    if (n > t->cap_floats) {
        float *p = (float *)realloc(t->floats, n * sizeof(float));
        if (!p) raise_oom(t);
        t->floats = p;
        t->cap_floats = n;
    }
    api->to_floats(row, t->floats, n);
    return n;
}

static PTensor data_tensor(ClayPluginValue v) {
    if (api->is_int(v) || api->is_float(v) || api->is_nil(v) || api->is_array(v) || api->is_string(v) || api->is_dict(v))
        return NULL;
    if (api->is_nil(api->get_slot((ClayPluginObject)v, "_annData"))) return NULL;
    return (PTensor)api->get_data((ClayPluginObject)v);
}

static void decode_row(Tok *t, const float *ids, size_t n, Buf *out) {
    out->n = 0;
    buf_reserve(out, 0);

    int first = 1;
    for (size_t k = 0; k < n; k++) {
        double v = ids[k];
        if (!(v > -1 && v < (double)t->n_by_id)) continue;

        int64_t id = (int64_t)v;        // truncated, as Python's int()
        if (id < 0 || (size_t)id >= t->n_by_id || t->by_id[id] < 0) continue;
        if (t->num_words && id >= t->num_words) continue;

        const TokEntry *e = &t->e[t->by_id[id]];
        if (!first && !t->char_level) buf_putc(out, ' ');
        buf_put(out, e->word, e->len);
        first = 0;
    }
}

//---------------------------------------------------------------------------------
// tok decode(sequences) -> texts: each sequence's words joined by spaces (padding
// and ids outside the vocabulary left out). sequences: Arrays of ids or an AnnData
//---------------------------------------------------------------------------------
static ClayPluginValue tok_decode(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    Tok *t = tok_of(self, "decode");
    if (argc < 1) return raise_v("decode needs sequences of ids", "ArgumentError");

    PTensor m = data_tensor(argv[0]);
    if (!m && !api->is_array(argv[0]))
        return raise_v("decode takes an Array of sequences (Arrays of ids) or an AnnData", "TypeError");

    ClayPluginValue out = api->array_new();
    size_t rows = m ? (size_t)m->rows : api->array_size(argv[0]);

    for (size_t r = 0; r < rows; r++) {
        const float *ids;
        size_t n;

        if (m) { ids = m->values + r * (size_t)m->cols; n = (size_t)m->cols; }
        else { n = row_floats(t, api->array_get(argv[0], r), "decode"); ids = t->floats; }

        decode_row(t, ids, n, &t->t3);
        if (t->t3.oom) { api->release(out); raise_oom(t); }

        ClayPluginValue s = str_val(t->t3.p ? t->t3.p : "", t->t3.n);
        api->array_push(out, s);
        api->release(s);
    }
    return out;
}

//---------------------------------------------------------------------------------
// tok vocabSize -> the ids the sequences can hold, padding included: the
// embedding's vocabulary
//---------------------------------------------------------------------------------
static ClayPluginValue tok_vocab(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    (void)argc; (void)argv;
    return api->make_int((intptr_t)tok_vocab_size(tok_of(self, "vocabSize")));
}

//---------------------------------------------------------------------------------
// tok wordIndex -> a Dict of word -> id for every word seen (also those past
// numWords), in id order
//---------------------------------------------------------------------------------
static ClayPluginValue tok_word_index(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    (void)argc; (void)argv;
    Tok *t = tok_of(self, "wordIndex");
    ClayPluginValue d = api->dict_new();

    for (size_t id = 1; id < t->n_by_id; id++)
        if (t->by_id[id] >= 0) api->dict_set(d, t->e[t->by_id[id]].word, api->make_int((intptr_t)id));
    return d;
}

//---------------------------------------------------------------------------------
// tok wordCounts -> a Dict of word -> count in the fitted texts, in the order first seen
//---------------------------------------------------------------------------------
static ClayPluginValue tok_word_counts(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    (void)argc; (void)argv;
    Tok *t = tok_of(self, "wordCounts");
    ClayPluginValue d = api->dict_new();

    for (size_t p = 0; p < t->n_order; p++) {
        const TokEntry *e = &t->e[t->order[p]];
        api->dict_set(d, e->word, api->make_int((intptr_t)e->count));
    }
    return d;
}

//---------------------------------------------------------------------------------
// tok word(id) -> the word with that id, or nil
//---------------------------------------------------------------------------------
static ClayPluginValue tok_word(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    Tok *t = tok_of(self, "word");
    if (argc < 1 || !api->is_int(argv[0])) return raise_v("word needs an integer id", "ArgumentError");

    intptr_t id = api->to_int(argv[0]);
    if (id < 1 || (size_t)id >= t->n_by_id || t->by_id[id] < 0) return CLAY_NIL;
    if (t->num_words && id >= t->num_words) return CLAY_NIL;

    const TokEntry *e = &t->e[t->by_id[id]];
    return str_val(e->word, e->len);
}

//---------------------------------------------------------------------------------
// tok id(word) -> the id encode gives that word (as one token): the oov token's for
// a word outside the vocabulary, or nil without one
//---------------------------------------------------------------------------------
static ClayPluginValue tok_id_of(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    Tok *t = tok_of(self, "id");
    if (argc < 1 || !api->is_string(argv[0])) return raise_v("id needs a String", "TypeError");

    size_t len;
    const char *w = api->to_bytes(argv[0], &len);
    int32_t id = tok_id(t, w, len);
    return id > 0 ? api->make_int(id) : CLAY_NIL;
}

//---------------------------------------------------------------------------------
// --- JSON (pyann's Tokenizer.to_json format) ------------------------------------
//---------------------------------------------------------------------------------

// a JSON string, as Python's json.dumps writes it (ensure_ascii)
static void json_str(Buf *b, const char *s, size_t n) {
    char tmp[16];
    size_t i = 0;

    buf_putc(b, '"');
    while (i < n) {
        unsigned char c = (unsigned char)s[i];

        if (c >= 0x20 && c < 0x7F && c != '"' && c != '\\') { buf_putc(b, (char)c); i++; continue; }

        uint32_t cp = u8_next(s, n, &i);
        switch (cp) {
        case '"': buf_puts(b, "\\\""); break;
        case '\\': buf_puts(b, "\\\\"); break;
        case '\n': buf_puts(b, "\\n"); break;
        case '\r': buf_puts(b, "\\r"); break;
        case '\t': buf_puts(b, "\\t"); break;
        case '\b': buf_puts(b, "\\b"); break;
        case '\f': buf_puts(b, "\\f"); break;
        default:
            if (cp > 0xFFFF) {
                uint32_t v = cp - 0x10000;
                snprintf(tmp, sizeof(tmp), "\\u%04x\\u%04x", 0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF));
            }
            else snprintf(tmp, sizeof(tmp), "\\u%04x", cp);
            buf_puts(b, tmp);
        }
    }
    buf_putc(b, '"');
}

static void json_opt_str(Buf *b, const char *s, size_t n) {
    if (s) json_str(b, s, n);
    else buf_puts(b, "null");
}

//---------------------------------------------------------------------------------
// tok toJson -> the settings and vocabulary as JSON (pyann's format): keep it with a
// trained network, so new texts get the same ids
//---------------------------------------------------------------------------------
static ClayPluginValue tok_to_json(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    (void)argc; (void)argv;
    Tok *t = tok_of(self, "toJson");
    Buf *b = &t->t3;
    char num[32];

    b->n = 0;
    buf_puts(b, "{\"num_words\": ");
    if (t->num_words) { snprintf(num, sizeof(num), "%d", t->num_words); buf_puts(b, num); }
    else buf_puts(b, "null");
    buf_puts(b, ", \"oov_token\": ");
    json_opt_str(b, t->oov, t->oov ? strlen(t->oov) : 0);
    buf_puts(b, t->lower ? ", \"lower\": true" : ", \"lower\": false");
    buf_puts(b, ", \"filters\": ");
    json_str(b, t->filters, t->filters_len);
    buf_puts(b, ", \"split\": ");
    json_opt_str(b, t->split, t->split_len);
    buf_puts(b, t->char_level ? ", \"char_level\": true" : ", \"char_level\": false");
    buf_puts(b, ", \"preprocess\": ");
    if (t->custom) buf_puts(b, "\"custom\"");
    else if (t->n_steps == 0) buf_puts(b, "null");
    else if (t->single && t->n_steps == 1) { buf_putc(b, '"'); buf_puts(b, step_name(t->steps[0])); buf_putc(b, '"'); }
    else {
        buf_putc(b, '[');
        for (int k = 0; k < t->n_steps; k++) {
            if (k) buf_puts(b, ", ");
            buf_putc(b, '"'); buf_puts(b, step_name(t->steps[k])); buf_putc(b, '"');
        }
        buf_putc(b, ']');
    }

    buf_puts(b, ", \"word_counts\": {");
    for (size_t p = 0; p < t->n_order; p++) {
        const TokEntry *e = &t->e[t->order[p]];
        if (p) buf_puts(b, ", ");
        json_str(b, e->word, e->len);
        snprintf(num, sizeof(num), ": %lld", (long long)e->count);
        buf_puts(b, num);
    }
    buf_puts(b, "}, \"word_index\": {");
    int first = 1;
    for (size_t id = 1; id < t->n_by_id; id++) {
        if (t->by_id[id] < 0) continue;
        const TokEntry *e = &t->e[t->by_id[id]];
        if (!first) buf_puts(b, ", ");
        json_str(b, e->word, e->len);
        snprintf(num, sizeof(num), ": %zu", id);
        buf_puts(b, num);
        first = 0;
    }
    buf_puts(b, "}}");

    if (b->oom) raise_oom(t);
    return str_val(b->p, b->n);
}

// A small JSON reader for fromJson: libann's json.c doesn't decode \u escapes
typedef struct JNode JNode;
enum { JN_NULL, JN_BOOL, JN_NUM, JN_STR, JN_ARR, JN_OBJ };
struct JNode {
    int type;
    double num;
    char *str;              // JN_STR, and each member's key for JN_OBJ (in keys)
    size_t len;
    JNode *kids;
    char **keys;
    size_t *klens;
    size_t n, cap;
};

typedef struct { const char *s; size_t n, i; int depth, err; } JParser;

static void jn_free(JNode *j) {
    if (j->type == JN_STR) free(j->str);
    for (size_t k = 0; k < j->n; k++) {
        jn_free(&j->kids[k]);
        if (j->keys) free(j->keys[k]);
    }
    free(j->kids); free(j->keys); free(j->klens);
    memset(j, 0, sizeof(*j));
}

static void jp_ws(JParser *p) {
    while (p->i < p->n && (p->s[p->i] == ' ' || p->s[p->i] == '\t' || p->s[p->i] == '\n' || p->s[p->i] == '\r'))
        p->i++;
}

static int jp_hex4(JParser *p, uint32_t *out) {
    if (p->i + 4 > p->n) return 0;
    uint32_t v = 0;
    for (int k = 0; k < 4; k++) {
        char c = p->s[p->i + k];
        if (!is_hex(c)) return 0;
        v = v * 16 + (uint32_t)hex_val(c);
    }
    p->i += 4;
    *out = v;
    return 1;
}

static int jp_string(JParser *p, char **out, size_t *len) {
    Buf b = { 0 };

    p->i++;     // the opening quote
    buf_reserve(&b, 16);
    while (p->i < p->n && p->s[p->i] != '"') {
        char c = p->s[p->i++];
        if ((unsigned char)c < 0x20) { buf_free(&b); return 0; }
        if (c != '\\') { buf_putc(&b, c); continue; }
        if (p->i >= p->n) { buf_free(&b); return 0; }

        c = p->s[p->i++];
        switch (c) {
        case '"': buf_putc(&b, '"'); break;
        case '\\': buf_putc(&b, '\\'); break;
        case '/': buf_putc(&b, '/'); break;
        case 'b': buf_putc(&b, '\b'); break;
        case 'f': buf_putc(&b, '\f'); break;
        case 'n': buf_putc(&b, '\n'); break;
        case 'r': buf_putc(&b, '\r'); break;
        case 't': buf_putc(&b, '\t'); break;
        case 'u': {
            uint32_t cp, lo;
            if (!jp_hex4(p, &cp)) { buf_free(&b); return 0; }
            if (cp >= 0xD800 && cp <= 0xDBFF && p->i + 6 <= p->n && p->s[p->i] == '\\' && p->s[p->i + 1] == 'u') {
                size_t save = p->i;
                p->i += 2;
                if (jp_hex4(p, &lo) && lo >= 0xDC00 && lo <= 0xDFFF) cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                else p->i = save;
            }
            buf_put_cp(&b, cp);
            break;
        }
        default: buf_free(&b); return 0;
        }
    }
    if (p->i >= p->n || b.oom) { buf_free(&b); return 0; }
    p->i++;

    *out = b.p;
    *len = b.n;
    return 1;
}

static int jn_push(JNode *j, char *key, size_t klen, JNode *kid) {
    if (j->n == j->cap) {
        size_t cap = j->cap ? j->cap * 2 : 8;
        JNode *k = (JNode *)realloc(j->kids, cap * sizeof(JNode));
        if (!k) return 0;
        j->kids = k;
        if (j->type == JN_OBJ) {
            char **ks = (char **)realloc(j->keys, cap * sizeof(char *));
            if (!ks) return 0;
            j->keys = ks;
            size_t *kl = (size_t *)realloc(j->klens, cap * sizeof(size_t));
            if (!kl) return 0;
            j->klens = kl;
        }
        j->cap = cap;
    }
    j->kids[j->n] = *kid;
    if (j->type == JN_OBJ) { j->keys[j->n] = key; j->klens[j->n] = klen; }
    j->n++;
    return 1;
}

static int jp_value(JParser *p, JNode *out) {
    memset(out, 0, sizeof(*out));
    jp_ws(p);
    if (p->i >= p->n || ++p->depth > 64) return 0;

    char c = p->s[p->i];
    int ok = 1;

    if (c == '"') {
        out->type = JN_STR;
        ok = jp_string(p, &out->str, &out->len);
    }
    else if (c == '{' || c == '[') {
        int obj = c == '{';
        out->type = obj ? JN_OBJ : JN_ARR;
        p->i++;
        jp_ws(p);
        if (p->i < p->n && p->s[p->i] == (obj ? '}' : ']')) { p->i++; p->depth--; return 1; }

        for (;;) {
            char *key = NULL;
            size_t klen = 0;
            JNode kid;

            if (obj) {
                jp_ws(p);
                if (p->i >= p->n || p->s[p->i] != '"' || !jp_string(p, &key, &klen)) { ok = 0; break; }
                jp_ws(p);
                if (p->i >= p->n || p->s[p->i] != ':') { free(key); ok = 0; break; }
                p->i++;
            }
            if (!jp_value(p, &kid)) { free(key); jn_free(&kid); ok = 0; break; }
            if (!jn_push(out, key, klen, &kid)) { free(key); jn_free(&kid); ok = 0; break; }

            jp_ws(p);
            if (p->i < p->n && p->s[p->i] == ',') { p->i++; continue; }
            if (p->i < p->n && p->s[p->i] == (obj ? '}' : ']')) { p->i++; break; }
            ok = 0;
            break;
        }
    }
    else if (p->i + 4 <= p->n && memcmp(p->s + p->i, "true", 4) == 0) { out->type = JN_BOOL; out->num = 1; p->i += 4; }
    else if (p->i + 5 <= p->n && memcmp(p->s + p->i, "false", 5) == 0) { out->type = JN_BOOL; p->i += 5; }
    else if (p->i + 4 <= p->n && memcmp(p->s + p->i, "null", 4) == 0) { out->type = JN_NULL; p->i += 4; }
    else if (c == '-' || (c >= '0' && c <= '9')) {
        char tmp[64];
        size_t k = 0;
        while (p->i < p->n && k < sizeof(tmp) - 1 && strchr("+-.eE0123456789", p->s[p->i])) tmp[k++] = p->s[p->i++];
        tmp[k] = 0;
        char *end;
        out->type = JN_NUM;
        out->num = strtod(tmp, &end);
        ok = *end == 0;
    }
    else ok = 0;

    p->depth--;
    return ok;
}

static const JNode *jn_get(const JNode *obj, const char *key) {
    if (obj->type != JN_OBJ) return NULL;
    for (size_t k = 0; k < obj->n; k++)
        if (strlen(key) == obj->klens[k] && memcmp(obj->keys[k], key, obj->klens[k]) == 0) return &obj->kids[k];
    return NULL;
}

static int jn_is_int(const JNode *j) {
    return j && j->type == JN_NUM && j->num == floor(j->num) && fabs(j->num) < 9e15;
}

// the tokenizer a parsed JSON describes; msg says why not (NULL result)
static Tok *tok_from_tree(const JNode *root, const TokSettings *override, int *warn, const char **msg) {
    const JNode *nw = jn_get(root, "num_words"), *oov = jn_get(root, "oov_token"),
                *lower = jn_get(root, "lower"), *filters = jn_get(root, "filters"),
                *split = jn_get(root, "split"), *cl = jn_get(root, "char_level"),
                *pre = jn_get(root, "preprocess"), *wc = jn_get(root, "word_counts"),
                *wi = jn_get(root, "word_index");

    *msg = "not a saved tokenizer (Ann tokenizer's toJson, or pyann's Tokenizer.to_json)";
    if (!nw || !(nw->type == JN_NULL || (jn_is_int(nw) && nw->num >= 1 && nw->num <= 0x7FFFFFFF))) return NULL;
    if (!oov || !(oov->type == JN_NULL || oov->type == JN_STR)) return NULL;
    if (!lower || lower->type != JN_BOOL || !cl || cl->type != JN_BOOL) return NULL;
    if (!filters || filters->type != JN_STR) return NULL;
    if (!split || !(split->type == JN_NULL || (split->type == JN_STR && split->len > 0))) return NULL;
    if (!wc || wc->type != JN_OBJ || !wi || wi->type != JN_OBJ) return NULL;
    for (size_t k = 0; k < wc->n; k++) if (!jn_is_int(&wc->kids[k]) || wc->kids[k].num < 0) return NULL;
    for (size_t k = 0; k < wi->n; k++)
        if (!jn_is_int(&wi->kids[k]) || wi->kids[k].num < 1 || wi->kids[k].num > 0x7FFFFFFF) return NULL;

    *msg = "ann: out of memory";
    Tok *t = (Tok *)calloc(1, sizeof(Tok));
    if (!t) return NULL;

    t->num_words = nw->type == JN_NULL ? 0 : (int)nw->num;
    t->lower = lower->num != 0;
    t->char_level = cl->num != 0;

    int ok = set_filters(t, filters->str, filters->len);
    if (oov->type == JN_STR) ok = ok && (t->oov = dup_bytes(oov->str, oov->len)) != NULL;
    if (split->type == JN_STR) {
        t->split = dup_bytes(split->str, split->len);
        t->split_len = split->len;
        ok = ok && t->split != NULL;
    }

    // the preprocess: the caller's, else the saved names
    *warn = 0;
    if (override) {
        t->n_steps = override->n_steps;
        memcpy(t->steps, override->steps, sizeof(t->steps));
        t->custom = override->custom;
        t->single = override->single;
    }
    else if (pre && pre->type == JN_STR && step_of(pre->str)) {
        t->steps[0] = step_of(pre->str);
        t->n_steps = 1;
        t->single = 1;
    }
    else if (pre && pre->type == JN_ARR && pre->n <= MAX_STEPS) {
        for (size_t k = 0; k < pre->n; k++) {
            int s = pre->kids[k].type == JN_STR ? step_of(pre->kids[k].str) : 0;
            if (!s) { t->n_steps = 0; *warn = 1; break; }
            t->steps[t->n_steps++] = s;
        }
    }
    else if (pre && pre->type != JN_NULL) *warn = 1;

    // word_counts in their order, then word_index's ids
    for (size_t k = 0; ok && k < wc->n; k++) {
        int64_t idx = tok_insert(t, wc->keys[k], wc->klens[k]);
        if (idx < 0 || !tok_mark_counted(t, (size_t)idx)) { ok = 0; break; }
        t->e[idx].count = (int64_t)wc->kids[k].num;
    }
    for (size_t k = 0; ok && k < wi->n; k++) {
        int64_t idx = tok_insert(t, wi->keys[k], wi->klens[k]);
        if (idx < 0) { ok = 0; break; }
        t->e[idx].id = (int32_t)wi->kids[k].num;
    }
    ok = ok && tok_index_ids(t);

    if (ok && t->oov) {
        int64_t idx = tok_find(t, t->oov, strlen(t->oov));
        t->oov_id = idx >= 0 ? t->e[idx].id : 0;
    }
    if (!ok) { tok_free(t); return NULL; }

    *msg = NULL;
    return t;
}

//---------------------------------------------------------------------------------
// AnnTokenizer fromJson(text, steps, single) -> the tokenizer toJson saved. With
// steps (not nil) its preprocess is those, else the saved names; a tokenizer saved
// with steps of the caller's gets the slot _lostPreprocess (Clay warns)
//---------------------------------------------------------------------------------
static ClayPluginValue tok_from_json(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    (void)self;
    if (argc < 1 || !api->is_string(argv[0])) return raise_v("fromJson needs a String of JSON", "TypeError");

    TokSettings over;
    memset(&over, 0, sizeof(over));
    int has_over = argc > 1 && !api->is_nil(argv[1]);
    if (has_over) {
        check_steps(argv[1], over.steps, &over.n_steps, &over.custom);
        over.single = argc > 2 && is_true(argv[2]);
    }

    size_t len;
    const char *s = api->to_bytes(argv[0], &len);
    JParser p = { s, len, 0, 0, 0 };
    JNode root;
    int parsed = jp_value(&p, &root);
    jp_ws(&p);
    if (!parsed || p.i != p.n) {
        jn_free(&root);
        return raise_v("fromJson: not valid JSON", "ArgumentError");
    }

    int warn = 0;
    const char *msg = NULL;
    Tok *t = tok_from_tree(&root, has_over ? &over : NULL, &warn, &msg);
    jn_free(&root);

    if (!t) {
        char m[200];
        snprintf(m, sizeof(m), "fromJson: %s", msg);
        return raise_v(m, strstr(msg, "memory") ? "RuntimeError" : "ArgumentError");
    }

    ClayPluginObject obj = api->clone(tok_proto);
    api->set_data(obj, t);
    if (warn) api->set_slot(obj, "_lostPreprocess", api->make_bool(1));
    return (ClayPluginValue)obj;
}

//---------------------------------------------------------------------------------
// tok settings -> a Dict: numWords, oovToken, lower, filters, split, charLevel, preprocess
//---------------------------------------------------------------------------------
static ClayPluginValue tok_settings(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    (void)argc; (void)argv;
    Tok *t = tok_of(self, "settings");
    ClayPluginValue d = api->dict_new(), v;

    api->dict_set(d, "numWords", t->num_words ? api->make_int(t->num_words) : CLAY_NIL);
    v = t->oov ? str_val(t->oov, strlen(t->oov)) : CLAY_NIL;
    api->dict_set(d, "oovToken", v); api->release(v);
    api->dict_set(d, "lower", api->make_bool(t->lower));
    v = str_val(t->filters, t->filters_len);
    api->dict_set(d, "filters", v); api->release(v);
    v = t->split ? str_val(t->split, t->split_len) : CLAY_NIL;
    api->dict_set(d, "split", v); api->release(v);
    api->dict_set(d, "charLevel", api->make_bool(t->char_level));

    if (t->custom) v = api->make_string("custom");
    else if (t->n_steps == 0) v = CLAY_NIL;
    else {
        v = api->array_new();
        for (int k = 0; k < t->n_steps; k++) {
            ClayPluginValue s = api->make_string(t->steps[k] == STEP_HTML ? "stripHtml" : "splitContractions");
            api->array_push(v, s);
            api->release(s);
        }
    }
    api->dict_set(d, "preprocess", v);
    api->release(v);
    return d;
}

//---------------------------------------------------------------------------------
// --- functions that need no tokenizer ------------------------------------------
//---------------------------------------------------------------------------------

// the scratch buffers for stripHtml and splitContractions: kept between calls,
// freed at exit by nobody (a few buffers the size of the longest text)
static Buf g_out, g_t1, g_t2;

//---------------------------------------------------------------------------------
// AnnTokenizer stripHtml(text) -> the text without its HTML: tags become spaces,
// comments, scripts and styles go, and entities are decoded (&amp; to &)
//---------------------------------------------------------------------------------
static ClayPluginValue tok_strip_html(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    (void)self;
    if (argc < 1 || !api->is_string(argv[0])) return raise_v("stripHtml needs a String", "TypeError");

    size_t len;
    const char *s = api->to_bytes(argv[0], &len);
    strip_html(s, len, &g_out, &g_t1, &g_t2);
    if (g_out.oom) { buf_free(&g_out); buf_free(&g_t1); buf_free(&g_t2); return raise_v("ann: out of memory", "RuntimeError"); }
    return str_val(g_out.p ? g_out.p : "", g_out.n);
}

//---------------------------------------------------------------------------------
// AnnTokenizer splitContractions(text) -> contractions split as GloVe's training
// text was: "don't" -> "do n't", "it's" -> "it 's"; quotes at a word's ends go
//---------------------------------------------------------------------------------
static ClayPluginValue tok_split_contractions(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    (void)self;
    if (argc < 1 || !api->is_string(argv[0])) return raise_v("splitContractions needs a String", "TypeError");

    size_t len;
    const char *s = api->to_bytes(argv[0], &len);
    split_contractions(s, len, &g_out, &g_t1, &g_t2);
    if (g_out.oom) { buf_free(&g_out); buf_free(&g_t1); buf_free(&g_t2); return raise_v("ann: out of memory", "RuntimeError"); }
    return str_val(g_out.p ? g_out.p : "", g_out.n);
}

// the scratch for padSequences
static size_t *g_offs;
static float *g_vals;

//---------------------------------------------------------------------------------
// AnnTokenizer padSequences(sequences, maxlen, padding, truncating, value) -> an AnnData,
// a row of maxlen ids per sequence (an Array of ids): shorter ones padded with value
// (default 0), longer ones cut. padding and truncating are "post" (the end, default)
// or "pre"; maxlen nil is the longest sequence's length
//---------------------------------------------------------------------------------
static ClayPluginValue tok_pad(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    (void)self;
    if (argc < 1 || !api->is_array(argv[0]))
        return raise_v("padSequences takes an Array of sequences (Arrays of ids)", "TypeError");

    int pad_pre = side_arg(argc, argv, 2, "padding");
    int trunc_pre = side_arg(argc, argv, 3, "truncating");
    real value = 0;
    if (argc > 4 && !api->is_nil(argv[4])) {
        if (!is_number(argv[4])) return raise_v("value must be a number", "ArgumentError");
        value = api->to_float(argv[4]);
    }

    size_t rows = api->array_size(argv[0]), total = 0, longest = 0;
    if (rows == 0) return raise_v("padSequences: no sequences", "ArgumentError");
    if (rows > 0x7FFFFFFF) return raise_v("padSequences: too many sequences", "ArgumentError");

    // sizes first (to_floats raises on anything but numbers, before we allocate)
    for (size_t r = 0; r < rows; r++) {
        ClayPluginValue row = api->array_get(argv[0], r);
        if (!api->is_array(row)) return raise_v("padSequences: each sequence must be an Array of ids", "TypeError");
        size_t n = api->to_floats(row, NULL, 0);
        total += n;
        if (n > longest) longest = n;
    }
    long maxlen = maxlen_arg(argc, argv, 1, longest);

    free(g_offs); free(g_vals);
    g_offs = (size_t *)malloc((rows + 1) * sizeof(size_t));
    g_vals = (float *)malloc((total ? total : 1) * sizeof(float));
    if (!g_offs || !g_vals) return raise_v("ann: out of memory", "RuntimeError");

    size_t at = 0;
    for (size_t r = 0; r < rows; r++) {
        g_offs[r] = at;
        at += api->to_floats(api->array_get(argv[0], r), g_vals + at, total - at);
    }
    g_offs[rows] = at;

    PTensor m = tensor_create((int)rows, (int)maxlen);
    if (!m) return raise_v("ann: out of memory", "RuntimeError");
    fill_padded(m, rows, maxlen, g_offs, NULL, g_vals, pad_pre, trunc_pre, value);
    return wrap_tensor(m);
}

//---------------------------------------------------------------------------------
// --- word vectors ------------------------------------------------------------------
//---------------------------------------------------------------------------------

// splitmix64 + Box-Muller, for "random" vectors
static uint64_t rng_next(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static double rng_normal(uint64_t *s) {
    double u1 = ((double)(rng_next(s) >> 11) + 1.0) / 9007199254740993.0;
    double u2 = (double)(rng_next(s) >> 11) / 9007199254740992.0;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

// one line of the file into b (without its \n or \r); 0 at the end
static int read_line(FILE *f, Buf *b) {
    b->n = 0;
    int c, any = 0;

    while ((c = fgetc(f)) != EOF) {
        any = 1;
        if (c == '\n') break;
        buf_putc(b, (char)c);
    }
    if (b->n && b->p[b->n - 1] == '\r') b->p[--b->n] = 0;
    if (!b->p) buf_reserve(b, 0);
    if (b->p) b->p[b->n] = 0;
    return any;
}

typedef struct {
    float *m;
    unsigned char *found;
    size_t size, dim, n_found;
} Vectors;

// the vectors of the words of t (ids below size, not pad_id or the oov token) from
// path; NULL on success, else why not (written to err)
static const char *read_vectors(Tok *t, const char *path, size_t size, size_t want_dim, int random,
                                int has_seed, uint64_t seed, long pad_id, Vectors *v, char *err, size_t errlen) {
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, errlen, "can't open %s", path); return err; }

    Buf line = { 0 };
    size_t dim = want_dim, line_no = 0;
    int first = 1;
    const char *res = NULL;
    int64_t oov_idx = t->oov ? tok_find(t, t->oov, strlen(t->oov)) : -1;

    memset(v, 0, sizeof(*v));
    v->size = size;
    v->found = (unsigned char *)calloc(size, 1);
    if (!v->found) { fclose(f); return "ann: out of memory"; }

    while (read_line(f, &line)) {
        line_no++;
        if (line.oom) { res = "ann: out of memory"; break; }

        char *s = line.p;
        size_t n = line.n;

        if (first) {
            first = 0;
            // word2vec / fastText: a "count dim" header line
            char *a = s, *end;
            while (*a == ' ' || *a == '\t') a++;
            if (*a >= '0' && *a <= '9') {
                unsigned long long c1 = strtoull(a, &end, 10);
                (void)c1;
                if (*end == ' ' || *end == '\t') {
                    char *b2 = end;
                    while (*b2 == ' ' || *b2 == '\t') b2++;
                    if (*b2 >= '0' && *b2 <= '9') {
                        unsigned long long d = strtoull(b2, &end, 10);
                        while (*end == ' ' || *end == '\t') end++;
                        if (*end == 0) {
                            if (want_dim && d != want_dim) {
                                snprintf(err, errlen, "%s has %llu-dimensional vectors, not %zu", path, d, want_dim);
                                res = err;
                                break;
                            }
                            dim = (size_t)d;
                            continue;
                        }
                    }
                }
            }
        }

        char *sp = (char *)memchr(s, ' ', n);
        size_t wlen = sp ? (size_t)(sp - s) : n;
        int64_t idx = tok_find(t, s, wlen);
        if (idx < 0 || idx == oov_idx) continue;

        int32_t id = t->e[idx].id;
        if (id <= 0 || (size_t)id >= size || id == pad_id || v->found[id]) continue;

        // count the numbers first
        size_t count = 0;
        char *p = sp ? sp + 1 : s + n;
        while (*p) {
            while (*p && is_space_cp((unsigned char)*p)) p++;
            if (!*p) break;
            count++;
            while (*p && !is_space_cp((unsigned char)*p)) p++;
        }

        if (count == 0) continue;       // a word alone on its line
        if (!dim) dim = count;
        if (count > dim) continue;      // a "word" with spaces in it that starts like ours
        if (count < dim) {
            snprintf(err, errlen, "%s line %zu: %zu numbers for \"%.40s\", expected %zu", path, line_no, count, t->e[idx].word, dim);
            res = err;
            break;
        }

        if (!v->m) {
            if (dim > 0x7FFFFFFF / (size ? size : 1)) { res = "the vectors are too large"; break; }
            v->m = (float *)calloc(size * dim, sizeof(float));
            v->dim = dim;
            if (!v->m) { res = "ann: out of memory"; break; }
        }

        float *row = v->m + (size_t)id * dim;
        p = sp + 1;
        for (size_t k = 0; k < dim; k++) {
            while (*p && is_space_cp((unsigned char)*p)) p++;
            char *end;
            double x = strtod(p, &end);
            if (end == p || (*end && !is_space_cp((unsigned char)*end))) {
                snprintf(err, errlen, "%s line %zu: not a number in the vector for \"%.40s\"", path, line_no, t->e[idx].word);
                res = err;
                break;
            }
            if (!isfinite(x) || !isfinite((float)x)) {
                snprintf(err, errlen, "%s has vectors with NaN or infinite values", path);
                res = err;
                break;
            }
            row[k] = (float)x;
            p = end;
        }
        if (res) break;

        v->found[id] = 1;
        v->n_found++;
    }
    fclose(f);
    buf_free(&line);

    if (!res && !dim) { snprintf(err, errlen, "%s has no vectors", path); res = err; }
    if (!res && !v->m) {
        v->m = (float *)calloc(size * dim, sizeof(float));
        v->dim = dim;
        if (!v->m) res = "ann: out of memory";
    }

    if (!res && random && v->n_found) {
        double sum = 0, sq = 0, cnt = (double)v->n_found * (double)dim;
        for (size_t i = 0; i < size; i++)
            if (v->found[i]) for (size_t k = 0; k < dim; k++) sum += v->m[i * dim + k];
        double mean = sum / cnt;
        for (size_t i = 0; i < size; i++)
            if (v->found[i])
                for (size_t k = 0; k < dim; k++) { double d = v->m[i * dim + k] - mean; sq += d * d; }
        double sd = sqrt(sq / cnt);

        uint64_t st = has_seed ? seed : ((uint64_t)rand() << 32) ^ (uint64_t)rand();
        for (size_t i = 0; i < size; i++)
            if (!v->found[i] && (long)i != pad_id)
                for (size_t k = 0; k < dim; k++) v->m[i * dim + k] = (float)(mean + sd * rng_normal(&st));
    }

    if (res) { free(v->m); v->m = NULL; }
    free(v->found);
    v->found = NULL;
    return res;
}

//---------------------------------------------------------------------------------
// AnnTokenizer _wordVectors(path, source, vocabSize, dim, missing, seed, paddingId)
//   -> [vectors (an AnnData, a row per id), found]. source: a tokenizer or a Dict of
//   word -> id
//---------------------------------------------------------------------------------
static ClayPluginValue tok_word_vectors(ClayPluginObject self, int argc, ClayPluginValue *argv) {
    (void)self;
    ClayPluginValue nil = CLAY_NIL;
#define ARG(i) ((i) < argc ? argv[i] : nil)

    if (!api->is_string(ARG(0))) return raise_v("loadWordVectors needs a file path", "ArgumentError");
    size_t plen;
    const char *path = api->to_bytes(ARG(0), &plen);
    if (plen >= 3 && strcmp(path + plen - 3, ".gz") == 0)
        return raise_v("loadWordVectors: gzip files aren't read; gunzip it first", "ArgumentError");

    ClayPluginValue src = ARG(1);
    Tok *t = NULL;
    int is_tok = !api->is_dict(src) && !api->is_nil(src) && !is_number(src) && !api->is_string(src) &&
                 !api->is_array(src) && !api->is_nil(api->get_slot((ClayPluginObject)src, "_annTokenizer"));
    if (is_tok) {
        t = (Tok *)api->get_data((ClayPluginObject)src);
        if (!t || t->n_ids == 0) return raise_v("loadWordVectors: the tokenizer has no vocabulary yet (call fit first)", "RuntimeError");
    }
    else if (api->is_dict(src)) {
        size_t pos = 0;
        const char *k;
        ClayPluginValue v;
        while (api->dict_next(src, &pos, &k, &v))
            if (!api->is_int(v) || api->to_int(v) < 0 || api->to_int(v) > 0x7FFFFFFF)
                return raise_v("loadWordVectors: the Dict's ids must be integers >= 0", "ArgumentError");
    }
    else return raise_v("loadWordVectors needs a tokenizer or a Dict of word -> id", "ArgumentError");

    long size = -1, dim = 0, pad_id = 0;
    if (!api->is_nil(ARG(2))) {
        if (!api->is_int(ARG(2)) || api->to_int(ARG(2)) < 1 || api->to_int(ARG(2)) > 0x7FFFFFFF)
            return raise_v("vocabSize must be a positive integer", "ArgumentError");
        size = (long)api->to_int(ARG(2));
    }
    if (!api->is_nil(ARG(3))) {
        if (!api->is_int(ARG(3)) || api->to_int(ARG(3)) < 1 || api->to_int(ARG(3)) > 100000)
            return raise_v("dim must be nil or a positive integer", "ArgumentError");
        dim = (long)api->to_int(ARG(3));
    }
    int random = 0;
    if (!api->is_nil(ARG(4))) {
        const char *m = api->is_string(ARG(4)) ? api->to_string(ARG(4)) : "";
        if (strcmp(m, "random") == 0) random = 1;
        else if (strcmp(m, "zeros") != 0) return raise_v("missing must be \"zeros\" or \"random\"", "ArgumentError");
    }
    int has_seed = !api->is_nil(ARG(5));
    if (has_seed && !api->is_int(ARG(5))) return raise_v("seed must be an integer", "ArgumentError");
    uint64_t seed = has_seed ? (uint64_t)api->to_int(ARG(5)) : 0;
    if (argc > 6) {
        if (api->is_nil(ARG(6))) pad_id = -1;
        else if (api->is_int(ARG(6))) pad_id = (long)api->to_int(ARG(6));
        else return raise_v("paddingId must be an integer or nil", "ArgumentError");
    }
#undef ARG

    // a Dict becomes a temporary tokenizer (no oov token), owned by a Clay object so
    // that an error can't leak it
    ClayPluginObject holder = NULL;
    if (!is_tok) {
        t = (Tok *)calloc(1, sizeof(Tok));
        if (!t) return raise_v("ann: out of memory", "RuntimeError");
        holder = api->clone(tok_proto);
        api->set_data(holder, t);

        size_t pos = 0;
        const char *k;
        ClayPluginValue v;
        int32_t max = 0;
        while (api->dict_next(src, &pos, &k, &v)) {
            int64_t idx = tok_insert(t, k, strlen(k));
            if (idx < 0) { api->release((ClayPluginValue)holder); return raise_v("ann: out of memory", "RuntimeError"); }
            t->e[idx].id = (int32_t)api->to_int(v);
            if (t->e[idx].id > max) max = t->e[idx].id;
        }
        if (size < 0) size = (long)max + 1;
    }
    if (size < 0) size = (long)tok_vocab_size(t);

    char err[300];
    Vectors v;
    const char *res = read_vectors(t, path, (size_t)size, (size_t)dim, random, has_seed, seed, pad_id, &v, err, sizeof(err));
    if (holder) api->release((ClayPluginValue)holder);

    if (res) {
        char msg[340];
        snprintf(msg, sizeof(msg), "loadWordVectors: %s", res);
        return raise_v(msg, strstr(res, "memory") ? "RuntimeError" : strstr(res, "can't open") ? "RuntimeError" : "ArgumentError");
    }

    PTensor m = tensor_create((int)size, (int)v.dim);
    if (!m) { free(v.m); return raise_v("ann: out of memory", "RuntimeError"); }
    for (size_t i = 0; i < (size_t)size * v.dim; i++) m->values[i] = (real)v.m[i];
    free(v.m);

    ClayPluginValue out = api->array_new();
    ClayPluginValue data = wrap_tensor(m);
    api->array_push(out, data);
    api->release(data);
    api->array_push(out, api->make_int((intptr_t)v.n_found));
    return out;
}

//---------------------------------------------------------------------------------
// registration
//---------------------------------------------------------------------------------
static ClayPluginMethod tok_methods[] = {
    { "create", tok_create },             { "fromJson", tok_from_json },
    { "_fit", tok_fit },                  { "_encode", tok_encode },
    { "_encodePadded", tok_encode_padded },{ "_tokenize", tok_tokenize },
    { "decode", tok_decode },             { "vocabSize", tok_vocab },
    { "wordIndex", tok_word_index },      { "wordCounts", tok_word_counts },
    { "word", tok_word },                 { "id", tok_id_of },
    { "toJson", tok_to_json },            { "settings", tok_settings },
    { "stripHtml", tok_strip_html },      { "splitContractions", tok_split_contractions },
    { "padSequences", tok_pad },          { "_wordVectors", tok_word_vectors },
    { NULL, NULL }
};

// called from clay_ann.c's ann_plugin_init with the AnnData prototype
void ann_text_init(const ClayPluginAPI *clay_api, ClayPluginObject annData) {
    api = clay_api;
    data_proto = annData;
    if (api->api_version < 5) return;       // needs make_bytes and Dicts

    ClayPluginType def = { "AnnTokenizer", tok_methods, tok_free, sizeof(void *) };
    tok_proto = api->register_type(&def);
    api->set_slot(tok_proto, "_annTokenizer", api->make_bool(1));
}
