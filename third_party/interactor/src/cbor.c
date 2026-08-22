// See include/weft/cbor.h.
//
// SPDX-License-Identifier: Apache-2.0

#include "weft/cbor.h"

#include <string.h>

void weft_cbor_raw(weft_cbor_t *c, unsigned char b) {
	if (c->n >= c->cap) {
		c->over = 1;
		return;
	}
	c->p[c->n++] = b;
}

// A CBOR head is the major type in the top three bits and the argument in the low five, with
// the argument spilling into 1, 2, 4 or 8 following bytes as it grows.
void weft_cbor_head(weft_cbor_t *c, int major, uint64_t v) {
	const unsigned char m = (unsigned char)(major << 5);
	if (v < 24) {
		weft_cbor_raw(c, (unsigned char)(m | v));
	} else if (v <= 0xff) {
		weft_cbor_raw(c, (unsigned char)(m | 24));
		weft_cbor_raw(c, (unsigned char)v);
	} else if (v <= 0xffff) {
		weft_cbor_raw(c, (unsigned char)(m | 25));
		for (int i = 1; i >= 0; i--) weft_cbor_raw(c, (unsigned char)(v >> (i * 8)));
	} else if (v <= 0xffffffffULL) {
		weft_cbor_raw(c, (unsigned char)(m | 26));
		for (int i = 3; i >= 0; i--) weft_cbor_raw(c, (unsigned char)(v >> (i * 8)));
	} else {
		weft_cbor_raw(c, (unsigned char)(m | 27));
		for (int i = 7; i >= 0; i--) weft_cbor_raw(c, (unsigned char)(v >> (i * 8)));
	}
}

// A negative number is major type 1 over its own encoding, `-1 - n`. A place is int64 and
// routinely negative — an under-market is thirty metres down — so this is not a spare case.
void weft_cbor_int(weft_cbor_t *c, int64_t v) {
	if (v < 0)
		weft_cbor_head(c, 1, (uint64_t)(-(v + 1)));
	else
		weft_cbor_head(c, 0, (uint64_t)v);
}

void weft_cbor_text(weft_cbor_t *c, const char *s) {
	const size_t n = s ? strlen(s) : 0;
	weft_cbor_head(c, 3, n);
	for (size_t i = 0; i < n; i++) weft_cbor_raw(c, (unsigned char)s[i]);
}

void weft_cbor_map(weft_cbor_t *c, uint64_t pairs) { weft_cbor_head(c, 5, pairs); }
void weft_cbor_array(weft_cbor_t *c, uint64_t items) { weft_cbor_head(c, 4, items); }
void weft_cbor_bool(weft_cbor_t *c, int b) { weft_cbor_raw(c, (unsigned char)(b ? 0xf5 : 0xf4)); }

// Closes an indefinite-length array, which is what a writer uses when it is streaming rows out
// of a query and does not know the count until the query is done.
void weft_cbor_break(weft_cbor_t *c) { weft_cbor_raw(c, 0xff); }

void weft_cbor_kv_int(weft_cbor_t *c, const char *k, int64_t v) {
	weft_cbor_text(c, k);
	weft_cbor_int(c, v);
}

void weft_cbor_kv_text(weft_cbor_t *c, const char *k, const char *v) {
	weft_cbor_text(c, k);
	weft_cbor_text(c, v);
}

void weft_cbor_kv_bool(weft_cbor_t *c, const char *k, int v) {
	weft_cbor_text(c, k);
	weft_cbor_bool(c, v);
}
