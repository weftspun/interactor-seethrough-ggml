// An interactor, driven with no transport at all.
//
// That is the whole proof. If this needs a socket to run, the contract has failed, so the test
// is as much that this file compiles and links against nothing as that its assertions hold.
//
// SPDX-License-Identifier: Apache-2.0

#include "weft/cbor.h"
#include "weft/interactor.h"

#include <stdio.h>
#include <string.h>

typedef struct {
	int count;
} tally_t;

// The smallest interactor that is not trivial: it holds state, it answers, and one command
// asks the service to stop. It knows nothing about how it was reached.
static size_t tally_ask(void *ctx, const char *command, unsigned char *reply, size_t cap,
                        int *stop) {
	tally_t *t = (tally_t *)ctx;
	weft_cbor_t c = weft_cbor_to(reply, cap);

	if (strcmp(command, "quit") == 0) {
		*stop = 1;
		return 0; // nothing to say, which is an answer
	}

	t->count++;
	weft_cbor_map(&c, 2);
	weft_cbor_kv_text(&c, "say", command);
	weft_cbor_kv_int(&c, "count", t->count);
	return weft_cbor_over(&c) ? 0 : c.n;
}

#define CHECK(cond, what)                                                                      \
	do {                                                                                       \
		if (!(cond)) {                                                                         \
			printf("  FAIL %s\n", what);                                                       \
			failed++;                                                                          \
		} else {                                                                               \
			printf("  ok   %s\n", what);                                                       \
		}                                                                                      \
	} while (0)

int main(void) {
	tally_t tally = {0};
	weft_interactor_t in = {tally_ask, &tally};
	unsigned char reply[256];
	int stop = 0, failed = 0;

	size_t n = weft_ask(&in, "hello", reply, sizeof reply, &stop);

	// a2                    map of 2
	//   63 73 61 79         "say"
	//   65 68 65 6c 6c 6f   "hello"
	//   65 63 6f 75 6e 74   "count"
	//   01                  1
	static const unsigned char want[] = {0xa2, 0x63, 's',  'a',  'y',  0x65, 'h',  'e',
	                                     'l',  'l',  'o',  0x65, 'c',  'o',  'u',  'n',
	                                     't',  0x01};
	CHECK(n == sizeof want, "the reply is the length the bytes say it is");
	CHECK(n == sizeof want && memcmp(reply, want, n) == 0, "and it is those bytes exactly");
	CHECK(stop == 0, "an ordinary command does not stop the service");

	n = weft_ask(&in, "hello", reply, sizeof reply, &stop);
	CHECK(n > 0 && reply[n - 1] == 0x02, "the interactor kept its own state");

	n = weft_ask(&in, "quit", reply, sizeof reply, &stop);
	CHECK(n == 0, "nothing to say is zero bytes and not an error");
	CHECK(stop == 1, "the interactor asked the service to wind down");

	// A buffer too small must be refused rather than truncated: a short batch and a cut one
	// decode the same way, and that is the failure the whole encoding exists to avoid.
	unsigned char tiny[4];
	n = weft_ask(&in, "a much longer command than four bytes", tiny, sizeof tiny, &stop);
	CHECK(n == 0, "a reply that does not fit is refused, not cut");

	printf(failed == 0 ? "\nthe interactor stands on its own\n" : "\n%d failed\n", failed);
	return failed == 0 ? 0 : 1;
}
