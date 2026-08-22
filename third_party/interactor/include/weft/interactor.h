// The interactor contract.
//
// An interactor is a command in and reply bytes out. It has no socket, no poll loop and no
// idea what carried the command to it. A transport is bytes to and from a wire, and has no
// idea what the command means. A service is the composition of some of each.
//
// The point of writing the contract down once, in a repository neither side owns, is that a
// transport can then be paired with an interactor that did not exist when it was written. A
// contract that lived in either one would make the other its dependent, which is the shape
// this exists to refuse.
//
// SPDX-License-Identifier: Apache-2.0
#ifndef WEFT_INTERACTOR_H
#define WEFT_INTERACTOR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── An interactor ─────────────────────────────────────────────────────────────

typedef struct {
	// One command, answered whole. Writes at most `cap` bytes into `reply` and returns how
	// many; 0 means there is nothing to say, which is an answer and not an error.
	//
	// `command` is borrowed and const: an interactor that needs to tokenise it copies it
	// first, because the transport may still own those bytes.
	//
	// `stop` is the service's own flag, not the transport's. An interactor sets it to ask
	// the service to wind down. A transport must never read it — a client that can close one
	// connection is not a client that can close the process.
	size_t (*ask)(void *ctx, const char *command, unsigned char *reply, size_t cap, int *stop);
	void *ctx;
} weft_interactor_t;

static inline size_t weft_ask(const weft_interactor_t *in, const char *command,
                              unsigned char *reply, size_t cap, int *stop) {
	return (in && in->ask) ? in->ask(in->ctx, command, reply, cap, stop) : 0;
}

// ── A transport ───────────────────────────────────────────────────────────────

// Four is the most any transport here needs: WebTransport takes a UDP socket and a protocol
// timer, and a listener takes one socket. A transport that wants more is a transport that has
// grown a loop of its own, which is the service's job and not its.
#define WEFT_TRANSPORT_MAX_FDS 4

typedef struct {
	// The descriptors the service must poll, written into `out`. Returns how many.
	//
	// It is asked again every time round the loop, because a listener's set changes as
	// clients arrive and a service that cached the answer would poll a closed socket.
	int (*fds)(void *ctx, int *out, int max);

	// That descriptor is readable. Which one it is belongs to the transport: a service that
	// knew a UDP socket from a timer would be a service that knew what QUIC is.
	void (*ready)(void *ctx, int fd);

	void (*close)(void *ctx);
	void *ctx;
} weft_transport_t;

#ifdef __cplusplus
}
#endif

#endif
