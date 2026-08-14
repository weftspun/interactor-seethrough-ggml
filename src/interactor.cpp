// The see-through interactor: a command in, reply bytes out.
//
// It has no socket and no poll loop. What carried the command here is not
// knowable from this file, and that is the point: `third_party/interactor` is
// the contract, held in a repository neither this nor any transport owns, so a
// transport written later can be paired with this without either depending on
// the other.
//
// SPDX-License-Identifier: Apache-2.0

#include "interactor.h"

#include "weft/cbor.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// ── Bytes in a reply ──────────────────────────────────────────────────────────
//
// `weft/cbor.h` writes rows and scalars because a ward reply is rows and
// scalars; it has no byte-string helper. A layer is a PNG, so major type 2 is
// written here rather than added to the contract: a repository neither side owns
// is not a place to grow a helper for one caller's convenience, and the head is
// three lines.
//
// `weft_cbor_raw` drops every write after the buffer runs out and records it in
// `over`, so the loop needs no bounds check of its own -- the caller checks
// `weft_cbor_over` once, which is the contract's stated way of reporting a
// truncated reply.
static void cbor_bytes(weft_cbor_t *c, const unsigned char *b, size_t n) {
	weft_cbor_head(c, 2, (uint64_t)n);
	for (size_t i = 0; i < n; i++) {
		weft_cbor_raw(c, b[i]);
	}
}

static void say_error(weft_cbor_t *c, const char *what) {
	weft_cbor_map(c, 1);
	weft_cbor_kv_text(c, "error", what);
}

// ── look ──────────────────────────────────────────────────────────────────────
//
// What this interactor is and what it would do, reaching nothing. A caller that
// has just been handed a ring has no other way to find out the defaults its
// commands will inherit, and guessing them from a README that may not match the
// binary is how a caller ends up rendering at the wrong resolution.
static void say_look(weft_cbor_t *c, const PipelineConfig &cfg) {
	weft_cbor_map(c, 7);
	weft_cbor_kv_text(c, "interactor", "seethrough");
	weft_cbor_kv_int(c, "res", cfg.res);
	weft_cbor_kv_int(c, "steps", cfg.steps);
	weft_cbor_kv_int(c, "depth_res", cfg.depth_res);
	weft_cbor_kv_int(c, "seed", (int64_t)cfg.seed);
	weft_cbor_kv_int(c, "command_max", SEETHROUGH_COMMAND_MAX);
	weft_cbor_kv_int(c, "reply_max", SEETHROUGH_REPLY_MAX);
}

// ── render ────────────────────────────────────────────────────────────────────

static void say_render(weft_cbor_t *c, const PipelineConfig &cfg, const char *path) {
	Image input;
	if (!load_image(path, input)) {
		say_error(c, "cannot read input");
		return;
	}

	SeeThroughResult r;
	if (!run_see_through(cfg, input, r)) {
		say_error(c, "render failed");
		return;
	}

	weft_cbor_map(c, 3);

	weft_cbor_text(c, "canvas");
	weft_cbor_array(c, 2);
	weft_cbor_int(c, r.canvas_w);
	weft_cbor_int(c, r.canvas_h);

	// Layers stay in the order run_see_through produced, which is z order back to
	// front. A reader that sorted by tag would composite the character wrong, so
	// the order is the answer and not an incidental property of the array.
	weft_cbor_text(c, "layers");
	weft_cbor_array(c, (uint64_t)r.png_layers.size());
	for (size_t i = 0; i < r.png_layers.size(); i++) {
		weft_cbor_map(c, 4);
		weft_cbor_kv_text(c, "tag", r.png_layers[i].first.c_str());

		weft_cbor_text(c, "xyxy");
		weft_cbor_array(c, 4);
		for (int k = 0; k < 4; k++) {
			weft_cbor_int(c, r.layer_xyxy[i][k]);
		}

		// Depth is a double in the sidecar; CBOR here carries integers, so it goes
		// out in thousandths. A reader dividing by 1000 gets what the PSD's
		// metadata holds -- naming the unit in the key is what keeps that honest.
		weft_cbor_kv_int(c, "depth_median_milli",
				(int64_t)(r.layer_depth_median[i] * 1000.0));

		weft_cbor_text(c, "png");
		cbor_bytes(c, r.png_layers[i].second.data(), r.png_layers[i].second.size());
	}

	// The spans the run already timed. A caller measuring from outside sees queue
	// time and transport time mixed in; these are the stage durations the process
	// itself recorded, which is the only timing this repository trusts.
	weft_cbor_text(c, "spans");
	weft_cbor_array(c, (uint64_t)r.spans.size());
	for (const Span &s : r.spans) {
		weft_cbor_map(c, 2);
		weft_cbor_kv_text(c, "name", s.name.c_str());
		weft_cbor_kv_int(c, "ms",
				(int64_t)((s.end_time_unix_nano - s.start_time_unix_nano) / 1000000ull));
	}
}

// ── the command line ──────────────────────────────────────────────────────────

// `strtok_r` is POSIX and `strtok_s` is MSVC's, and this repository ships MinGW,
// MSVC, Linux and macOS builds from one tree. Splitting on spaces is four lines,
// so it is written here rather than reached for behind a platform #if.
static char *next_token(char **save) {
	char *s = *save;
	while (*s == ' ' || *s == '\t') {
		s++;
	}
	if (!*s) {
		*save = s;
		return nullptr;
	}
	char *start = s;
	while (*s && *s != ' ' && *s != '\t') {
		s++;
	}
	if (*s) {
		*s++ = '\0';
	}
	*save = s;
	return start;
}

static void run_command(const PipelineConfig &base, char *line, weft_cbor_t *c, int *stop) {
	char *save = line;
	const char *verb = next_token(&save);

	if (!verb || !strcmp(verb, "look")) {
		say_look(c, base);
		return;
	}
	if (!strcmp(verb, "quit")) {
		if (stop) {
			*stop = 1;
		}
		weft_cbor_map(c, 1);
		weft_cbor_kv_bool(c, "stopping", 1);
		return;
	}
	if (strcmp(verb, "render")) {
		say_error(c, "no such command");
		return;
	}

	const char *path = next_token(&save);
	if (!path) {
		say_error(c, "render needs a path");
		return;
	}

	// The knobs a caller may set are the ones the CLI exposes and no others. The
	// precision flags stay out on purpose: they change output for a whole run and
	// their defaults were chosen against measured per-layer IoU, so a caller
	// flipping one per command would silently leave that evidence behind.
	PipelineConfig cfg = base;
	for (const char *k; (k = next_token(&save));) {
		const char *v = next_token(&save);
		if (!v) {
			say_error(c, "flag needs a value");
			return;
		}
		if (!strcmp(k, "--res")) {
			cfg.res = atoi(v);
		} else if (!strcmp(k, "--steps")) {
			cfg.steps = atoi(v);
		} else if (!strcmp(k, "--seed")) {
			cfg.seed = strtoull(v, nullptr, 10);
		} else if (!strcmp(k, "--depth-res")) {
			cfg.depth_res = atoi(v);
		} else {
			say_error(c, "no such flag");
			return;
		}
	}

	say_render(c, cfg, path);
}

static size_t seethrough_ask(void *ctx, const char *command, unsigned char *reply, size_t cap,
		int *stop) {
	PipelineConfig *base = (PipelineConfig *)ctx;
	weft_cbor_t c = weft_cbor_to(reply, cap);

	// `run_command` writes NULs into the line to split it, and the command is the
	// transport's to own, so it is copied first. A command that does not fit is
	// refused rather than rendered from its truncation.
	char line[SEETHROUGH_COMMAND_MAX];
	if (snprintf(line, sizeof line, "%s", command) >= (int)sizeof line) {
		say_error(&c, "command too long");
		return c.n;
	}

	run_command(*base, line, &c, stop);

	// A truncated batch decodes as a short one and the reader cannot tell, so an
	// overflowed reply is replaced by the reason rather than sent. If even that
	// does not fit there is nothing to say, and 0 is an answer.
	if (weft_cbor_over(&c)) {
		c = weft_cbor_to(reply, cap);
		say_error(&c, "reply too large");
		return weft_cbor_over(&c) ? 0 : c.n;
	}
	return c.n;
}

weft_interactor_t seethrough_interactor(PipelineConfig *cfg) {
	weft_interactor_t in = { seethrough_ask, cfg };
	return in;
}
