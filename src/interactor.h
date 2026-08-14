// The see-through interactor: a command in, reply bytes out.
//
// It has no socket, no poll loop and no idea what carried the command here --
// `third_party/interactor` is the contract that lets it be composed with a
// transport written before this file existed (RFD 0111: this repository is an
// interactor, and a transport layer is a separate process).
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "pipeline.h"

#include "weft/interactor.h"

// The longest command this interactor will read. A longer one is refused rather
// than truncated, because a truncated command is a different command: cutting
// "--steps 30" short would silently render at the default step count and the
// caller could not tell from the reply.
#define SEETHROUGH_COMMAND_MAX 1024

// The largest reply, and the size a transport must be prepared to carry.
//
// Measured, not guessed: a 1280px run of the in-repo sample produces 27 layers
// totalling 1.4 MB of PNG plus a few hundred bytes of metadata. The bound is set
// far above that because layer bytes scale with how much of the canvas each tag
// actually covers -- an ornate character with every accessory tag populated is
// several times the sample, and a reply that overflowed would decode as a short
// batch that a reader cannot tell from a complete one (weft_cbor_over exists for
// exactly that reason, and `ask` reports it as an error rather than a reply).
//
// This is deliberately not the 256 KB a ward reply uses. Sizing an image
// pipeline's ring from a database interactor's constant is how a chunk size
// becomes wrong by three orders of magnitude.
#define SEETHROUGH_REPLY_MAX (64u << 20)

// Binds the pipeline to the contract. The returned interactor borrows `cfg` and
// does not own it.
//
// Commands (one line, verb first):
//
//   render <path> [--res N] [--steps N] [--seed N] [--depth-res N]
//                              the pipeline, answered whole. Reply is a CBOR map
//                              {canvas:[w,h], layers:[{tag,xyxy,depth_median,png}],
//                               spans:[{name,ms}]}.
//   look                       what this interactor is and what it would do:
//                              defaults, bounds and device. Reaches nothing.
//   quit                       asks the service to wind down (sets *stop).
//
// `render` takes a path rather than inline image bytes: a command is a
// NUL-terminated string, so it cannot carry a PNG, and materialising the input is
// the transport's job -- it is the side that spoke to the network.
weft_interactor_t seethrough_interactor(PipelineConfig *cfg);
