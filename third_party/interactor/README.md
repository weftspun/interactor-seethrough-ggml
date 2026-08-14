# fabric-interactor

The interactor contract. A command in, reply bytes out, and nothing about how the command
arrived.

```
include/weft/interactor.h   an interactor, and a transport
include/weft/cbor.h         what a reply is made of
src/cbor.c                  the writer
proof/roundtrip.c           an interactor driven with no transport at all
```

## What an interactor is

An interactor answers one command. It has no socket, no poll loop, and no idea what carried
the command to it.

A transport is bytes to and from a wire. It has no idea what the command means.

A **service** is the composition: a state to act on, one or more interactors over it, and one
or more transports in front of them.

```
    interactor          transport            service
    command → bytes     bytes ↔ wire         state + interactors + transports
```

## Why this is its own repository

Because a contract that lives in either side makes the other its dependent.

Put it in the transport and every interactor vendors a QUIC implementation to get at a type.
Put it in the interactor and a transport cannot be written before the interactor it will
carry. Neither is composable, and composable is the whole point: a transport written today
should pair with an interactor that does not exist yet.

`fabric-harness` is the same argument about the bus and the limits. This is the argument about
the command.

## The boundary is held by the linker

`weft::interactor` links nothing. Not a socket library, not a QUIC implementation, not a
database client. If one ever appears in its `target_link_libraries`, the separation is gone,
and a build failure finds that out faster than a reviewer does.

`proof/roundtrip.c` is the other half: an interactor built, asked, and checked, with no
transport anywhere in the test. If that file ever needs a socket to run, this repository has
failed at its one job.

## What is deliberately absent

**Framing.** A length prefix belongs to whichever transport needs one. A byte stream does; a
WebTransport stream does not, because its FIN is already the boundary. A writer that always
prefixed would put a stream's framing inside a datagram.

**Dispatch.** Which interactor answers which command is a service's decision. A contract that
routed would be a framework.

**Authority.** An interactor checks its own rules on receipt, because a filtered menu is a
convenience and never an authorization. Who may run a command at all is a question for the
relations, and those are not here.

## Building

```sh
cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure
```

Vendored into a service with `git subtree add --prefix=thirdparty/interactor`, the proof does
not build: a service's build is not the place to compile somebody else's test.
