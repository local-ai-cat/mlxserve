# MCP transports

`stdio` and `streamable-http` are implemented as request/response MCP transports.

`sse` in config is accepted only as a compatibility alias for `sse-endpoint`.
This mode performs the legacy HTTP+SSE endpoint handshake: it opens the SSE URL
long enough to discover the POST endpoint, then sends JSON-RPC requests over
HTTP POST and expects each POST response to contain the result. It is not a full
MCP SSE streaming transport because it does not keep the GET event stream open
and correlate responses by JSON-RPC id.

Full persistent SSE response streaming remains a known gap.
