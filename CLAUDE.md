# LZ4 Implementation Guardrails

Follow these rules for future changes to the local LZ4 block codec.

1. Enforce output bounds on every write path.
- Check `max_output` before copying literals.
- Check `max_output` before copying match bytes.

2. Keep variable-length parsing centralized.
- Use one helper for literal and match extension lengths.
- For extension bytes, continue reading while byte value is `255`.
- Do not stop after a single `255`; that truncates long lengths and corrupts decoding.
- Use checked addition while accumulating extension bytes to avoid length overflow.

3. Keep match candidate checks strict before slicing.
- Use an explicit sentinel in the hash table (not implicit `0`).
- Store table positions as `usize` (or reject oversized inputs if using narrower types).
- Reject candidates that are not strictly behind the current position.
- Reject candidates beyond `u16::MAX` offset.
- Verify 4-byte prefix equality only after candidate validity checks.

4. Avoid unsigned-underflow hazards in match math.
- Use `saturating_sub`/checked math for match ceilings.
- Do not rely on undocumented constant relationships to keep subtractions valid.

5. Preserve LZ4 block compatibility.
- Keep interop tests with `lz4_flex` for both directions:
  - our compress -> their decompress
  - their compress -> our decompress

6. Keep malformed-input tests.
- Include tests for zero/invalid offsets and truncated length streams.
- Include at least one test that proves `max_output` is enforced on literal-heavy input.

7. Keep overlap-safe and fast match copy behavior.
- Copy matches with chunked `extend_from_within` so overlapping offsets stay correct.
- Avoid byte-at-a-time loops unless needed for correctness.
