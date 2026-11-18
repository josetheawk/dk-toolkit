# Raw images: firmware, ROMs, flash and disk dumps

A flat byte array with no file format around it. Read this before mapping one; `./dk map` applies these rules.

## Layout

- **Split the image on long runs of one fill byte first.** Why: unused space is 0xFF (erased flash) or 0x00 (padding), and everything between the runs is a region with one purpose (a boot stage, a payload, a filesystem, a config block). Apply: a run of at least one erase sector (4096 bytes) separates regions; shorter runs are data inside a region.
- **A region that repeats an earlier one byte for byte is a redundant copy.** Why: configuration is commonly kept twice as a backup, and the copy adds nothing to read. Apply: compare region hashes and read one copy.

## Code

- **An ARM classic vector table gives the image's link address.** Why: the reset word is a `b` or `ldr pc, [pc, #imm]`, and the next seven words are `ldr pc, [pc, #imm]` whose literal pool holds absolute handler addresses. Apply: the link address is that literal with its low bits cleared (try 64 KiB, then 1 MiB, then 16 MiB) so every handler lands on a non-erased word in the image; then virtual address = link + file offset − table offset.
- **A second table at another offset with the same link address is a second boot stage built from the same tree.** Why: stages share their start-up code, so the tables match while the bodies diverge. Apply: diff the two stages from the table onward; the shared prefix is the common start-up code.
- **Literal pools follow the vector words.** Why: a linear sweep decodes the pool's addresses as instructions. Apply: stop reading code at the first literal the table points through.

## Containers

- **A magic alone is not a container.** Why: the same bytes occur as a string in the code that parses the container, and at random inside compressed data. Apply: accept a hit only when its header describes a span that fits the image (non-zero lengths, a CRC that checks, a stream that decodes to its end).
- **`gziphead` is a 16-byte header, not a gzip member.** The header is `u32` compressed length, `u32` uncompressed length, then the eight bytes `gziphead`, and the compressed bytes follow. Why: the payload is a gzip member only when it starts with `1f 8b`. Apply: record both lengths and carve the payload; do not run gzip on it unless the member magic is there.

## JFFS2

- **A JFFS2 hit is a chain of nodes, not the two magic bytes.** Nodes are 4-byte aligned, magic `0x1985` in the image's byte order, with a known node type and a `totlen` that stays inside the image. Why: the magic occurs inside compressed payloads. Apply: list names only from a chain of at least two nodes that includes a printable directory name.
- **The node header CRC is JFFS2's crc32, not zlib's.** Why: JFFS2 seeds the CRC with 0 and does not invert the result, so `zlib.crc32` of the first 8 bytes never matches. Apply: compute `zlib.crc32(buf, 0xFFFFFFFF) ^ 0xFFFFFFFF`; a mismatch with that formula means a damaged node.
- **A directory entry's name is `nsize` bytes at offset 40; an inode's data starts at offset 68.** `csize` is the stored length and `dsize` the inflated length; compression type 6 is zlib, 0 is stored. Why: the directory node has the name and not the data. Apply: order each inode's fragments by file offset, inflate each to `dsize`, and build paths from parent inode numbers.
