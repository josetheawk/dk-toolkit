# SPI NOR dumps

A full-chip read of a SPI NOR part. The bytes are whatever the host stored; the memory
chip is not the CPU. Everything in [raw.md](raw.md) applies; these rules add the flash medium.

- **Erased cells read as 0xFF.** Why: a NOR erase sets every bit, so a long 0xFF run is unused flash rather than code or a file. Apply: split on 0xFF runs of at least one erase sector (4096 bytes). Some dumps read erased cells differently; set that value before trusting the split.
- **The dump's length names the density, not the vendor.** Why: an 8388608-byte image is a 64 Mbit part, and those parts share a 256-byte page, a 4096-byte sector and a 65536-byte block. The JEDEC id is answered by a command and is not stored in the array. Apply: take geometry from the length, and treat a part name in the project config as a label.
- **Regions start on erase-block boundaries.** Why: the host erases whole sectors or blocks before writing a partition, so partitions are placed on those boundaries. Apply: when a region does not start on a sector boundary, check whether the gap before it is really erased or is data that belongs to the region before it.
