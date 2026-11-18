"""dk: the decomp toolkit. Run `./dk` from the repo root for the command list.

Layers:
    core/       generic: project config, status, docs, sync, and the matching flow (build, verify, floor)
    platforms/  one package per kind of target; a platform may build on another (BASE):
                raw (any binary), spi_nor (on raw: SPI NOR dumps), dotnet (.NET assemblies),
                gc_mwcc (GameCube/Wii + CodeWarrior)
    assets/     project data formats (later)

Project-specific values live only in dk.toml.
"""
