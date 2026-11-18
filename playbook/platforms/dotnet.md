# .NET (CLI) assemblies

## Identify

- **A PE with a CLI header is .NET: decompile it from metadata, don't match it.** Why: the metadata keeps every type, method, field, signature and string by name, and IL decodes without guessing. The split/compiler-search/matching pipeline buys nothing until a recompile-and-compare flow exists. Detect: data directory 14 (COM descriptor) is non-zero; `file` says "Mono/.Net assembly". Apply: `platform = "dotnet"`, then `./dk info`.
- **VB.NET has tells; reconstruct in VB, not C#.** The tells are a `Microsoft.VisualBasic` reference, `My.MyProject` with nested `MyForms`/`MyWebServices`/`ThreadSafeObjectProvider`, `_Name` fields behind `Friend WithEvents` properties, and `ProjectData.SetProjectError`/`ClearProjectError` in every `Catch ex`. Why: those types and calls are compiler output, and C# would need them all spelled out. Apply: skip the `My.*` types, which the compiler and Visual Studio generate. Write the forms and modules only.

## Read

- **Start from external member references, grouped by namespace, with their callers.** Why: a small managed program's whole behaviour (mail, network, process, registry, files) shows in which methods call which framework APIs. In practice one handler held all of it. Apply: the capabilities section of `./dk info`, then `./dk il <that method>`.
- **Tie each user string to the method that loads it.** Why: `#US` strings have no owner; the `ldstr` scan gives one, and hosts, credentials and lure text then sit together under one method. Apply: `./dk strings`.
- **Count exception clauses as `DataSize // clause size`.** Why: compilers disagree on whether a small EH section's `DataSize` includes its 4-byte header (12 or 16 for one clause). `(DataSize - 4) / 12` silently drops the only clause; the runtime divides. Apply: advance past the section by `4 + count * clause size`.
- **WinForms images live in BinaryFormatter blobs inside `.resources`.** Why: they are most of a WinForms binary's size, and a `Bitmap` or `Icon` entry wraps the file in serialization records. Apply: carve `ArraySinglePrimitive` byte arrays (record `0x0F`, element type `2`), then identify each by its magic bytes (`./dk resources --extract`).

## Handle

- **Keep a malware sample in the git-ignored `orig/`, with no execute bit, and never run it.** Why: with wine installed, a double-click runs a Windows binary. Static reading answers everything a credential stealer does. Apply: `chmod a-x`, and open extracted resources as data only.
