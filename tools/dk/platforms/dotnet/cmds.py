"""dk info / il / strings / resources: read a .NET assembly without running it."""

from __future__ import annotations

import argparse
import hashlib
import struct
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ...core.command import MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from . import resources
from .assembly import Assembly, token
from .listing import method_listing, type_listing, xrefs
from .pe import MACHINES, SUBSYSTEMS, FormatError

# API uses worth a look in triage: (tag, prefixes of "Namespace.Type::Member")
CAPABILITIES = (
    ("mail", ("System.Net.Mail.",)),
    ("network", ("System.Net.", "System.Web.", "System.ServiceModel.")),
    ("process", ("System.Diagnostics.Process::",)),
    ("registry", ("Microsoft.Win32.Registry",)),
    ("files", ("System.IO.File::", "System.IO.FileInfo::", "System.IO.Directory::", "System.IO.FileStream::",
               "System.IO.StreamWriter::")),
    ("crypto", ("System.Security.Cryptography.",)),
    ("code-loading", ("System.Reflection.Assembly::Load", "System.AppDomain::Load", "System.Activator::CreateInstance",
                      "System.CodeDom.Compiler.CodeDomProvider::", "Microsoft.CSharp.CSharpCodeProvider::",
                      "Microsoft.VisualBasic.VBCodeProvider::")),
    ("native", ("System.Runtime.InteropServices.Marshal::",)),
    ("clipboard", ("System.Windows.Forms.Clipboard::",)),
    ("keystrokes", ("System.Windows.Forms.SendKeys::",)),
    ("screen", ("System.Drawing.Graphics::CopyFromScreen",)),
    ("host-info", ("System.Environment::get_UserName", "System.Environment::get_MachineName",
                   "System.Environment::GetFolderPath", "System.Net.Dns::")),
)


def capability_tags(member: str) -> list[str]:
    if member.endswith("Attribute::.ctor"):
        return []                       # applying an attribute does nothing at run time
    return [tag for tag, prefixes in CAPABILITIES if member.startswith(prefixes)]


def _load(cfg: Config) -> Assembly | None:
    if not cfg.original.is_file():
        print(f"error: {cfg.rel(cfg.original)} not found (dk.toml [target] original)")
        return None
    try:
        return Assembly.load(cfg.original)
    except FormatError as e:
        print(f"error: {cfg.rel(cfg.original)}: {e}")
        return None


def _short(name: str, own: str) -> str:
    """Drop the assembly's own root namespace from a name, for compact caller lists."""
    return name[len(own) + 1:] if own and name.startswith(own + ".") else name


# info

def info_lines(asm: Assembly) -> list[str]:
    md, pe, data = asm.md, asm.pe, asm.pe.data
    out = [f"size   {len(data)} bytes"]
    out += [f"{algo:<6} {hashlib.new(algo, data).hexdigest()}" for algo in ("md5", "sha1", "sha256")]
    linked = datetime.fromtimestamp(pe.timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    out.append(f"PE     {'PE32+' if pe.pe32plus else 'PE32'} {MACHINES.get(pe.machine, hex(pe.machine))} "
               f"{SUBSYSTEMS.get(pe.subsystem, pe.subsystem)}, linked {linked}")
    out.append(f"CLR    header {asm.cli.runtime[0]}.{asm.cli.runtime[1]}, metadata {md.version}, "
               f"flags {' '.join(asm.cli.flag_names()) or '-'}")
    entry = asm.cli.entry_token
    if entry >> 24 == 0x06:
        out.append(f"entry  {asm.method_signature(entry & 0xFFFFFF)}")
    for row in md.rows("Module"):
        out.append(f"module {md.string(row['Name'])}  mvid {md.guid(row['Mvid'])}")

    for row in md.rows("Assembly"):
        version = ".".join(str(row[k]) for k in ("MajorVersion", "MinorVersion", "BuildNumber", "RevisionNumber"))
        out += ["", f"assembly {md.string(row['Name'])} {version}"]
        for attr, args in sorted(asm.custom_attributes(("Assembly", 1))):
            out.append(f"  [{attr.removesuffix('Attribute')}] {', '.join(args)}")

    out += ["", "references:"]
    for row in md.rows("AssemblyRef"):
        version = ".".join(str(row[k]) for k in ("MajorVersion", "MinorVersion", "BuildNumber", "RevisionNumber"))
        out.append(f"  {md.string(row['Name'])} {version}")
    for row in md.rows("ModuleRef"):
        out.append(f"  module {md.string(row['Name'])}")

    out += ["", f"types ({len(md.rows('TypeDef'))}):"]
    for rid in range(1, len(md.rows("TypeDef")) + 1):
        extends = asm.type_name(md.row("TypeDef", rid)["Extends"])
        out.append(f"  {asm.typedef_name(rid)}" + (f" : {extends}" if extends else "") +
                   f"   ({len(asm.methods_of(rid))} methods, {len(asm.fields_of(rid))} fields)")

    res = asm.manifest_resources()
    if res:
        out += ["", f"manifest resources ({len(res)}):"]
        out += [f"  {name}  " + (f"{len(blob)} bytes" if blob is not None else "(in another file)") for name, blob in res]

    pinvoke = md.rows("ImplMap")
    if pinvoke:
        out += ["", f"P/Invoke ({len(pinvoke)}):"]
        for row in pinvoke:
            table, rid = row["MemberForwarded"]
            module = md.row("ModuleRef", row["ImportScope"])
            target = asm.method_name(rid) if table == "MethodDef" else asm.field_name(rid)
            out.append(f"  {target} → {md.string(module['Name']) if module else '?'}!{md.string(row['ImportName'])}")

    used, problems = xrefs(asm)
    own = md.string(md.rows("Assembly")[0]["Name"]) if md.rows("Assembly") else ""
    flagged, by_type = [], defaultdict(set)
    for rid in range(1, len(md.rows("MemberRef")) + 1):
        member = asm.memberref_name(rid)
        parent, _, name = member.rpartition("::")
        by_type[parent].add(name)
        tags = capability_tags(member)
        if tags:
            callers = sorted(_short(asm.method_name(m), own) for m in used.get(token("MemberRef", rid), ()))
            flagged.append((tags, member, callers))
    out += ["", "capabilities (external API uses worth reading):"]
    if not flagged and not pinvoke:
        out.append("  (none matched)")
    for tags, member, callers in sorted(flagged, key=lambda f: (f[0], f[1])):
        out.append(f"  {','.join(tags):<16} {member}")
        out += [f"  {'':<16}   ← {c}" for c in callers] or [f"  {'':<16}   ← (not called)"]
    out += ["", f"member references by type ({len(by_type)}):"]
    out += [f"  {t}: {', '.join(sorted(names))}" for t, names in sorted(by_type.items())]
    if problems:
        out += ["", f"IL decode problems ({len(problems)}):"] + [f"  {p}" for p in problems]
    return out


def run_info(cfg: Config, args: argparse.Namespace) -> int:
    asm = _load(cfg)
    if asm is None:
        return USAGE
    print(cfg.rel(cfg.original))
    for line in info_lines(asm):
        print(line)
    hint("strings", "il <type or method>", "resources")
    return OK


# il

def _select(asm: Assembly, patterns: list[str]) -> tuple[list[int], list[int]]:
    """(types, methods) matching any pattern; a matched type takes all its methods."""
    types = [t for t in range(1, len(asm.md.rows("TypeDef")) + 1)
             if not patterns or any(p == asm.typedef_name(t) or p == asm.typedef_name(t).rpartition(".")[2] for p in patterns)]
    taken = {m for t in types for m in asm.methods_of(t)}
    methods = [m for m in range(1, len(asm.md.rows("MethodDef")) + 1)
               if patterns and m not in taken and any(p in asm.method_name(m) for p in patterns)]
    return types, methods


def run_il(cfg: Config, args: argparse.Namespace) -> int:
    asm = _load(cfg)
    if asm is None:
        return USAGE
    types, methods = _select(asm, args.names)
    if not types and not methods:
        print(f"error: nothing matches {' '.join(args.names)} (type name, or part of Type::Method)")
        hint("info")
        return USAGE
    chunks = [(asm.typedef_name(t), type_listing(asm, t)) for t in types]
    chunks += [(asm.method_name(m), method_listing(asm, m)) for m in methods]
    incomplete = sum(1 for _, lines in chunks for line in lines if "decode stopped" in line or "body unreadable" in line)
    if args.out:
        out_dir = Path(args.out) if args.out != "-default-" else cfg.root / cfg.platform_options.get("il", "build/il")
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, lines in chunks:
            (out_dir / f"{resources.safe_name(name)}.il").write_text("\n".join(lines) + "\n")
        print(f"wrote {len(chunks)} listing(s) to {cfg.rel(out_dir)}")
    else:
        for i, (_, lines) in enumerate(chunks):
            print(("\n" if i else "") + "\n".join(lines))
    if incomplete:
        print(f"MISMATCH: {incomplete} method(s) could not be fully decoded")
        return MISMATCH
    hint("strings", "resources")
    return OK


# strings

def run_strings(cfg: Config, args: argparse.Namespace) -> int:
    asm = _load(cfg)
    if asm is None:
        return USAGE
    used, _ = xrefs(asm)
    own = asm.md.string(asm.md.rows("Assembly")[0]["Name"]) if asm.md.rows("Assembly") else ""
    count = 0
    for offset, text in asm.md.user_strings():
        callers = sorted(_short(asm.method_name(m), own) for m in used.get(0x70000000 | offset, ()))
        if args.unreferenced and callers:
            continue
        count += 1
        print(f"{offset:#07x} {asm.token_name(0x70000000 | offset)}")
        for c in callers or ["(unreferenced)"]:
            print(f"        ← {c}")
    print(f"{count} user string(s)")
    hint("il <method>")
    return OK


# resources

def run_resources(cfg: Config, args: argparse.Namespace) -> int:
    asm = _load(cfg)
    if asm is None:
        return USAGE
    out_dir = None
    if args.extract:
        out_dir = Path(args.extract) if args.extract != "-default-" else cfg.root / cfg.platform_options.get("resources", "build/resources")
    written = 0
    for name, blob in asm.manifest_resources():
        if blob is None:
            print(f"{name}  (stored in another file)")
            continue
        try:
            entries = resources.parse(blob)
        except (FormatError, IndexError, ValueError, struct.error) as e:
            kind = resources.sniff(blob)[1]
            print(f"{name}  {len(blob)} bytes, {kind}" + (f" (not .resources: {e})" if name.endswith(".resources") else ""))
            entries = None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / resources.safe_name(name)).write_bytes(blob)
            written += 1
        if entries is None:
            continue
        print(f"{name}  {len(blob)} bytes, {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}")
        for entry in entries:
            value = f" = {entry.value!r}" if entry.value is not None else f"  {len(entry.raw)} bytes"
            print(f"  {entry.name}  [{entry.type}]{value}")
            for i, (ext, desc, data) in enumerate(resources.contained_files(entry)):
                print(f"      holds {desc}, {len(data)} bytes")
                if out_dir is not None:
                    sub = out_dir / resources.safe_name(name.removesuffix(".resources"))
                    sub.mkdir(exist_ok=True)
                    suffix = f".{i}" if i else ""
                    (sub / f"{resources.safe_name(entry.name)}{suffix}.{ext}").write_bytes(data)
                    written += 1
    if out_dir is not None:
        print(f"wrote {written} file(s) to {cfg.rel(out_dir)}")
    hint("strings", "il <type>")
    return OK


def _configure_il(p: argparse.ArgumentParser) -> None:
    p.add_argument("names", nargs="*", help="type names (Ns.Type or Type) or parts of Type::Method; none = everything")
    p.add_argument("--out", nargs="?", const="-default-",
                   help="write one .il file per type/method (default dir: dk.toml [platform] il)")


def _configure_strings(p: argparse.ArgumentParser) -> None:
    p.add_argument("--unreferenced", action="store_true", help="only strings no method loads")


def _configure_resources(p: argparse.ArgumentParser) -> None:
    p.add_argument("--extract", nargs="?", const="-default-",
                   help="write each resource and the files inside it (default dir: dk.toml [platform] resources)")


INFO = Command("info", "analysis", "assembly overview: hashes, headers, references, types, resources, capabilities",
               run_info)
IL = Command("il", "analysis", "IL listing with tokens resolved to names (exit 1 if a body can't be fully decoded)",
             run_il, _configure_il)
STRINGS = Command("strings", "analysis", "user strings (#US) with the methods that load each one",
                  run_strings, _configure_strings)
RESOURCES = Command("resources", "analysis", "manifest resources, their entries and the files inside them",
                    run_resources, _configure_resources)
