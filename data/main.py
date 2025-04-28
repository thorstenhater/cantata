#!/usr/bin/env python3

from pathlib import Path
import subprocess as sp
import site
import sys
import re
from argparse import ArgumentParser

# process arguments
ap = ArgumentParser()
ap.add_argument(
    "-e",
    "--use-venv",
    action="store_true",
    default=False,
    help="Setup and use virtual env.",
)
ap.add_argument(
    "-s",
    "--use-stats",
    action="store_true",
    default=False,
    help="Collect and print simulation statistics.",
)
ap.add_argument(
    "-t",
    "--use-times",
    action="store_true",
    default=False,
    help="Collect and print simulation timings.",
)
ap.add_argument("-m", "--use-mpi", action="store_true", default=False, help="Use MPI.")
ap.add_argument("-g", "--use-gpu", action="store_true", default=False, help="Use GPU.")
ap.add_argument(
    "-p",
    "--plot",
    action="store_true",
    default=False,
    help="Plot data in addition to storing.",
)

args = ap.parse_args()

have_timing = args.use_times
have_stats = args.use_stats
have_venv = args.use_venv
have_mpi = args.use_mpi
have_gpu = args.use_gpu
have_plots = args.plot

# version meta data
cur_version = [0, 11, 0]
nxt_version = [0, 12, 0]
cur_version_str = f"{cur_version[0]}.{cur_version[1]}.{cur_version[2]}"
nxt_version_str = "nxt_version[0]}.{nxt_version[1]}.{nxt_version[2]}"

here = Path(__file__).parent

# Setup venv, if needed
if have_venv:
    env = here / ".env"
    if not Path(env).exists():
        print(f"No .env found, setting up {here}/.env")
        sp.run(
            f'bash -c "/usr/bin/env python3 -mvenv {env} && source {env}/bin/activate && pip3 install arbor=={cur_version_str} matplotlib numpy pandas cbor2"',
            shell=True,
            check=True,
        )

    # activate our env manually and ensure it's in front
    version = re.match(r"(\d+\.\d+)\..+", sys.version).group(1)
    site_packages = here / ".env" / "lib" / f"python{version}" / "site-packages"
    old_path = list(sys.path)
    site.addsitedir(site_packages)
    sys.real_prefix = sys.prefix
    sys.prefix = here
    sys.path[:0] = [item for item in list(sys.path) if item not in old_path] + sys.path

from collections import defaultdict
import arbor as A
from arbor import units as U
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter as pc
from cbor2 import load as load_data
from math import ceil

# check arbor version
ver = re.match(r"(\d+)\.(\d+)\.(\d+)(-\w+)?", A.__version__)
if ver:
    mj, mn, pt, sf = ver.groups()
    ver = [int(mj), int(mn), int(pt)]
    assert (
        cur_version <= ver <= nxt_version
    ), f"Arbor {cur_version_str} <= version <= {nxt_version_str} is required, got {A.__version__}"
else:
    print(f"Couldn't parse version {A.__version__}")
    exit(-42)


def load_morphology(path):
    sfx = path.suffix
    if sfx == ".swc":
        try:
            res = A.load_swc_neuron(path)
            return res.morphology
        except Exception as _:
            pass
        try:
            res = A.load_swc_arbor(path)
            return res.morphology
        except Exception as _:
            raise RuntimeError(
                f"Could load {path} neither as NEURON nor Arbor flavour."
            )

    elif sfx == ".asc":
        return A.load_asc(path).morphology
    elif sfx == ".nml":
        nml = A.neuroml(path)
        if len(nml.morphology_ids()) == 1:
            return nml.morphology(nml.morphology_ids()[0]).morphology
        else:
            raise RuntimeError(f"NML file {path} contains multiple morphologies.")
    else:
        raise RuntimeError(f"Unknown morphology file type {sfx}")


class Timing:
    def __init__(self):
        self.timings = defaultdict(lambda: 0.0)
        self.times = defaultdict(lambda: 0.0)
        self.children = defaultdict(set)

    def tic(self, key):
        self.timings[key] -= pc()

    def toc(self, key):
        self.timings[key] += pc()

    def show_times(self, root, prefix):
        lbl = f"{' '*prefix}* {root}"
        print(f"{lbl:<37}{self.times[root]:0.3f}")
        for child in self.children[root]:
            self.show_times(child, prefix + 2)

    def report(self):
        for path, time in self.timings.items():
            last = "Total"
            self.times[last] += time
            for k in path.split("/"):
                self.children[last].add(k)
                last = k
                self.times[k] += time
        print(
            """
Timings
==========
    """
        )
        self.show_times("Total", 0)


class TimingNull:
    def __init__(self):
        pass

    def tic(self, _):
        pass

    def toc(self, _):
        pass

    def report(self):
        pass


if have_timing:
    timing = Timing()
else:
    timing = TimingNull()


def open_sim():
    with open(here / "dat/sim.cbor", "rb") as fd:
        return load_data(fd)


def close_sim(raw):
    del raw


def read_int_dict(raw, key):
    res = raw[key]
    assert isinstance(res, dict)
    return res


def read_dict(raw, key):
    res = raw[key]
    assert isinstance(res, dict)
    return res


def read_array(raw, key):
    res = raw[key]
    assert isinstance(res, list)
    return res


def read_int(raw, key):
    res = raw[key]
    assert isinstance(res, int)
    return res


def read_float(raw, key):
    res = raw[key]
    assert isinstance(res, float)
    return res


class recipe(A.recipe):
    def __init__(self):
        A.recipe.__init__(self)

        data = open_sim()

        # Some initial, global data
        self.N = read_int(data, "size")
        self.T = read_float(data, "time")
        self.dt = read_float(data, "time_step")
        self.threshold = read_float(data, "spike_threshold")

        # gid -> (morphology id, acc id)
        self.gid_to_bio = read_int_dict(data, "cell_bio_ids")
        # gid -> lif cell data
        self.gid_to_lif = read_int_dict(data, "gid_to_lif")
        # gid -> virtual cell data
        self.gid_to_vrt = read_int_dict(data, "gid_to_vrt")
        # morphology id -> morphology resource file
        self.mid_to_mrf = read_array(data, "morphology")
        # cell id -> decor file
        self.cid_to_acc = read_array(data, "decoration")
        # gid -> cell kind
        self.gid_to_kid = read_array(data, "cell_kind")
        # gid -> cell metadata
        self.gid_to_meta = read_array(data, "metadata")
        # gid -> incoming connections
        self.gid_to_inc = read_int_dict(data, "incoming_connections")
        # gid -> synapse
        self.gid_to_syn = read_int_dict(data, "synapses")
        # gid -> iclamps. NOTE must only be set if kind[gid] == cable
        self.gid_to_icp = read_int_dict(data, "current_clamps")
        # probes
        self.gid_to_prb = read_int_dict(data, "probes")
        # cell kind specific counts
        self.kind_to_count = read_array(data, "count_by_kind")

        # convert raw data into things we handle
        self.cable_props = A.neuron_cable_properties()
        properties = read_dict(data, "cable_cell_globals")
        self.cable_props.catalogue.extend(A.allen_catalogue(), "")
        if properties:
            self.cable_props.set_property(
                Vm=properties["v_init"] * U.mV, tempK=properties["celsius"] * U.Celsius
            )
        max_extent = read_float(data, "max_cv_length")
        if max_extent:
            self.cv_policy = A.cv_policy_max_extent(max_extent)
        else:
            self.cv_policy = A.cv_policy_single()
        # clean-up...
        close_sim(data)
        # caches some data
        self.cable_data = {}

    def cell_kind(self, gid):
        kind = self.gid_to_kid[gid]
        if kind == 0:
            return A.cell_kind.cable
        elif kind == 1:
            return A.cell_kind.lif
        elif kind == 2:
            return A.cell_kind.spike_source
        else:
            raise RuntimeError("Unknown cell kind")

    def num_cells(self):
        return self.N

    def connections_on(self, gid):
        if not gid in self.gid_to_inc:
            return []
        return [
            A.connection((src, f"src-{lbl}"), f"syn-{tgt}", w, max(d, self.dt) * U.ms)
            for src, lbl, tgt, w, d in self.gid_to_inc[gid]
        ]

    def global_properties(self, kind):
        if kind == A.cell_kind.cable:
            return self.cable_props
        return None

    def cell_description(self, gid):
        kind = self.gid_to_kid[gid]
        if kind == 0:
            return self.make_cable_cell(gid)
        elif kind == 1:
            return self.make_lif_cell(gid)
        elif kind == 2:
            return self.make_vrt_cell(gid)
        else:
            raise RuntimeError("Unknown cell kind")

    def probes(self, gid):
        res = []
        if gid in self.gid_to_prb:
            kind = self.cell_kind(gid)
            for loc, var, tag in self.gid_to_prb[gid]:
                tag = f"probe-{tag}"
                if kind == A.cell_kind.cable:
                    loc = f'(on-components 0.5 (region "{loc}"))'
                    if var == "voltage":
                        res.append(A.cable_probe_membrane_voltage(loc, tag))
                    elif var.endswith("i"):
                        res.append(
                            A.cable_probe_ion_int_concentration(loc, var[:-1], tag)
                        )
                    elif var.endswith("o"):
                        res.append(
                            A.cable_probe_ion_ext_concentration(loc, var[:-1], tag)
                        )
                    else:
                        print(f"[UNSUPPORTED] Skipping cable probe: {var}.")
                elif kind == A.cell_kind.lif:
                    if var == "voltage":
                        res.append(A.lif_probe_voltage(tag))
                    else:
                        raise RuntimeError(f"Probing var={var} not yet implemented")
                else:
                    raise RuntimeError(f"Probing cell of kind={kind} not implemented")
        return res

    def make_cable_cell(self, gid):
        mrf, dec = self.load_cable_data(gid)
        pwl = A.place_pwlin(mrf)
        lbl = A.label_dict().add_swc_tags()
        # NOTE in theory we could have more and in other places...
        dec.place(
            "(location 0 0.5)", A.threshold_detector(self.threshold * U.mV), "src-0"
        )
        if gid in self.gid_to_syn:
            for x, y, z, synapse, params, tag in self.gid_to_syn[gid]:
                loc, _ = pwl.closest(x, y, z)
                dec.place(str(loc), A.synapse(synapse, **params), f"syn-{tag}")
        if gid in self.gid_to_icp:
            for loc, delay, duration, amplitude, tag in self.gid_to_icp[gid]:
                dec.place(
                    loc,
                    A.iclamp(
                        tstart=delay * U.ms,
                        duration=duration * U.ms,
                        current=amplitude * U.nA,
                    ),
                    f"ic-{tag}",
                )
        # Try to determine whether NRN would use Nernst.
        # Arbor applies the Nernst rule globally, not per region.
        regs = dict()
        for reg, item in dec.paintings():
            if reg not in regs:
                regs[reg] = dict()
            if isinstance(item, A.density):
                if item.mech.values.get("gbar") == 0:
                    continue
                name = item.mech.name
                mech = self.cable_props.catalogue[name]
                for ion, data in mech.ions.items():
                    if ion not in regs[reg]:
                        # r_conc, w_conc, r_erev, w_erew
                        regs[reg][ion] = [False, False, False, False]
                    regs[reg][ion][0] |= data.read_int_con | data.read_ext_con
                    regs[reg][ion][1] |= data.write_int_con | data.write_ext_con
                    regs[reg][ion][2] |= data.read_rev_pot
                    regs[reg][ion][3] |= data.write_rev_pot
        for reg, ions in regs.items():
            for ion, [rc, wc, rp, wp] in ions.items():
                if wc and rp and not wp:
                    dec.set_ion(
                        ion,
                        int_con=self.cable_props.ions[ion].internal_concentration
                        * U.mM,
                        ext_con=self.cable_props.ions[ion].external_concentration
                        * U.mM,
                        method=f"nernst/x={ion}",
                    )
                elif rc and rp and not wp:
                    # TODO Set eX once using Nernst
                    dec.set_ion(
                        ion,
                        int_con=self.cable_props.ions[ion].internal_concentration
                        * U.mM,
                        ext_con=self.cable_props.ions[ion].external_concentration
                        * U.mM,
                        method=f"nernst/x={ion}",
                    )

        return A.cable_cell(mrf, dec, lbl, self.cv_policy)

    def make_lif_cell(self, gid):
        cell = A.lif_cell("src-0", "syn-0")
        data = self.gid_to_lif[gid]
        # setup the cell to adhere to NEURON's defaults
        cell.C_m = 0.6 * data["cm"] * U.pF
        cell.tau_m = data["tau"] * U.ms
        cell.E_L = data["U_neutral"] * U.mV
        cell.E_R = data["U_reset"] * U.mV
        cell.V_m = data["U_0"] * U.mV
        cell.V_th = data["U_th"] * U.mV
        cell.t_ref = data["t_ref"] * U.ms
        return cell

    def make_vrt_cell(self, gid):
        return A.spike_source_cell(
            "src-0", A.explicit_schedule([t * U.ms for t in self.gid_to_vrt[gid]])
        )

    def load_cable_data(self, gid):
        mid, cid = self.gid_to_bio[gid]
        if gid not in self.cable_data:
            timing.tic("build/simulation/io")
            mrf = load_morphology(here / "mrf" / self.mid_to_mrf[mid])
            dec = A.load_component(here / "acc" / self.cid_to_acc[cid]).component
            self.cable_data[gid] = (mrf, dec)
            timing.toc("build/simulation/io")
        mrf, dec = self.cable_data[gid]
        return mrf, A.decor(dec) # NOTE copy that decor!!


timing.tic("build/recipe")
rec = recipe()
timing.toc("build/recipe")

timing.tic("build/simulation")

comm = None
if have_mpi:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

gpu = None
if have_gpu:
    gpu = 0
    if comm:
        gpu = A.env.find_private_gpu(comm)

ctx = A.context(gpu_id=gpu, mpi=comm)
hints = {}
for kind, tag in zip(
    [A.cell_kind.cable, A.cell_kind.lif, A.cell_kind.spike_source], [0, 1, 2]
):
    if tag in rec.kind_to_count:
        hints[kind] = A.partition_hint(
            cpu_group_size=int(ceil(rec.kind_to_count[tag] / ctx.threads))
        )
ddc = A.partition_load_balance(rec, ctx, hints)
sim = A.simulation(rec, context=ctx, domains=ddc)
timing.toc("build/simulation")

timing.tic("build/sampling")
sim.record(A.spike_recording.all)

schedule = A.regular_schedule(tstart=0 * U.ms, dt=rec.dt * U.ms)
handles = {
    (gid, tag): sim.sample((gid, f"probe-{tag}"), schedule=schedule)
    for gid, prbs in rec.gid_to_prb.items()
    for _, _, tag in prbs
}
timing.toc("build/sampling")

timing.tic("run")
sim.run(rec.T * U.ms, rec.dt * U.ms)
timing.toc("run")

timing.tic("output/spikes")
spikes = sim.spikes()
df = pd.DataFrame(
    {
        "time": spikes["time"],
        "gid": spikes["source"]["gid"],
        "lid": spikes["source"]["index"],
    }
)
df["kind"] = df["gid"].map(lambda i: rec.gid_to_kid[i])
df["population"] = df["gid"].map(lambda i: rec.gid_to_meta[i]["population"])
df["type"] = df["gid"].map(lambda i: rec.gid_to_meta[i]["type_id"])
df.to_csv(here / "out" / "spikes.csv")
timing.toc("output/spikes")

timing.tic("output/samples")
for (gid, tag), handle in handles.items():
    dfs = []
    for data, meta in sim.samples(handle):
        if isinstance(meta, list):
            columns = list(map(str, meta))
        else:
            columns = [str(meta)]
        assert data.shape[1] == len(columns) + 1
        dfs.append(pd.DataFrame(data=data[:, 1:], columns=columns, index=data[:, 0]))
    if not dfs:
        print(f"[WARN] No data collected for tag '{tag}' on cell {gid}")
        continue
    df = pd.concat(dfs, axis=1)
    df.index.name = "t/ms"
    df.to_csv(here / "out" / f"gid_{gid}-tag_{tag}.csv")

    if have_plots:
        fg, ax = plt.subplots()
        df.plot(ax=ax)
        fg.savefig(here / "out" / f"gid_{gid}-tag_{tag}.pdf")
        plt.close(fg)
timing.toc("output/samples")

if have_stats:
    timing.tic("output/statistics")
    N = rec.num_cells()

    cells = defaultdict(lambda: defaultdict(lambda: 0))
    spike = defaultdict(lambda: defaultdict(lambda: 0))
    conns = defaultdict(lambda: 0)

    for gid in range(N):
        meta = rec.gid_to_meta[gid]
        pop = meta["population"]
        kind = meta["type_id"]
        cells[pop][kind] += 1
        cells[pop][-1] += 1
        for conn in rec.connections_on(gid):
            src = rec.gid_to_meta[conn.source.gid]["population"]
            conns[(src, pop)] += 1

    for (gid, _), _ in spikes:
        meta = rec.gid_to_meta[gid]
        pop = meta["population"]
        kind = meta["type_id"]
        spike[pop][kind] += 1
        spike[pop][-1] += 1
    C = sum(conns.values())
    timing.toc("output/statistics")

    print(
        f"""
Statistics
==========

* Cells                  {N:>13}"""
    )
    for pop, kinds in cells.items():
        print(f"  * {pop:<20} {kinds[-1]:>13}")
        for kind, num in kinds.items():
            if kind == -1:
                continue
            print(f"    * {kind:<18} {num:>13}")
    print(f"* Connections            {C:>13}")
    for (src, tgt), num in conns.items():
        print(f"  * {src:<8} -> {tgt:<8} {num:>13}")
    print(f"* Spikes                 {len(spikes):>13}")
    for pop, kinds in spike.items():
        print(f"  * {pop:<20} {kinds[-1]:>13}")
        for kind, num in kinds.items():
            if kind == -1:
                continue
            print(f"    * {kind:<18} {num:>13}")


timing.report()
