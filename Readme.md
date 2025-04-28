# Cantata

Translate SONATA simulations into Arbor simulation bundles. 

## Dependencies

We require a recent version of Rust (1.80), Python (3.10 or later), and HDF5
(development, i.e. headers and libraries). Acquiring some examples in SONATA
will be helpful; you can do so here

https://github.com/AllenInstitute/sonata

or here

https://github.com/AllenInstitute/bmtk

:warning: Cantata requires that synapse positions are included in the SONATA
files, since the native storage format relies on unportable Neuron internals.
The relevant HDF5 dataset names are ``afferent_section_xcoords``,
``afferent_section_ycoords``, and ``afferent_section_zcoords``, with the obivous
meaning.


## Setup (stable)

``` sh
cargo install cantata
```
Run a test
``` sh
cantata build path/to/sonata/simulation.json out-dir
cd out-dir
python3 main.py
```

This will perform the translation and create a working simulation in the output
directory `out-dir`. Running the simulation will generate the requested outputs
(spikes and traces) in `out-dir/out`. You may want/need to tweak the simulation.

There is a convenience wrapper that does all of the above and will construct and
    execute the simulation in `simulation.sim`.

```sh
cantata run path/to/sonata/simulation.json
```

## Setup (dev)

``` sh
git clone https://github.com/thorstenhater/cantata.git
```
Run a test
``` sh
cd cantata
cargo run -- build path/to/sonata/simulation.json out-dir
cd out-dir
python3 main.py
```
This will perform the translation and create a working simulation in the output
directory `out-dir`. Running the simulation will generate the requested outputs
(spikes and traces) in `out-dir/out`. You may want/need to tweak the simulation.
