use std::{
    fs::File,
    path::{Path, PathBuf},
    str::FromStr,
};

use crate::{
    Map,
    err::{Context, Result, anyhow},
    sup::{Components, Manifest, resolve_manifest},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;

// Raw SONATA types
// Used to ingest simulation.json files.

/// default=true
fn yes() -> bool {
    true
}

/// default=false
fn no() -> bool {
    false
}

/// default=0.0
fn zero() -> f64 {
    0.0
}

/// To extract a list or a singleton value, i.e.
///   x: []        => Nil
///   y: foo       => One(foo)
///   z: [a, b, c] => Many([a, b, c])
#[derive(Debug, Deserialize, Serialize, Default)]
#[serde(untagged, deny_unknown_fields)]
enum MaybeVec<T> {
    #[default]
    Nil,
    One(T),
    Many(Vec<T>),
}

impl<T> From<MaybeVec<T>> for Vec<T> {
    fn from(value: MaybeVec<T>) -> Self {
        match value {
            MaybeVec::Nil => Vec::new(),
            MaybeVec::One(v) => vec![v],
            MaybeVec::Many(vs) => vs,
        }
    }
}

fn maybe_vec<'a, S: serde::Deserializer<'a>, T: Deserialize<'a>>(
    ser: S,
) -> Result<Vec<T>, S::Error> {
    let v = MaybeVec::<T>::deserialize(ser)?;
    Ok(v.into())
}

/// Simulation proceeds in blocks, either a count of steps, or a time interval
#[derive(Debug, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
pub enum TimeBlock {
    #[default]
    None,
    #[serde(rename = "nsteps_block")]
    Steps(u64),
    #[serde(rename = "tsteps_block")]
    Time(f64),
}

/// How to run the simulation / Timestepping
#[derive(Debug, Deserialize, Serialize)]
pub struct Run {
    /// Begin of simulation [ms]
    #[serde(default = "zero")]
    pub tstart: f64,
    /// End of simulation [ms]
    pub tstop: f64,
    /// Timestep [ms]
    pub dt: f64,
    /// maximum CV length [um]
    /// If not given, models or simulators choose
    /// May be overwritten downstream
    #[serde(rename = "dL")]
    pub dl: Option<f64>,
    /// Spiking threshold [mV]
    /// If not given, models or simulators choose
    /// May be overwritten downstream
    pub spike_threshold: Option<f64>,
    /// time blocking
    #[serde(default)]
    pub block: TimeBlock,
    /// Seed for PRNG
    pub random_seed: Option<u64>,
    // NOTE There's some ignored fields here:
    // * block_run
    // * block_size
    // * overwrite_output_dir
}

/// Physical conditions
#[derive(Debug, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields, untagged)]
pub enum Conditions {
    /// None given, just use whatever
    #[default]
    None,
    /// For NEURON/Arbor
    Detailled { celsius: f64, v_init: f64 },
    /// Unused
    LGN {
        jitter_lower: f64,
        jitter_upper: f64,
    },
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Waveform {
    shape: String,
    /// delay [ms]
    del: f64,
    /// amplitude [??]
    amp: f64,
    /// duration [ms]
    dur: f64,
    /// frequency
    freq: f64,
}

/// A node set specifies subsets of cells that act as targets for different
/// reports or stimulations, or can also be used to name and define the target
/// subpopulation to simulate.
///
/// Simple nodesets are defined using a dictionary of node attributes and
/// attribute values. An attribute value can be either a scalar (number, string,
/// bool) or an array of scalars.
///
///    "basic_nodeset": {
///        "<Property_Key1>": "<Prop_Val_11>"
///        "<Property_Key2>": ["<Prop_Val_21>", "<Prop_Val_22>", ...],
///    },
///
/// Each entry specifies a rule. For scalar attributes a node matches if the
/// value of its attribute matches the value in the entry. For arrays, a node
/// matches if its value matches any of the values in the array. A node is part
/// of a node set if it matches all the rules in the node set definition
/// (logical AND).
///
/// Compound node sets are declared as an array of node sets names, where each
/// name may refer to another compound node set or a basic node set. The final
/// node set is the union of all the node sets in the array.
///
///    "compound_node_set>": [<Basic_Node_Set_1>, <Compound_Node_Set_M>, ...],
///    ...
///
/// Two special attributes are allowed in the key-value pairs of basic node
/// sets. The first one is "population", this attribute refers to the node
/// populations to be considered. Node populations and their names are
/// implicitly defined in the Node Set namespace, and needn’t be declared
/// explicitly.
///
/// An Example of a Node Set File
///
///{
///    "bio_layer45": {
///        "model_type": "biophysical",
///        "location": ["layer4", "layer5"]
///    },
///    "V1_point_prime": {
///        "population": "biophysical",
///        "model_type": "point",
///        "node_id": [1, 2, 3, 5, 7, 9, ...]
///    }
///    "combined": ["bio_layer45", "V1_point_prime"]
///}
#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum NodeSet {
    Name(String),
    Basic {
        population: Option<String>,
        #[serde(flatten)]
        rules: Map<String, Value>,
    },
    Ids(Vec<u64>),
    Compound(Vec<NodeSet>),
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "input_type")]
pub enum Input {
    #[serde(rename = "current_clamp")]
    CurrentClamp {
        module: String,
        #[serde(deserialize_with = "maybe_vec")]
        amp: Vec<f64>,
        #[serde(deserialize_with = "maybe_vec")]
        delay: Vec<f64>,
        #[serde(deserialize_with = "maybe_vec")]
        duration: Vec<f64>,
        node_set: Option<NodeSet>,
        #[serde(default = "yes")]
        enabled: bool,
    },
    #[serde(rename = "spikes")]
    Spikes {
        module: String,
        #[serde(deserialize_with = "maybe_vec")]
        input_file: Vec<String>,
        node_set: NodeSet,
    },
    #[serde(rename = "lfp")]
    LFP {
        node_set: NodeSet,
        module: String,
        positions_file: String,
        mesh_files_dir: String,
        resistance: f64,
        waveform: Waveform,
    },
    #[serde(rename = "csv")]
    CSV {
        module: String,
        node_set: Option<NodeSet>,
        rates: Option<String>,
        file: Option<String>,
    },
    #[serde(rename = "nwb")]
    NWB {
        module: String,
        node_set: NodeSet,
        file: String,
        sweep_id: u64,
        downsample: f64,
    },
    #[serde(rename = "syn_activity")]
    SynActivity {
        module: String,
        precell_filter: Map<String, Value>,
        #[serde(deserialize_with = "maybe_vec")]
        timestamps: Vec<f64>,
        node_set: NodeSet,
    },
    #[serde(rename = "movie")]
    Movie { module: String },
}

impl Input {
    fn resolve_manifest(&mut self, manifest: &Manifest, base: &Path) -> Result<()> {
        match self {
            Input::Spikes { input_file, .. } => input_file
                .iter_mut()
                .try_for_each(|i| resolve_manifest(i, manifest, base)),
            Input::NWB { file, .. } => resolve_manifest(file, manifest, base),
            Input::LFP {
                positions_file,
                mesh_files_dir,
                ..
            } => {
                resolve_manifest(positions_file, manifest, base)?;
                resolve_manifest(mesh_files_dir, manifest, base)
            }
            Input::CSV {
                file: Some(file), ..
            } => resolve_manifest(file, manifest, base),
            _ => Ok(()),
        }
    }
}

pub type Inputs = Map<String, Input>;

/// Sort spikes by...
#[derive(Debug, Deserialize, Serialize, Default)]
pub enum SpikeSortOrder {
    /// ... unsorted, as from the source
    #[default]
    None,
    /// ... time of occurence
    #[serde(rename = "time")]
    Time,
    /// ... id of source
    #[serde(rename = "id")]
    Id,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Output {
    log_file: String,
    output_dir: String,
    spikes_file: Option<String>,
    spikes_file_csv: Option<String>,
    #[serde(default)]
    spikes_sort_order: SpikeSortOrder,
    #[serde(default = "no")]
    overwrite_output_dir: bool,
}

/// The set of nodes are represented by a vector of Nodes.
/// Each entry specifies a file for node types and node
/// instances where
#[derive(Debug, Deserialize, Serialize)]
pub struct Nodes {
    #[serde(rename = "nodes_file")]
    pub nodes: String,
    #[serde(rename = "node_types_file")]
    pub types: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Edges {
    #[serde(rename = "edges_file")]
    pub edges: String,
    #[serde(rename = "edge_types_file")]
    pub types: String,
}

impl Edges {
    fn resolve_manifest(&mut self, manifest: &Manifest, base: &Path) -> Result<()> {
        resolve_manifest(&mut self.edges, manifest, base)?;
        resolve_manifest(&mut self.types, manifest, base)
    }
}

impl Nodes {
    fn resolve_manifest(&mut self, manifest: &Manifest, base: &Path) -> Result<()> {
        resolve_manifest(&mut self.nodes, manifest, base)?;
        resolve_manifest(&mut self.types, manifest, base)
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Network {
    pub nodes: Vec<Nodes>,
    #[serde(default)]
    pub edges: Vec<Edges>,
}

impl Network {
    fn resolve_manifest(&mut self, manifest: &Manifest, base: &Path) -> Result<()> {
        self.nodes
            .iter_mut()
            .try_for_each(|n| n.resolve_manifest(manifest, base))?;
        self.edges
            .iter_mut()
            .try_for_each(|n| n.resolve_manifest(manifest, base))
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct NetworkFile {
    #[serde(default)]
    manifest: Manifest,
    #[serde(default)]
    components: Components,
    #[serde(alias = "networks")]
    pub network: Network,
}

#[derive(Debug, Deserialize, Serialize, Default)]
#[serde(untagged)]
pub enum NetworkOrFile {
    #[default]
    Empty,
    File(String),
    Inline(Network),
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Report {
    pub cells: NodeSet,
    pub variable_name: String,
    pub module: String,
    #[serde(default, deserialize_with = "maybe_vec")]
    pub sections: Vec<String>,
    pub start_time: Option<f64>,
    pub end_time: Option<f64>,
    pub dt: Option<f64>,
    pub unit: Option<String>,
    pub file_name: Option<String>,
    #[serde(default = "yes")]
    pub enabled: bool,
    // There's a metric ton of undocumented keys:
    // * electrode positions
    // * transform
    // * syn_types
    // ...
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SimulationRaw {
    #[serde(default)]
    manifest: Manifest,
    run: Run,
    target_simulator: String,
    #[serde(default)]
    conditions: Conditions,
    inputs: Inputs,
    output: Output,
    #[serde(alias = "networks", default)]
    network: NetworkOrFile,
    #[serde(default)]
    components: Components,
    #[serde(default)]
    reports: Map<String, Report>,
    #[serde(default)]
    node_sets: Map<String, NodeSet>,
    node_sets_file: Option<String>,
}

#[derive(Debug)]
pub struct Simulation {
    pub run: Run,
    pub target_simulator: String,
    pub conditions: Conditions,
    pub inputs: Inputs,
    pub output: Output,
    pub network: Network,
    pub components: Components,
    pub reports: Map<String, Report>,
    pub node_sets: Map<String, NodeSet>,
}

impl Simulation {
    pub fn from_file(path: &str) -> Result<Self> {
        let path = PathBuf::from_str(path)
            .map_err(anyhow::Error::from)
            .and_then(|p| p.canonicalize().map_err(anyhow::Error::from))
            .with_context(|| format!("Resolving simulation path {path:?}"))?;
        let base_dir = path
            .parent()
            .ok_or_else(|| anyhow!("Couldn't find parent of {path:?}."))?;
        let rd = File::open(&path).with_context(|| format!("Opening {path:?}"))?;

        // Read JSON
        let mut raw: SimulationRaw = serde_json::de::from_reader(rd)
            .with_context(|| format!("Parsing simulation {path:?}"))?;

        raw.manifest.insert(
            "$configdir".to_string(),
            base_dir
                .to_str()
                .expect("Couldn't convert path to str")
                .into(),
        );
        raw.components
            .iter_mut()
            .try_for_each(|(_, it)| resolve_manifest(it, &raw.manifest, base_dir))?;
        raw.components
            .insert("base_dir".into(), base_dir.to_str().unwrap().into());

        // Resolve the Network to an object
        let mut network = match raw.network {
            NetworkOrFile::Empty => return Err(anyhow!("No network defined!")),
            NetworkOrFile::File(file) => {
                let mut path: String = base_dir.join(file).to_str().unwrap().into();
                resolve_manifest(&mut path, &raw.manifest, base_dir)?;
                let rd = File::open(&path).with_context(|| format!("Opening {path:?}"))?;
                let mut net: NetworkFile = serde_json::de::from_reader(rd)
                    .with_context(|| format!("Parsing network {path:?}"))?;
                net.manifest.insert(
                    "$configdir".to_string(),
                    base_dir
                        .to_str()
                        .expect("Couldn't convert path to str")
                        .into(),
                );
                net.components
                    .iter_mut()
                    .try_for_each(|(_, it)| resolve_manifest(it, &net.manifest, base_dir))?;
                raw.components.append(&mut net.components);
                net.network.resolve_manifest(&net.manifest, base_dir)?;
                net.network
            }
            NetworkOrFile::Inline(net) => net,
        };
        network.resolve_manifest(&raw.manifest, base_dir)?;

        // Resolve the NodeSets
        let node_sets = {
            if let Some(file) = raw.node_sets_file {
                let mut path: String = base_dir.join(file).to_str().unwrap().into();
                resolve_manifest(&mut path, &raw.manifest, base_dir)?;
                let rd = File::open(&path).with_context(|| format!("Opening {path:?}"))?;
                let nds: Map<String, NodeSet> = serde_json::de::from_reader(rd)
                    .with_context(|| format!("Parsing nodesets {path:?}"))?;
                raw.node_sets.extend(nds);
            }
            raw.node_sets
        };

        raw.inputs
            .values_mut()
            .try_for_each(|i| i.resolve_manifest(&raw.manifest, base_dir))?;

        Ok(Simulation {
            run: raw.run,
            target_simulator: raw.target_simulator,
            conditions: raw.conditions,
            inputs: raw.inputs,
            output: raw.output,
            network,
            components: raw.components,
            reports: raw.reports,
            node_sets,
        })
    }
}
