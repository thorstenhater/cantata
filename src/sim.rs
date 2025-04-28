use crate::{
    Map,
    err::{Context, Result},
    fit::Attribute,
    raw,
    sup::{Components, find_component},
};
use anyhow::{anyhow, bail};
use hdf5_metno as hdf5;
use serde::{Deserialize, Serialize};
use std::{fs::File, str::FromStr};

/// Model types currently known to SONATA
///
/// * single_compartment
///
///   A single cylindrical compartment is created with length equal to diameter,
///   thus the same effective area as that of a sphere of the same diameter. The
///   diameter is defined by an additional expected dynamics_param `D`, which
///   defaults to 1 um. Further, the passive mechanism is inserted and the
///   additional mechanisms named in the `model_template` required attribute.
///
/// * point_neuron
///
///   The actual model type is defined by the `model_template` required
///   attribute, eg an NMODL file for NRN and for NEST/PyNN, model_template will
///   provide the name of a built-in model.
///
/// * virtual
///
///   Placeholder neuron, which is not otherwise simulated, but can a the source
///   of spikes.
///
/// * biophysical
///
///   A compartmental neuron. The attribute morphology must be defined, either
///   via the node or node_type.
///
///  In addition, any number of Key:Value pairs may be listed
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "model_type")]
pub enum ModelType {
    #[serde(rename = "biophysical")]
    Biophysical {
        model_template: String,
        #[serde(flatten)]
        attributes: Map<String, Attribute>,
    },
    #[serde(rename = "single_compartment")]
    Single {
        model_template: String,
        #[serde(flatten)]
        attributes: Map<String, Attribute>,
    },
    #[serde(rename = "point_neuron")]
    Point {
        model_template: String,
        #[serde(flatten)]
        attributes: Map<String, Attribute>,
    },
    #[serde(rename = "virtual")]
    Virtual {
        #[serde(flatten)]
        attributes: Map<String, Attribute>,
    },
}

/// Types are defined in CSV files with one named column for each attribute.
/// Separator is a single space.
/// Non scalar attributes may be given provided values are quoted and their
/// components are separated by spaces.
/// Node type columns will be assigned to node attributes indexed by the type id.
/// Columns are:
/// - node_type_id: required; defines the node_type_id of this row.
/// - population: required in either this, or the nodes.h5; defines the population.
///               Multiple populations may define the same node_type_id,
/// - model_type: required, and may be defined only in the node_types.csv
/// - required columns may also appear in the instance H5; defined by the population
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NodeType {
    /// unique -- within population -- id of this type. Used in instances to
    /// reference this type.
    #[serde(rename = "node_type_id")]
    pub type_id: u64,
    /// population using this type. Might be given by instance file, this, or
    /// both
    #[serde(rename = "pop_name")]
    pub population: Option<String>,
    /// holds the actual parameter data
    #[serde(flatten)]
    pub model_type: ModelType,
    /// type-wide dynamics, pulled from the model-type
    #[serde(default)]
    pub dynamics: Map<String, f64>,
}

impl NodeType {
    fn clean_up(&mut self) {
        match &mut self.model_type {
            ModelType::Biophysical { attributes, .. }
            | ModelType::Single { attributes, .. }
            | ModelType::Point { attributes, .. }
            | ModelType::Virtual { attributes } => {
                attributes.retain(|_, v| !matches!(v, Attribute::String(s) if s == "NULL"));
            }
        }
    }

    pub fn attributes(&self) -> &Map<String, Attribute> {
        match &self.model_type {
            ModelType::Biophysical { attributes, .. }
            | ModelType::Single { attributes, .. }
            | ModelType::Point { attributes, .. }
            | ModelType::Virtual { attributes } => attributes,
        }
    }
}

#[derive(Debug)]
pub struct ParameterGroup {
    pub id: u64,
    pub dynamics: Map<String, Vec<f64>>,
    pub custom: Map<String, Vec<f64>>,
}

/// Populations are stored HDF5 files, and have an associated node types file to
/// define node_type_ids and assign attributes common across a population.
/// NOTE: node_types file may be shared by multiple population files.
///
/// Node groups are represented as HDF5 groups (with population as parent)
/// containing a dataset for each parameter of length equal to the number of
/// nodes in the group.
///
/// If an attribute is defined in both the node types and the node instances, the
/// latter overrides the former.
///
/// We have the following layout for the node instance HDF5 file:
///
/// Path                                 Type                      Required
/// =================================    ======================    ============
/// /nodes                               Group
///     * /<population_name>             Group
///         * /node_type_id              Dataset{N_total_nodes}    X
///         * /node_id                   Dataset{N_total_nodes}    X
///         * /node_group_id             Dataset{N_total_nodes}    X
///         * /node_group_index          Dataset{N_total_nodes}    X
///         * /<group_id>                Group                     one per unique group_id, at least one
///             * /dynamics_params       Group                     X (may be empty, though)
///                 * /<param>           Dataset {M_nodes}
///             * /<custom_attribute>    Dataset {M_nodes}
///
/// Notes:
/// * For each unique entry in node_group_id we expect one <group_id> group
///   under the population
#[derive(Debug)]
pub struct NodePopulation {
    pub name: String,
    pub size: usize,
    pub type_ids: Vec<u64>,
    pub node_ids: Vec<u64>,
    pub group_ids: Vec<u64>,
    pub group_indices: Vec<usize>,
    pub groups: Vec<ParameterGroup>,
}

#[derive(Debug)]
pub struct NodeList {
    pub types: Vec<NodeType>,
    pub populations: Vec<NodePopulation>,
}

impl NodeList {
    fn new(nodes: &raw::Nodes) -> Result<Self> {
        let path = &nodes.types;
        let path = std::path::PathBuf::from_str(&nodes.types)
            .map_err(anyhow::Error::from)
            .and_then(|p| p.canonicalize().map_err(anyhow::Error::from))
            .with_context(|| format!("Resolving node types {path}"))?;
        let rd = File::open(&path).with_context(|| format!("Opening {path:?}"))?;
        let mut tys = csv::ReaderBuilder::new()
            .delimiter(b' ')
            .from_reader(rd)
            .deserialize()
            .map(|it| it.map_err(anyhow::Error::from))
            .collect::<Result<Vec<NodeType>>>()
            .with_context(|| format!("Parsing node types {path:?}"))?;
        tys.iter_mut().for_each(|ty| ty.clean_up());
        let path = &nodes.nodes;
        let path = std::path::PathBuf::from_str(&nodes.nodes)
            .map_err(anyhow::Error::from)
            .and_then(|p| p.canonicalize().map_err(anyhow::Error::from))
            .with_context(|| format!("Resolving node instances {path}"))?;
        let node_instance_file = hdf5::file::File::open(&path)?;
        let nodes = node_instance_file.group("nodes")?;
        let populations = nodes
            .groups()
            .with_context(|| "Opening population list".to_string())?;

        let mut pops = Vec::new();
        for population in &populations {
            let name = population.name().rsplit_once('/').unwrap().1.to_string();
            let type_ids = get_dataset::<u64>(population, "node_type_id")?;
            let size = type_ids.len();
            let node_ids = get_dataset::<u64>(population, "node_id")
                .unwrap_or_else(|_| (0..size as u64).collect::<Vec<_>>());
            if size != node_ids.len() {
                anyhow::bail!(
                    "Population {name} in {path:?} has mismatched #type_ids ./. #node_ids"
                )
            }
            let group_ids = get_dataset::<u64>(population, "node_group_id")?;
            if size != group_ids.len() {
                anyhow::bail!(
                    "Population {name} in {path:?} has mismatched #type_ids ./. #group_id"
                )
            }
            let group_indices = get_dataset::<usize>(population, "node_group_index")?;
            if size != group_indices.len() {
                anyhow::bail!(
                    "Population {name} in {path:?} has mismatched #type_ids ./. #group_index"
                )
            }
            let mut groups = Vec::new();
            // NOTE We assume here that group ids are contiguous; yet, that is
            // said nowhere.
            let mut group_id = 0;
            loop {
                if let Ok(group) = population.group(&format!("{group_id}")) {
                    let mut dynamics = Map::new();
                    let mut custom = Map::new();
                    if let Ok(dynamics_params) = group.group("dynamics_params") {
                        for param in dynamics_params.datasets()?.iter() {
                            let values = param.read_1d::<f64>()?.to_vec();
                            let name = param.name().rsplit_once('/').unwrap().1.to_string();
                            dynamics.insert(name, values);
                        }
                    }
                    for param in group.datasets()?.iter() {
                        let values = param.read_1d::<f64>()?.to_vec();
                        let name = param.name().rsplit_once('/').unwrap().1.to_string();
                        custom.insert(name, values);
                    }

                    groups.push(ParameterGroup {
                        id: group_id as u64,
                        dynamics,
                        custom,
                    });
                } else {
                    break;
                }
                group_id += 1;
            }
            pops.push(NodePopulation {
                name,
                size,
                type_ids,
                node_ids,
                group_ids,
                group_indices,
                groups,
            })
        }
        Ok(Self {
            types: tys,
            populations: pops,
        })
    }
}

/// types are defined in a CSV file of named columns; separator is a single space.
/// - edge_type_id; required
/// - population; required either in CSV or H5; handles populations defining the same edge_type_id
/// - any number of additional columns may freely be added.
#[derive(Debug, Serialize, Deserialize)]
struct EdgeType {
    #[serde(rename = "edge_type_id")]
    pub type_id: u64,
    #[serde(rename = "pop_name")]
    pub population: Option<String>,
    #[serde(flatten)]
    pub attributes: Map<String, Attribute>,
    #[serde(default)]
    pub dynamics: Map<String, f64>,
}

impl EdgeType {
    fn clean_up(&mut self) {
        self.attributes
            .retain(|_, v| !matches!(v, Attribute::String(s) if s == "NULL"));
    }
}

/// Populations are stored HDF5 files, and have an associated edge types file to
/// define edge_type_ids and assign attributes common across a population.
/// NOTE: edge_types file may be shared by multiple population files.
///
/// Edge groups are represented as HDF5 groups (with population as parent)
/// containing a dataset for each parameter of length equal to the number of
/// edges in the group.
///
/// Edge populations are defined as groups and stored as sparse table via the
/// `source_node_id` and `target_node_id` datasets. These datasets have an
/// associated attribute "node_population" that specifies the node population
/// for resolving the node_ids of the source or target.
///
/// If an attribute is defined in both the edge types and the edge instances, the
/// latter overrides the former.
///
/// We have the following layout for the edge instance HDF5 file:
///
/// Path                                 Type                      Required
/// =================================    ======================    ============
/// /edges                               Group
///     * /<population_name>             Group
///         * /edge_type_id              Dataset{N_total_edges}    X
///         * /edge_id                   Dataset{N_total_edges}    X
///         * /edge_group_id             Dataset{N_total_edges}    X
///         * /edge_group_index          Dataset{N_total_edges}    X
///         * /source_node_id            Dataset{N_total_edges}    X
///             * /population_name       Attribute                 X
///         * /target_node_id            Dataset{N_total_edges}    X
///             * /population_name       Attribute                 X
///         * /<group_id>                Group
///             * /dynamics_params       Group
///                 * /<param>           Dataset {M_edges}
///             * /<custom_attribute>    Dataset {M_edges}
///
/// Notes:
/// * For each unique entry in edge_group_id we expect one <group_id> group
///   under the population
#[derive(Debug)]
struct EdgePopulation {
    name: String,
    size: usize,
    type_ids: Vec<u64>,
    group_ids: Vec<u64>,
    source_ids: Vec<u64>,
    source_pop: String,
    target_ids: Vec<u64>,
    target_pop: String,
    group_indices: Vec<usize>,
    groups: Vec<ParameterGroup>,
}

#[derive(Debug)]
pub struct EdgeList {
    types: Vec<EdgeType>,
    populations: Vec<EdgePopulation>,
}

fn get_dataset<T: hdf5::H5Type + Clone>(g: &hdf5::Group, nm: &str) -> Result<Vec<T>> {
    Ok(g.dataset(nm)
        .with_context(|| format!("Group {} has no dataset {nm}", g.name()))?
        .read_1d::<T>()?
        .to_vec())
}

impl EdgeList {
    fn new(edges: &raw::Edges) -> Result<Self> {
        let path = &edges.types;
        let path = std::path::PathBuf::from_str(&edges.types)
            .map_err(anyhow::Error::from)
            .and_then(|p| p.canonicalize().map_err(anyhow::Error::from))
            .with_context(|| format!("Resolving edge types {path}"))?;
        let rd = File::open(&path).with_context(|| format!("Opening {path:?}"))?;
        let mut tys = csv::ReaderBuilder::new()
            .delimiter(b' ')
            .from_reader(rd)
            .deserialize()
            .map(|it| it.map_err(anyhow::Error::from))
            .collect::<Result<Vec<EdgeType>>>()
            .with_context(|| format!("Parsing edge types {path:?}"))?;
        tys.iter_mut().for_each(|nt| nt.clean_up());
        let path = &edges.edges;
        let path = std::path::PathBuf::from_str(&edges.edges)
            .map_err(anyhow::Error::from)
            .and_then(|p| p.canonicalize().map_err(anyhow::Error::from))
            .with_context(|| format!("Resolving edge instances {path}"))?;
        let edge_instance_file = hdf5::file::File::open(&path)?;
        let edges = edge_instance_file.group("edges")?;
        let populations = edges
            .groups()
            .with_context(|| "Opening population list".to_string())?;

        let mut pops = Vec::new();
        for population in &populations {
            let name = population.name().rsplit_once('/').unwrap().1.to_string();
            let type_ids = get_dataset(population, "edge_type_id")?;
            let size = type_ids.len();
            let edge_ids = get_dataset::<u64>(population, "edge_id")
                .unwrap_or_else(|_| (0..size as u64).collect::<Vec<_>>());
            if size != edge_ids.len() {
                anyhow::bail!(
                    "Population {name} in {path:?} has mismatched #type_ids ./. #edge_ids"
                )
            }
            let group_ids = get_dataset::<u64>(population, "edge_group_id")?;
            if size != group_ids.len() {
                anyhow::bail!(
                    "Population {name} in {path:?} has mismatched #type_ids ./. #group_id"
                )
            }
            let group_indices = get_dataset::<usize>(population, "edge_group_index")?;
            if size != group_indices.len() {
                anyhow::bail!(
                    "Population {name} in {path:?} has mismatched #type_ids ./. #group_index"
                )
            }
            let sources = population
                .dataset("source_node_id")
                .with_context(|| format!("Extracting source indices from population {name}"))?;
            let source_ids = sources.read_1d::<u64>()?.to_vec();
            if size != source_ids.len() {
                anyhow::bail!(
                    "Population {name} in {path:?} has mismatched #type_ids ./. #source_ids"
                )
            }
            let source_pop = sources
                .attr("node_population")
                .with_context(|| {
                    format!("Extracting source population from population {name}; not found")
                })?
                .read_scalar::<hdf5::types::VarLenUnicode>()
                .with_context(|| {
                    format!("Extracting source population from population {name}; not a string")
                })?
                .as_str()
                .to_string();
            let targets = population
                .dataset("target_node_id")
                .with_context(|| format!("Extracting target indices from population {name}"))?;

            let target_ids = targets.read_1d::<u64>()?.to_vec();
            let target_pop = targets
                .attr("node_population")
                .with_context(|| {
                    format!("Extracting target population from population {name}; not found")
                })?
                .read_scalar::<hdf5::types::VarLenUnicode>()
                .with_context(|| {
                    format!("Extracting target population from population {name}; not a string")
                })?
                .as_str()
                .to_string();
            if size != target_ids.len() {
                anyhow::bail!(
                    "Population {name} in {path:?} has mismatched #type_ids ./. #target_ids"
                )
            }

            let group_indices = get_dataset::<usize>(population, "edge_group_index")?;
            if size != group_indices.len() {
                anyhow::bail!(
                    "Population {name} in {path:?} has mismatched #type_ids ./. #group_index"
                )
            }
            let mut groups = Vec::new();
            let mut group_id = 0;
            loop {
                if let Ok(group) = population.group(&format!("{group_id}")) {
                    let mut dynamics = Map::new();
                    let mut custom = Map::new();
                    if let Ok(dynamics_params) = group.group("dynamics_params") {
                        for param in dynamics_params.datasets()?.iter() {
                            let values = param.read_1d::<f64>()?.to_vec();
                            let name = param.name().rsplit_once('/').unwrap().1.to_string();
                            dynamics.insert(name, values);
                        }
                    }
                    for param in group.datasets()?.iter() {
                        let values = param.read_1d::<f64>()?.to_vec();
                        let name = param.name().rsplit_once('/').unwrap().1.to_string();
                        custom.insert(name, values);
                    }

                    groups.push(ParameterGroup {
                        id: group_id,
                        dynamics,
                        custom,
                    });
                } else {
                    break;
                }
                group_id += 1;
            }
            pops.push(EdgePopulation {
                name,
                size,
                type_ids,
                group_ids,
                source_ids,
                target_ids,
                source_pop,
                target_pop,
                groups,
                group_indices,
            })
        }
        Ok(Self {
            types: tys,
            populations: pops,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub src_gid: u64,
    pub source: (u64, f64),
    pub target: (f64, f64, f64),
    pub mech: Option<String>,
    pub delay: f64,
    pub weight: f64,
    pub dynamics: Map<String, f64>,
}

/// Reified node, containing all information we currently have
#[derive(Debug)]
pub struct Node {
    /// globally (!) unique id
    pub gid: usize,
    /// owning population
    pub pop: String,
    /// id within the population
    pub node_id: u64,
    /// id of containing group.
    pub group_id: u64,
    /// index within the group
    pub group_index: usize,
    /// id of node type in population used to instantiate.
    pub node_type_id: u64,
    /// node type used to instantiate
    pub node_type: NodeType,
    /// connections terminating here.
    pub incoming_edges: Vec<Edge>,
    /// Dynamics parameters extracted from the population.
    pub dynamics: Map<String, f64>,
    /// Custom parameters extracted from the population.
    pub custom: Map<String, f64>,
}

/// Bookeeping: index into top-level structure, ie population `pop` is stored at
/// node_lists[pop.list_index].populations[pop.pop_index]. The cells in this
/// population have identifiers in the range [start, start + size)
#[derive(Debug)]
pub struct PopId {
    /// Index into the containing {node, edge}_list
    pub list_index: usize,
    /// Index into the containing population list
    pub pop_index: usize,
    /// GID of first cell in this population
    pub start: usize,
    /// Number of cells in this population
    pub size: usize,
}

impl PopId {
    fn new_nodes(lid: usize, pid: usize, start: usize, pop: &NodePopulation) -> Result<Self> {
        Ok(PopId {
            list_index: lid,
            pop_index: pid,
            start,
            size: pop.size,
        })
    }

    fn new_edges(lid: usize, pid: usize, start: usize, pop: &EdgePopulation) -> Result<Self> {
        Ok(PopId {
            list_index: lid,
            pop_index: pid,
            start,
            size: pop.size,
        })
    }
}

#[derive(Debug)]
pub struct GlobalProperties {
    pub celsius: f64,
    pub v_init: f64,
}

#[derive(Debug)]
pub struct Simulation {
    /// runtime
    pub tfinal: f64,
    /// timestep
    pub dt: f64,
    /// raw-ish node data
    pub node_lists: Vec<NodeList>,
    /// raw-ish edge data
    pub edge_lists: Vec<EdgeList>,
    /// GID from (population, id)
    pub node_population_to_gid: Map<(usize, u64), u64>,
    /// GID to (population, id)
    pub gid_to_node_population: Vec<(usize, u64)>,
    /// node population list, to avoid copying strings and storing the indices
    /// into node_list and population.
    pub node_populations: Vec<PopId>,
    /// reverse mapping Name -> Id
    pub node_population_ids: Map<String, usize>,
    /// Number of total cells
    pub size: usize,
    /// reportable variables per population id
    pub reports: Map<u64, Vec<Probe>>,
    /// current clamps
    pub iclamps: Map<u64, Vec<IClamp>>,
    /// CV discretization
    pub cv_policy: CVPolicy,
    /// threshold for spike generation
    pub spike_threshold: f64,
    /// virtual cell spikes
    pub virtual_spikes: Map<String, Map<u64, Vec<f64>>>,
    /// cable cell global properties
    pub global_properties: Option<GlobalProperties>,
    /// components for delayed/lazy lookup
    pub components: Components,
}

#[derive(Debug, Clone)]
pub enum CVPolicy {
    Default,
    MaxExtent(f64),
}

#[derive(Debug, Clone)]
pub enum Probe {
    CableVoltage(Vec<String>),
    CableIntConc(String, Vec<String>),
    CableExtConc(String, Vec<String>),
    CableState(String, Vec<String>),
    Lif,
}

#[allow(non_snake_case)]
#[derive(Debug, Clone)]
pub struct IClamp {
    pub amplitude_nA: f64,
    pub delay_ms: f64,
    pub duration_ms: f64,
    pub tag: usize,
    pub location: String,
}

fn read_virtual_spikes(
    components: &Components,
    inputs: &raw::Inputs,
) -> Result<Map<String, Map<u64, Vec<f64>>>> {
    let mut res: Map<String, Map<u64, Vec<f64>>> = Map::new();
    for input in inputs.values() {
        if let raw::Input::Spikes {
            module,
            input_file,
            node_set,
        } = input
        {
            if module != "sonata" {
                bail!("Unknown module '{module}' for Input::Spikes");
            }
            if let raw::NodeSet::Name(node_set_name) = node_set {
                let data = res.entry(node_set_name.to_string()).or_default();
                for ifn in input_file {
                    let ifn = find_component(ifn, components)?;
                    let spikes = hdf5::file::File::open(&ifn)?
                        .group("spikes")?
                        .group(node_set_name)?;
                    let nodes = get_dataset::<u32>(&spikes, "node_ids")?;
                    let times = get_dataset::<f64>(&spikes, "timestamps")?;
                    if times.len() != nodes.len() {
                        bail!(
                            "Virtual spike data for {node_set_name} has different lengths on timestamps and node_ids"
                        );
                    }
                    for (id, ts) in nodes.iter().zip(times.iter()) {
                        data.entry(*id as u64).or_default().push(*ts);
                    }
                }
            } else {
                bail!("NodeSet on spikes must be a named reference, is {node_set:?}");
            }
        }
    }
    Ok(res)
}

impl Simulation {
    pub fn new(sim: &raw::Simulation) -> Result<Self> {
        let mut node_lists = sim
            .network
            .nodes
            .iter()
            .map(NodeList::new)
            .collect::<Result<Vec<_>>>()?;
        let mut edge_lists = sim
            .network
            .edges
            .iter()
            .map(EdgeList::new)
            .collect::<Result<Vec<_>>>()?;

        let mut start = 0;
        let mut edge_populations = Vec::new();
        for (lid, edge_list) in edge_lists.iter_mut().enumerate() {
            for ty in edge_list.types.iter_mut() {
                if let Some(Attribute::String(name)) = ty.attributes.get("dynamics_params") {
                    let fname = find_component(name, &sim.components)?;
                    let fdata = std::fs::File::open(fname)?;
                    let fdata: Map<String, serde_json::Value> = serde_json::from_reader(&fdata)
                        .with_context(|| format!("Parsing JSON from {name}"))?;
                    let mut param = fdata
                        .into_iter()
                        .filter_map(|(k, v)| v.as_f64().map(|v| (k.to_string(), v)))
                        .collect();
                    ty.dynamics.append(&mut param);
                }
            }
            for (pid, population) in edge_list.populations.iter().enumerate() {
                edge_populations.push(PopId::new_edges(lid, pid, start, population)?);
                start += population.size;
            }
        }
        let mut gid = 0;
        let mut start = 0;
        let mut node_population_to_gid = Map::new();
        let mut node_population_ids = Map::new();
        let mut gid_to_node_population = Vec::new();
        let mut node_populations = Vec::new();
        for (lid, node_list) in node_lists.iter_mut().enumerate() {
            for ty in node_list.types.iter_mut() {
                match &ty.model_type {
                    ModelType::Biophysical { attributes, .. }
                    | ModelType::Single { attributes, .. }
                    | ModelType::Point { attributes, .. }
                    | ModelType::Virtual { attributes, .. } => {
                        if let Some(Attribute::String(name)) = attributes.get("dynamics_params") {
                            let fname = find_component(name, &sim.components)?;
                            let fdata = std::fs::File::open(fname)?;
                            let fdata =
                                serde_json::from_reader::<_, Map<String, serde_json::Value>>(
                                    &fdata,
                                )
                                .with_context(|| format!("Parsing JSON from {name}"))?
                                .into_iter()
                                .filter_map(|(k, v)| v.as_f64().map(|v| (k.to_string(), v)));
                            ty.dynamics.extend(fdata);
                        }
                    }
                }
            }
            for (pid, population) in node_list.populations.iter().enumerate() {
                let pop_idx = node_populations.len();
                node_populations.push(PopId::new_nodes(lid, pid, start, population)?);
                node_population_ids.insert(population.name.to_string(), pop_idx);
                for nid in &population.node_ids {
                    node_population_to_gid.insert((pop_idx, *nid), gid);
                    gid_to_node_population.push((pop_idx, *nid));
                    gid += 1;
                }
                start += population.size;
            }
        }

        let cv_policy = if let Some(d) = sim.run.dl {
            CVPolicy::MaxExtent(d)
        } else {
            CVPolicy::Default
        };

        let virtual_spikes = read_virtual_spikes(&sim.components, &sim.inputs)?;

        let global_properties =
            if let raw::Conditions::Detailled { celsius, v_init } = sim.conditions {
                Some(GlobalProperties { celsius, v_init })
            } else {
                None
            };

        let mut res = Self {
            tfinal: sim.run.tstop,
            dt: sim.run.dt,
            node_lists,
            edge_lists,
            gid_to_node_population,
            node_population_to_gid,
            node_populations,
            node_population_ids,
            reports: Default::default(),
            iclamps: Default::default(),
            cv_policy,
            virtual_spikes,
            global_properties,
            components: sim.components.clone(),
            size: start,
            spike_threshold: sim.run.spike_threshold.expect("No spike threshold?"),
        };

        res.read_reports(&sim.reports)?;
        res.read_iclamps(&sim.inputs)?;

        Ok(res)
    }

    fn read_reports(&mut self, reports: &Map<String, raw::Report>) -> Result<()> {
        for raw::Report {
            cells,
            variable_name,
            sections,
            enabled,
            ..
        } in reports.values()
        {
            if !enabled {
                continue;
            }
            let gids = self.resolve_nodeset(cells)?;
            let probe = if !sections.is_empty() {
                let sections = sections.clone();
                if variable_name == "v" {
                    Probe::CableVoltage(sections)
                } else if variable_name.ends_with('i') {
                    eprintln!("Interpreting {variable_name} as internal ion concentration.");
                    Probe::CableIntConc(variable_name.clone(), sections)
                } else if variable_name.ends_with('o') {
                    eprintln!("Interpreting {variable_name} as external ion concentration.");
                    Probe::CableExtConc(variable_name.clone(), sections)
                } else {
                    Probe::CableState(variable_name.clone(), sections)
                }
            } else if variable_name == "v" {
                Probe::Lif
            } else {
                bail!("No clue how to probe {variable_name} on LIF cells.");
            };
            for gid in gids {
                self.reports.entry(gid).or_default().push(probe.clone());
            }
        }
        Ok(())
    }

    fn read_iclamps(&mut self, inputs: &Map<String, raw::Input>) -> Result<()> {
        let mut tag = 0;
        for input in inputs.values() {
            if let raw::Input::CurrentClamp {
                amp,
                delay,
                duration,
                node_set,
                enabled,
                ..
            } = input
            {
                if !enabled {
                    continue;
                }
                assert!(amp.len() == delay.len());
                assert!(amp.len() == duration.len());
                let set = node_set
                    .as_ref()
                    .ok_or_else(|| anyhow!("No nodeset given."))?;
                let gids = self.resolve_nodeset(set)?;
                // TODO find a better way here, there might be `electrode_file` present...
                let location = String::from("(location 0 0.5)");
                eprintln!("Using placeholder current clamp location {location} for {input:?}.");
                for gid in gids.into_iter() {
                    for ix in 0..amp.len() {
                        let ic = IClamp {
                            amplitude_nA: amp[ix],
                            delay_ms: delay[ix],
                            duration_ms: duration[ix],
                            tag,
                            location: location.clone(),
                        };
                        self.iclamps.entry(gid).or_default().push(ic);
                        tag += 1;
                    }
                }
            }
        }
        Ok(())
    }

    fn resolve_nodeset(&self, cells: &raw::NodeSet) -> Result<Vec<u64>> {
        let res = match cells {
            raw::NodeSet::Name(nm) => {
                let pid = self
                    .node_population_ids
                    .get(nm)
                    .ok_or(anyhow!("Unkown population <{nm}>"))?;
                let PopId { start, size, .. } = self.node_populations[*pid];
                (start..start + size).map(|i| i as u64).collect()
            }
            raw::NodeSet::Basic { population, rules } => {
                let mut res = Vec::new();
                let pop = population
                    .as_ref()
                    .ok_or_else(|| anyhow!("Basic NodeSet {cells:?} has no population."))?;
                let pid = self
                    .node_population_ids
                    .get(pop)
                    .ok_or_else(|| anyhow!("Basic NodeSet refers to unknown population {pop}."))?;
                let PopId { start, size, .. } = &self.node_populations[*pid];
                for gid in *start..*start + *size {
                    let Node {
                        dynamics, custom, ..
                    } = self.reify_node(gid)?;
                    let mut pred = true;
                    for (k, vs) in rules.iter() {
                        // TODO Can we match on anything else?
                        if let Some(u) = dynamics.get(k).or(custom.get(k)) {
                            match vs {
                                serde_json::Value::Number(v) => {
                                    let v = v.as_f64().ok_or_else(|| anyhow!("Basic NodeSet {cells:?} has non-float number for param {k}."))?;
                                    pred &= v == *u;
                                }
                                serde_json::Value::Array(vs) => {
                                    // Array filters are true if any value matches
                                    let mut found = false;
                                    for v in vs {
                                        let v = v.as_f64().ok_or_else(|| anyhow!("Basic NodeSet {cells:?} has non-float number for param {k}."))?;
                                        found |= v == *u;
                                    }
                                    pred &= found;
                                }
                                _ => bail!(
                                    "Cannot match on a {vs:?} in BasicNodeSet, must be list or value."
                                ),
                            };
                        } else {
                            // TODO What happens if we do not match the parameter name here?
                            eprintln!(
                                "Note: paramter {k} was not found in node {gid}. Currently we reject such nodes. If you consider this a bug, please report it as such."
                            )
                        }
                    }
                    if pred {
                        res.push(gid as u64);
                    }
                }
                res
            }
            raw::NodeSet::Ids(vs) => vs.clone(),
            raw::NodeSet::Compound(nss) => {
                let mut res = Vec::new();
                for ns in nss {
                    res.append(&mut self.resolve_nodeset(ns)?);
                }
                res
            }
        };
        Ok(res)
    }

    fn reify_edges(&self, target_population: &str, target_id: u64) -> Result<Vec<Edge>> {
        let mut incoming_edges = Vec::new();
        for edge_list in &self.edge_lists {
            for edge_population in edge_list
                .populations
                .iter()
                .filter(|p| p.target_pop == target_population)
            {
                for (edge_index, _) in edge_population
                    .target_ids
                    .iter()
                    .enumerate()
                    .filter(|it| *it.1 == target_id)
                {
                    let edge_index_error = || {
                        anyhow!(
                            "Index {} overruns size {} of population {}",
                            edge_index,
                            edge_population.size,
                            edge_population.name
                        )
                    };

                    let type_id = *edge_population
                        .type_ids
                        .get(edge_index)
                        .ok_or_else(edge_index_error)?;
                    let src_id = edge_population
                        .source_ids
                        .get(edge_index)
                        .ok_or_else(edge_index_error)?;
                    let src_pop = &edge_population.source_pop;
                    let src_idx = self
                        .node_population_ids
                        .get(src_pop)
                        .ok_or_else(|| anyhow!("Unknown population <{src_pop}>"))?;
                    let src_gid = *self
                        .node_population_to_gid
                        .get(&(*src_idx, *src_id))
                        .unwrap();
                    let edge_group_id = *edge_population
                        .group_ids
                        .get(edge_index)
                        .ok_or_else(edge_index_error)?;
                    let ty = edge_list
                        .types
                        .iter()
                        .find(|ty| ty.type_id == type_id)
                        .ok_or_else(|| {
                            anyhow!(
                                "Couldn't find edge type {type_id} in population {}",
                                edge_population.name
                            )
                        })?;
                    let edge_group = edge_population
                        .groups
                        .iter()
                        .find(|g| g.id == edge_group_id)
                        .ok_or_else(|| {
                            anyhow!(
                                "Couldn't find group id {edge_group_id} edge population {}",
                                edge_population.name
                            )
                        })?;
                    // index into group for this edge
                    let group_index = *edge_population
                        .group_indices
                        .get(edge_index)
                        .ok_or_else(edge_index_error)?;

                    let delay = if let Some(ds) = edge_group.custom.get("delay") {
                        ds[group_index]
                    } else if let Some(Attribute::Float(d)) = ty.attributes.get("delay") {
                        *d
                    } else {
                        bail!(
                            "Edge {type_id} in population {} has no delay",
                            edge_population.name
                        )
                    };
                    let weight = if let Some(ds) = edge_group.custom.get("syn_weight") {
                        ds[group_index]
                    } else if let Some(Attribute::Float(d)) = ty.attributes.get("syn_weight") {
                        *d
                    } else {
                        bail!(
                            "Edge {type_id} in population {} has no weight",
                            edge_population.name
                        )
                    };

                    let mech = if let Some(s) = ty.attributes.get("model_template") {
                        if let Attribute::String(s) = s {
                            Some(s.to_string())
                        } else {
                            bail!(
                                "Edge type {type_id} in population {} has non-string model",
                                edge_population.name
                            );
                        }
                    } else {
                        None
                    };

                    let tgt_pos_x = if let Some(ds) =
                        edge_group.custom.get("afferent_section_xcoords")
                    {
                        ds[group_index]
                    } else if let Some(s) = ty.attributes.get("afferent_section_xcoords") {
                        if let Attribute::Float(s) = s {
                            *s
                        } else {
                            bail!(
                                "Edge type {type_id} in population {} has non-numeric segment position",
                                edge_population.name
                            );
                        }
                    } else {
                        bail!(
                            "[UNSUPPORTED] Edge type {type_id} in population {} is missing the `x` coordinates for the target segments. (x,y,z) coordinates of traget segmemnts are required for synapse placement.",
                            edge_population.name
                        );
                    };

                    let tgt_pos_y = if let Some(ds) =
                        edge_group.custom.get("afferent_section_ycoords")
                    {
                        ds[group_index]
                    } else if let Some(s) = ty.attributes.get("afferent_section_ycoords") {
                        if let Attribute::Float(s) = s {
                            *s
                        } else {
                            bail!(
                                "Edge type {type_id} in population {} has non-numeric segment position",
                                edge_population.name
                            );
                        }
                    } else {
                        bail!(
                            "[UNSUPPORTED] Edge type {type_id} in population {} is missing the `y` coordinates for the target segments. (x,y,z) coordinates of traget segmemnts are required for synapse placement.",
                            edge_population.name
                        );
                    };

                    let tgt_pos_z = if let Some(ds) =
                        edge_group.custom.get("afferent_section_zcoords")
                    {
                        ds[group_index]
                    } else if let Some(s) = ty.attributes.get("afferent_section_zcoords") {
                        if let Attribute::Float(s) = s {
                            *s
                        } else {
                            bail!(
                                "Edge type {type_id} in population {} has non-numeric segment position",
                                edge_population.name
                            );
                        }
                    } else {
                        bail!(
                            "[UNSUPPORTED] Edge type {type_id} in population {} is missing the `z` coordinates for the target segments. (x,y,z) coordinates of traget segmemnts are required for synapse placement.",
                            edge_population.name
                        );
                    };

                    let src_pos = if let Some(ds) = edge_group.custom.get("efferent_swc_pos") {
                        ds[group_index]
                    } else if let Some(s) = ty.attributes.get("efferent_swc_pos") {
                        if let Attribute::Float(s) = s {
                            *s
                        } else {
                            bail!(
                                "Edge type {type_id} in population {} has non-numeric segment position",
                                edge_population.name
                            );
                        }
                    } else {
                        0.5 // default to centering
                    };

                    let src_id = if let Some(id) = edge_group.custom.get("efferent_swc_id") {
                        bail!(
                            "[UNSUPPORTED] Edge type {type_id} in population {} has efferent id {id:?}",
                            edge_population.name
                        );
                        // ds[group_index]
                    } else if let Some(id) = ty.attributes.get("efferent_swc_id") {
                        bail!(
                            "[UNSUPPORTED] Edge type {type_id} in population {} has efferent SWC id {id:?}",
                            edge_population.name
                        );
                        // if let Attribute::Float(s) = s {
                        // *s
                        // } else {
                        // bail!(
                        // "Edge type {type_id} in population {} has non-numeric segment position",
                        // edge_population.name
                        // );
                        // }
                    } else {
                        0.0 // default to first (=soma?)
                    } as u64;

                    // Construct dynamics parameters by merging the type level defaults with the group level overrides.
                    let mut dynamics = ty.dynamics.clone();
                    for (k, vs) in &edge_group.dynamics {
                        let v = vs[group_index];
                        dynamics.insert(k.to_string(), v);
                    }

                    incoming_edges.push(Edge {
                        src_gid,
                        source: (src_id, src_pos),
                        target: (tgt_pos_x, tgt_pos_y, tgt_pos_z),
                        mech,
                        delay,
                        weight,
                        dynamics,
                    });
                }
            }
        }
        Ok(incoming_edges)
    }

    pub fn reify_node(&self, gid: usize) -> Result<Node> {
        let (node_pop_idx, node_id) = self
            .gid_to_node_population
            .get(gid)
            .ok_or_else(|| anyhow!("Unknown gid {gid}, must be in [0, {})", self.size))?;
        let node_pop_id = self.node_populations.get(*node_pop_idx).expect("");
        let node_list = &self.node_lists[node_pop_id.list_index];
        let node_population = &node_list.populations[node_pop_id.pop_index];
        let node_index = gid - node_pop_id.start;

        // store pre-built error for later
        let node_index_error = || {
            anyhow!(
                "Index {} overruns size {} of population {}",
                node_index,
                node_pop_id.size,
                node_population.name
            )
        };
        let node_group_id = *node_population
            .group_ids
            .get(node_index)
            .ok_or_else(node_index_error)?;
        let node_type_id = *node_population
            .type_ids
            .get(node_index)
            .ok_or_else(node_index_error)?;
        let group_index = *node_population
            .group_indices
            .get(node_index)
            .ok_or_else(node_index_error)?;
        let node_type = node_list
            .types
            .iter()
            .find(|ty| ty.type_id == node_type_id)
            .ok_or_else(|| {
                anyhow!(
                    "Couldn't find node type {node_type_id} in population {}",
                    node_population.name
                )
            })?
            .clone();
        let group = node_population
            .groups
            .iter()
            .find(|g| g.id == node_group_id)
            .ok_or_else(|| {
                anyhow!(
                    "Couldn't find group id {node_group_id} node population {}",
                    node_population.name
                )
            })?;

        let mut dynamics = node_type.dynamics.clone();
        for (k, vs) in &group.dynamics {
            dynamics.insert(k.to_string(), vs[group_index]);
        }

        let mut custom = Map::new();
        for (k, v) in node_type.attributes() {
            if let Attribute::Float(v) = v {
                custom.insert(k.to_string(), *v);
            }
        }
        for (k, vs) in group.custom.iter() {
            custom.insert(k.to_string(), vs[group_index]);
        }

        let incoming_edges = self.reify_edges(&node_population.name, *node_id)?;

        Ok(Node {
            gid,
            pop: node_population.name.clone(),
            group_id: node_group_id,
            node_id: node_index as u64,
            group_index,
            node_type_id,
            node_type,
            incoming_edges,
            dynamics,
            custom,
        })
    }
}
