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
use tracing::{event, span, Level};
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
        attributes: Map<Box<str>, Attribute>,
    },
    #[serde(rename = "single_compartment")]
    Single {
        model_template: String,
        #[serde(flatten)]
        attributes: Map<Box<str>, Attribute>,
    },
    #[serde(rename = "point_neuron")]
    Point {
        model_template: String,
        #[serde(flatten)]
        attributes: Map<Box<str>, Attribute>,
    },
    #[serde(rename = "virtual")]
    Virtual {
        #[serde(flatten)]
        attributes: Map<Box<str>, Attribute>,
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
    pub type_id: u32,
    /// population using this type. Might be given by instance file, this, or
    /// both
    #[serde(rename = "pop_name")]
    pub population: Option<String>,
    /// holds the actual parameter data
    #[serde(flatten)]
    pub model_type: ModelType,
    /// type-wide dynamics, pulled from the model-type
    #[serde(default)]
    pub dynamics: Map<Box<str>, f64>,
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

    pub fn attributes(&self) -> &Map<Box<str>, Attribute> {
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
    pub id: u32,
    pub dynamics: Map<Box<str>, Vec<f64>>,
    pub custom: Map<Box<str>, Vec<f64>>,
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
    pub name: Box<str>,
    pub size: usize,
    pub type_ids: Vec<u32>,
    pub node_ids: Vec<u32>,
    pub group_ids: Vec<u32>,
    pub group_indices: Vec<u32>,
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
        let rd = std::io::BufReader::with_capacity(
            1 * 1024 * 1024,
            File::open(&path).with_context(|| format!("Opening {path:?}"))?,
        );
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
            let type_ids = get_dataset::<u32>(population, "node_type_id")?;
            let size = type_ids.len();
            let node_ids = get_dataset::<u32>(population, "node_id")
                .unwrap_or_else(|_| (0..size as u32).collect::<Vec<_>>());
            if size != node_ids.len() {
                anyhow::bail!(
                    "Population {name} in {path:?} has mismatched #type_ids ./. #node_ids"
                )
            }
            let group_ids = get_dataset::<u32>(population, "node_group_id")?;
            if size != group_ids.len() {
                anyhow::bail!(
                    "Population {name} in {path:?} has mismatched #type_ids ./. #group_id"
                )
            }
            let group_indices = get_dataset::<u32>(population, "node_group_index")?;
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
                            let name: Box<str> = param.name().rsplit_once('/').unwrap().1.into();
                            dynamics.insert(name, values);
                        }
                    }
                    for param in group.datasets()?.iter() {
                        let values = param.read_1d::<f64>()?.to_vec();
                        let name = param.name().rsplit_once('/').unwrap().1.into();
                        custom.insert(name, values);
                    }

                    groups.push(ParameterGroup {
                        id: group_id as u32,
                        dynamics,
                        custom,
                    });
                } else {
                    break;
                }
                group_id += 1;
            }
            pops.push(NodePopulation {
                name: name.into(),
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
#[derive(Debug, Serialize, Deserialize, Clone)]
struct EdgeType {
    #[serde(rename = "edge_type_id")]
    pub type_id: u32,
    #[serde(rename = "pop_name")]
    pub population: Option<String>,
    #[serde(flatten)]
    pub attributes: Map<Box<str>, Attribute>,
    #[serde(default)]
    pub dynamics: Map<Box<str>, f64>,
}

impl EdgeType {
    fn clean_up_mut(&mut self) {
        self.attributes
            .retain(|_, v| !matches!(v, Attribute::String(s) if s == "NULL"));
    }
    fn clean_up(mut self) -> Self {
        self.clean_up_mut();
        self
    }
}

/// Parameters of an EdgeGroup
#[derive(Debug)]
struct EdgeGroup {
    pub id: u32,
    // afferent position in 3d
    pub x: Option<Vec<f64>>,
    pub y: Option<Vec<f64>>,
    pub z: Option<Vec<f64>>,
    // synaptic weight
    pub weight: Option<Vec<f32>>,
    pub delay: Option<Vec<f32>>,
    // custom dynamics
    pub dynamics: Map<Box<str>, Vec<f64>>,
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
    type_ids: Vec<u32>,
    group_ids: Vec<u32>,
    source_ids: Vec<u32>,
    source_pop: String,
    target_ids: Vec<u32>,
    target_pop: String,
    group_indices: Vec<u32>,
    groups: Vec<EdgeGroup>,
}

impl EdgePopulation {
    fn new(population: &hdf5::Group) -> Result<Self> {
        let name = population.name().rsplit_once('/').unwrap().1.to_string();
        span!(Level::INFO, "edge population {name}");
        let type_ids = get_dataset(population, "edge_type_id")?;
        let size = type_ids.len();
        let group_ids = get_dataset::<u32>(population, "edge_group_id")?;
        if size != group_ids.len() {
            anyhow::bail!("Population {name} has mismatched #type_ids ./. #group_id")
        }
        let group_indices = get_dataset::<u32>(population, "edge_group_index")?;
        if size != group_indices.len() {
            anyhow::bail!("Population {name} has mismatched #type_ids ./. #group_index")
        }
        let sources = population
            .dataset("source_node_id")
            .with_context(|| format!("Extracting source indices from population {name}"))?;
        let source_ids = sources.read_1d::<u32>()?.to_vec();
        if size != source_ids.len() {
            anyhow::bail!("Population {name} has mismatched #type_ids ./. #source_ids")
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

        let target_ids = targets.read_1d::<u32>()?.to_vec();
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
            anyhow::bail!("Population {name} has mismatched #type_ids ./. #target_ids")
        }
        event!(name: "Edges", Level::INFO, "{name} #edges={size}");
        let mut group_id = 0;
        let mut groups = Vec::new();
        while let Ok(group) = population.group(&format!("{group_id}")) {
            event!(name: "EdgeGroup", Level::INFO, "group={group_id}");
            let mut dynamics = Map::new();
            if let Ok(dynamics_params) = group.group("dynamics_params") {
                for param in dynamics_params.datasets()?.iter() {
                    println!("   * name={} size={} ", param.name(), param.size());
                    let values = param.read_1d::<f64>()?.to_vec();
                    let name: Box<str> = param.name().rsplit_once('/').unwrap().1.into();
                    dynamics.insert(name, values);
                }
            }
            let x = group.dataset("afferent_section_xcoords").ok().map(|ds| {
                ds.read_1d::<f64>()
                    .expect(&format!("Extracting group position from id {group_id}"))
                    .to_vec()
            });

            let y = group.dataset("afferent_section_ycoords").ok().map(|ds| {
                ds.read_1d::<f64>()
                    .expect(&format!("Extracting group position from id {group_id}"))
                    .to_vec()
            });

            let z = group.dataset("afferent_section_zcoords").ok().map(|ds| {
                ds.read_1d::<f64>()
                    .expect(&format!("Extracting group position from id {group_id}"))
                    .to_vec()
            });

            let weight = group.dataset("syn_weight").ok().map(|ds| {
                ds.read_1d::<f32>()
                    .expect(&format!("Extracting group position from id {group_id}"))
                    .to_vec()
            });

            let delay = group.dataset("delay").ok().map(|ds| {
                ds.read_1d::<f32>()
                    .expect(&format!("Extracting group position from id {group_id}"))
                    .to_vec()
            });

            groups.push(EdgeGroup {
                id: group_id,
                x,
                y,
                z,
                delay,
                weight,
                dynamics,
            });
            group_id += 1;
        }

        Ok(Self {
            name,
            size,
            type_ids,
            group_ids,
            source_ids,
            source_pop,
            target_ids,
            target_pop,
            group_indices,
            groups,
        })
    }
}

#[derive(Debug, Default)]
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
        let rd = std::io::BufReader::with_capacity(
            1 * 1024 * 1024,
            File::open(&path).with_context(|| format!("Opening {path:?}"))?,
        );
        let types = csv::ReaderBuilder::new()
            .delimiter(b' ')
            .from_reader(rd)
            .deserialize()
            .map(|it| it.map(EdgeType::clean_up).map_err(anyhow::Error::from))
            .collect::<Result<Vec<EdgeType>>>()
            .with_context(|| format!("Parsing edge types {path:?}"))?;

        let path = &edges.edges;
        let path = std::path::PathBuf::from_str(&edges.edges)
            .map_err(anyhow::Error::from)
            .and_then(|p| p.canonicalize().map_err(anyhow::Error::from))
            .with_context(|| format!("Resolving edge instances {path}"))?;
        let edge_instance_file = hdf5::file::File::open(&path)?;
        let edges = edge_instance_file.group("edges")?;
        let populations = edges
            .groups()
            .with_context(|| "Opening population list".to_string())?
            .iter()
            .map(EdgePopulation::new)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { types, populations })
    }
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub src_gid: u32,
    pub source: (u32, f64),
    pub target: (f64, f64, f64),
    pub mech: Option<String>,
    pub delay: f32,
    pub weight: f32,
    pub dynamics: Map<Box<str>, f64>,
}

/// Reified node, containing all information we currently have
#[derive(Debug)]
pub struct Node {
    /// globally (!) unique id
    pub gid: usize,
    /// owning population
    pub pop: Box<str>,
    /// id within the population
    pub node_id: u32,
    /// id of containing group.
    pub group_id: u32,
    /// index within the group
    pub group_index: u32,
    /// id of node type in population used to instantiate.
    pub node_type_id: u32,
    /// node type used to instantiate
    pub node_type: NodeType,
    /// Dynamics parameters extracted from the population.
    pub dynamics: Map<Box<str>, f64>,
    /// Custom parameters extracted from the population.
    pub custom: Map<Box<str>, f64>,
    /// Node position coordinates (x, y, z)
    pub position: (f64, f64, f64),
    /// Node rotation angles (rotation_angle_xaxis, rotation_angle_yaxis, rotation_angle_zaxis)
    pub rotation: (f64, f64, f64),
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
    pub node_population_to_gid: Map<(usize, u32), u32>,
    /// GID to (population, id)
    pub gid_to_node_population: Vec<(usize, u32)>,
    /// node population list, to avoid copying strings and storing the indices
    /// into node_list and population.
    pub node_populations: Vec<PopId>,
    /// reverse mapping Name -> Id
    pub node_population_ids: Map<Box<str>, usize>,
    /// Number of total cells
    pub size: usize,
    /// reportable variables per population id
    pub reports: Map<u32, Vec<Probe>>,
    /// current clamps
    pub iclamps: Map<u32, Vec<IClamp>>,
    /// CV discretization
    pub cv_policy: CVPolicy,
    /// threshold for spike generation
    pub spike_threshold: f64,
    /// virtual cell spikes
    pub virtual_spikes: Map<String, Map<u32, Vec<f64>>>,
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
) -> Result<Map<String, Map<u32, Vec<f64>>>> {
    let mut res: Map<String, Map<u32, Vec<f64>>> = Map::new();
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
                        data.entry(*id as u32).or_default().push(*ts);
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
        span!(Level::INFO, "Building simulation from raw");
        let mut edge_lists = Vec::new();
        let mut edge_populations = Vec::new();
        let mut start = 0;
        for (lid, edges) in sim.network.edges.iter().enumerate() {
            let mut edge_list = EdgeList::new(edges)?;
            for ty in edge_list.types.iter_mut() {
                if let Some(Attribute::String(name)) = ty.attributes.get("dynamics_params") {
                    let fname = find_component(name, &sim.components)?;
                    let fdata = std::io::BufReader::with_capacity(
                        1 * 1024 * 1024,
                        std::fs::File::open(fname)?,
                    );
                    let fdata: Map<Box<str>, serde_json::Value> = serde_json::from_reader(fdata)
                        .with_context(|| format!("Parsing JSON from {name}"))?;
                    let mut param = fdata
                        .into_iter()
                        .filter_map(|(k, v)| v.as_f64().map(|v| (k.clone(), v)))
                        .collect();
                    ty.dynamics.append(&mut param);
                }
            }
            for (pid, population) in edge_list.populations.iter().enumerate() {
                edge_populations.push(PopId::new_edges(lid, pid, start, population)?);
                start += population.size;
            }
            edge_lists.push(edge_list);
        }

        let mut gid = 0;
        let mut start = 0;
        let mut node_population_to_gid = Map::new();
        let mut node_population_ids = Map::new();
        let mut gid_to_node_population = Vec::new();
        let mut node_populations = Vec::new();
        let mut node_lists = sim
            .network
            .nodes
            .iter()
            .map(NodeList::new)
            .collect::<Result<Vec<_>>>()?;
        for (lid, node_list) in node_lists.iter_mut().enumerate() {
            for ty in node_list.types.iter_mut() {
                match &ty.model_type {
                    ModelType::Biophysical { attributes, .. }
                    | ModelType::Single { attributes, .. }
                    | ModelType::Point { attributes, .. }
                    | ModelType::Virtual { attributes, .. } => {
                        if let Some(Attribute::String(name)) = attributes.get("dynamics_params") {
                            let fname = find_component(name, &sim.components)?;
                            let fdata = std::io::BufReader::with_capacity(
                                1 * 1024 * 1024,
                                std::fs::File::open(fname)?,
                            );
                            let fdata =
                                serde_json::from_reader::<_, Map<String, serde_json::Value>>(fdata)
                                    .with_context(|| format!("Parsing JSON from {name}"))?
                                    .into_iter()
                                    .filter_map(|(k, v)| {
                                        v.as_f64().map(|v| (k.into_boxed_str(), v))
                                    });
                            ty.dynamics.extend(fdata);
                        }
                    }
                }
            }
            for (pid, population) in node_list.populations.iter().enumerate() {
                let pop_idx = node_populations.len();
                node_populations.push(PopId::new_nodes(lid, pid, start, population)?);
                node_population_ids.insert(population.name.to_string().into_boxed_str(), pop_idx);
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

        event!(name: "Raw2SimDone", Level::INFO, "Instantiated simulation with {start} nodes.");
        
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
            size: std::cmp::min(start, 2000),
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

    fn resolve_nodeset(&self, cells: &raw::NodeSet) -> Result<Vec<u32>> {
        let res = match cells {
            raw::NodeSet::Name(nm) => {
                let pid = self
                    .node_population_ids
                    .get(nm.as_str())
                    .ok_or(anyhow!("Unkown population <{nm}>"))?;
                let PopId { start, size, .. } = self.node_populations[*pid];
                (start..start + size).map(|i| i as u32).collect()
            }
            raw::NodeSet::Basic { population, rules } => {
                let mut res = Vec::new();
                let pop = population
                    .as_ref()
                    .ok_or_else(|| anyhow!("Basic NodeSet {cells:?} has no population."))?;
                let pid = self
                    .node_population_ids
                    .get(pop.as_str())
                    .ok_or_else(|| anyhow!("Basic NodeSet refers to unknown population {pop}."))?;
                let PopId { start, size, .. } = &self.node_populations[*pid];
                for gid in *start..*start + *size {
                    let Node {
                        dynamics, custom, ..
                    } = self.reify_node(gid)?;
                    let mut pred = true;
                    for (k, vs) in rules.iter() {
                        // TODO Can we match on anything else?
                        if let Some(u) = dynamics.get(k.as_str()).or(custom.get(k.as_str())) {
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
                        res.push(gid as u32);
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

    pub fn reify_edges(&self, gid: usize) -> Result<Vec<Edge>> {
        let (node_pop_idx, node_id) = self
            .gid_to_node_population
            .get(gid)
            .ok_or_else(|| anyhow!("Unknown gid {gid}, must be in [0, {})", self.size))?;
        let node_pop_id = self
            .node_populations
            .get(*node_pop_idx)
            .expect("Unknown population");
        let node_list = &self.node_lists[node_pop_id.list_index];
        let node_population = &node_list.populations[node_pop_id.pop_index];
        let target_population = &node_population.name;

        let mut incoming_edges = Vec::new();
        for edge_list in &self.edge_lists {
            for edge_population in edge_list
                .populations
                .iter()
                .filter(|p| p.target_pop.as_str() == target_population.as_ref())
            {
                for (edge_index, _) in edge_population
                    .target_ids
                    .iter()
                    .enumerate()
                    .filter(|it| *it.1 == *node_id)
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
                        .get(src_pop.as_str())
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

                    let delay = if let Some(ds) = edge_group.delay.as_ref() {
                        ds[group_index as usize]
                    } else if let Some(Attribute::Float(d)) = ty.attributes.get("delay") {
                        *d as f32
                    } else {
                        bail!(
                            "Edge {type_id} in population {} has no delay",
                            edge_population.name
                        )
                    };

                    let x = if let Some(ds) = edge_group.x.as_ref() {
                        ds[group_index as usize]
                    } else if let Some(Attribute::Float(d)) =
                        ty.attributes.get("afferent_section_xcoords")
                    {
                        *d
                    } else {
                        bail!(
                            "Edge {type_id} in population {} has no x",
                            edge_population.name
                        )
                    };

                    let y = if let Some(ds) = edge_group.y.as_ref() {
                        ds[group_index as usize]
                    } else if let Some(Attribute::Float(d)) =
                        ty.attributes.get("afferent_section_ycoords")
                    {
                        *d
                    } else {
                        bail!(
                            "Edge {type_id} in population {} has no y",
                            edge_population.name
                        )
                    };

                    let z = if let Some(ds) = edge_group.z.as_ref() {
                        ds[group_index as usize]
                    } else if let Some(Attribute::Float(d)) =
                        ty.attributes.get("afferent_section_zcoords")
                    {
                        *d
                    } else {
                        bail!(
                            "Edge {type_id} in population {} has no z",
                            edge_population.name
                        )
                    };

                    let weight = if let Some(ds) = edge_group.weight.as_ref() {
                        ds[group_index as usize]
                    } else if let Some(Attribute::Float(d)) = ty.attributes.get("syn_weight") {
                        *d as f32
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

                    // NOTE: These are fixed for now
                    let src_pos = 0.5;
                    let src_id = 0u32;

                    // Construct dynamics parameters by merging the type level defaults with the group level overrides.
                    let mut dynamics = ty.dynamics.clone();
                    for (k, vs) in &edge_group.dynamics {
                        let v = vs[group_index as usize];
                        dynamics.insert(k.clone(), v);
                    }

                    incoming_edges.push(Edge {
                        src_gid,
                        source: (src_id, src_pos),
                        target: (x, y, z),
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
        let (node_pop_idx, _node_id) = self
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

        let mut dynamics: Map<Box<str>, _> = node_type.dynamics.clone();
        for (k, vs) in &group.dynamics {
            dynamics.insert(k.clone(), vs[group_index as usize]);
        }

        let mut custom: Map<Box<str>, _> = Map::new();
        for (k, v) in node_type.attributes() {
            if let Attribute::Float(v) = v {
                custom.insert(k.clone().into(), *v);
            }
        }
        for (k, vs) in group.custom.iter() {
            custom.insert(k.clone(), vs[group_index as usize]);
        }

        // Extract position coordinates (x, y, z)
        let get_cell_attributes = |c: &str| -> f64 {
            if let Some(cs) = group.custom.get(c) {
                cs[group_index as usize]
            } else if let Some(Attribute::Float(val)) = node_type.attributes().get(c) {
                *val
            } else {
                0.0 // default value
            }
        };
        let pos_x = get_cell_attributes("x");
        let pos_y = get_cell_attributes("y");
        let pos_z = get_cell_attributes("z");

        // Extract rotation angles
        let rot_x = get_cell_attributes("rotation_angle_xaxis");
        let rot_y = get_cell_attributes("rotation_angle_yaxis");
        let rot_z = get_cell_attributes("rotation_angle_zaxis");

        Ok(Node {
            gid,
            pop: node_population.name.clone(),
            group_id: node_group_id,
            node_id: node_index as u32,
            group_index,
            node_type_id,
            node_type,
            dynamics,
            custom,
            position: (pos_x, pos_y, pos_z),
            rotation: (rot_x, rot_y, rot_z),
        })
    }
}
