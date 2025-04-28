use crate::{
    Map,
    err::Result,
    fit::Attribute,
    sim::{CVPolicy, GlobalProperties, IClamp, ModelType, Node, Probe, Simulation},
};
use anyhow::{anyhow, bail};
use serde::Serialize;

// Resources to store in the output.

/// (src_gid, src_tag, tgt_tag, weight, delay)
type ConnectionData = (usize, usize, usize, f64, f64);
/// (location, variable, tag)
type ProbeData = (Option<String>, String, usize);
/// (x, y, z, mech, params, tag)
type SynapseData = (f64, f64, f64, String, Map<String, f64>, usize);
/// (location, delay, duration, current, tag)
type IClampData = (String, f64, f64, f64, usize);

/// Metadata about cell,
/// mostly info discarded during generation
#[derive(Debug, Serialize)]
pub struct CellMetaData {
    /// cell kind
    pub kind: String,
    /// population name
    pub population: String,
    /// cell type label
    pub type_id: u64,
}

impl CellMetaData {
    pub fn from(node: &Node) -> Self {
        let kind = match node.node_type.model_type {
            ModelType::Biophysical { .. } => String::from("biophys"),
            ModelType::Single { .. } => String::from("single"),
            ModelType::Point { .. } => String::from("point"),
            ModelType::Virtual { .. } => String::from("virtual"),
        };
        Self {
            population: node.pop.to_string(),
            kind,
            type_id: node.node_type.type_id,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct CableGlobalProperties {
    pub celsius: f64,
    pub v_init: f64,
}

#[derive(Debug, Serialize)]
pub struct Bundle {
    pub time: f64,
    pub time_step: f64,
    pub size: usize,
    pub max_cv_length: Option<f64>,
    /// gid to morphology and acc ids
    /// this works as an index into the next two fields.
    pub cell_bio_ids: Map<usize, (usize, usize)>,
    /// acc index to name
    pub morphology: Vec<String>,
    /// morphology index to name
    pub decoration: Vec<String>,
    /// cell kinds, 0 = cable, 1 = lif, 2 = spike source, ...
    pub cell_kind: Vec<u64>,
    /// synapse data, cross-linked with incoming connections.
    /// Location, Synapse, Parameters, Tag
    /// May only be set iff kind==cable
    pub synapses: Map<usize, Vec<SynapseData>>,
    /// stimuli; May only be set iff kind==cable
    /// location, delay, duration, amplitude, tag
    pub current_clamps: Map<usize, Vec<IClampData>>,
    /// List of data exporters
    /// location, variable, tag. NOTE _could_ make variable an u64?!
    pub probes: Map<usize, Vec<ProbeData>>,
    /// Incoming connections as (src_gid, src_tag, tgt_tag, weight, delay)
    pub incoming_connections: Map<usize, Vec<ConnectionData>>,
    /// Spiking threshold
    pub spike_threshold: f64,
    /// sparse map of gids to LIF cell descrption. Valid iff kind(gid) == LIF
    pub gid_to_lif: Map<usize, Map<String, f64>>,
    /// sparse map of gids to virtual cell spike trains. Valid iff kind(gid) == Virtual
    /// Will generate SpikeSource cells in Arbor
    pub gid_to_vrt: Map<usize, Vec<f64>>,
    /// dense map of gids to metadata
    pub metadata: Vec<CellMetaData>,
    /// cable cell global settings
    pub cable_cell_globals: Option<CableGlobalProperties>,
    /// cell counts by kind
    pub count_by_kind: [usize; 3],
}

const KIND_CABLE: u64 = 0;
const KIND_LIF: u64 = 1;
const KIND_SOURCE: u64 = 2;

const SYNAPSES: &[(&str, &str)] = &[("exp2syn", "Exp2Syn")];

fn fudge_synapse_dynamics(old: &str) -> String {
    if let Some(rep) = SYNAPSES.iter().find(|p| p.0 == old) {
        rep.1
    } else {
        old
    }
    .to_string()
}

impl Bundle {
    pub fn new(sim: &Simulation) -> Result<Self> {
        // Reverse lookup tables, used internally for uniqueness and index generation.
        let mut acc_to_cid = Map::new();
        let mut mrf_to_mid = Map::new();

        // Look up tables to write out
        let mut gid_to_meta = Vec::new();
        let mut cell_bio_ids = Map::new();
        let mut morphology = Vec::new();
        let mut decoration = Vec::new();
        let mut cell_kind = Vec::new();
        let mut incoming_connections = Map::new();
        let mut synapses = Map::new();
        let mut current_clamps = Map::new();
        let mut probes = Map::new();
        let mut gid_to_lif = Map::new();
        let mut gid_to_vrt = Map::new();
        let mut count_by_kind = [0; 3];
        for gid in 0..sim.size {
            let node = sim.reify_node(gid)?;
            gid_to_meta.push(CellMetaData::from(&node));
            if !node.incoming_edges.is_empty() {
                if matches!(node.node_type.model_type, ModelType::Biophysical { .. }) {
                    let mut inc = Vec::new();
                    let mut syn = Vec::new();
                    for (ix, edge) in node.incoming_edges.iter().enumerate() {
                        inc.push((
                            edge.src_gid as usize,
                            0, // in our SONATA model, there is _one_ source on each cell.
                            ix,
                            edge.weight,
                            edge.delay,
                        ));
                        let mech = edge.mech.as_ref().ok_or(anyhow!("Edge has no mechanism"))?;
                        let mech = fudge_synapse_dynamics(mech);
                        syn.push((
                            edge.target.0,
                            edge.target.1,
                            edge.target.2,
                            mech,
                            edge.dynamics.clone(),
                            ix,
                        ));
                    }
                    incoming_connections.insert(gid, inc);
                    synapses.insert(gid, syn);
                } else {
                    let inc = node
                        .incoming_edges
                        .iter()
                        .map(|e| {
                            (
                                e.src_gid as usize,
                                0, // in our SONATA model, there is _one_ source on each cell.
                                0,
                                e.weight,
                                e.delay,
                            )
                        })
                        .collect::<Vec<_>>();
                    incoming_connections.insert(gid, inc);
                };
            }
            match &node.node_type.model_type {
                ModelType::Biophysical {
                    model_template,
                    attributes,
                } => {
                    cell_kind.push(KIND_CABLE);
                    count_by_kind[KIND_CABLE as usize] += 1;
                    match model_template.as_ref() {
                        "ctdb:Biophys1.hoc" => {
                            let mid = if let Some(Attribute::String(mrf)) =
                                attributes.get("morphology")
                            {
                                if !mrf_to_mid.contains_key(mrf) {
                                    let mid = morphology.len();
                                    morphology.push(mrf.to_string());
                                    mrf_to_mid.insert(mrf.to_string(), mid);
                                }
                                mrf_to_mid[mrf]
                            } else {
                                bail!("GID {gid} is a biophysical cell, but has no morphology.");
                            };
                            let cid = if let Some(Attribute::String(fit)) =
                                attributes.get("dynamics_params")
                            {
                                let acc = fit;
                                if !acc_to_cid.contains_key(acc) {
                                    let cid = decoration.len();
                                    decoration.push(acc.to_string());
                                    acc_to_cid.insert(acc.to_string(), cid);
                                }
                                acc_to_cid[acc]
                            } else {
                                bail!(
                                    "GID {gid} is a biophysical cell, but has no dynamics_params."
                                );
                            };
                            cell_bio_ids.insert(gid, (mid, cid));
                        }
                        acc if acc.starts_with("nml:") => {
                            let mid = if let Some(Attribute::String(mrf)) =
                                attributes.get("morphology")
                            {
                                if !mrf_to_mid.contains_key(mrf) {
                                    let mid = morphology.len();
                                    morphology.push(mrf.to_string());
                                    mrf_to_mid.insert(mrf.to_string(), mid);
                                }
                                mrf_to_mid[mrf]
                            } else {
                                bail!("GID {gid} is a biophysical cell, but has no morphology.");
                            };

                            let acc = acc.strip_prefix("nml:").unwrap();
                            if !acc_to_cid.contains_key(acc) {
                                let cid = decoration.len();
                                decoration.push(acc.to_string());
                                acc_to_cid.insert(acc.to_string(), cid);
                            };
                            let cid = acc_to_cid[acc];
                            cell_bio_ids.insert(gid, (mid, cid));
                        }
                        t => bail!("Unknown model template <{t}> for gid {gid}"),
                    }
                }
                ModelType::Virtual { .. } => {
                    // The fields are largely irrelevant here
                    let data: &mut Vec<f64> = gid_to_vrt.entry(gid).or_default();
                    if let Some(group) = sim.virtual_spikes.get(&node.pop) {
                        if let Some(ts) = group.get(&node.node_id) {
                            data.append(&mut ts.clone());
                        }
                    }
                    cell_kind.push(KIND_SOURCE);
                    count_by_kind[KIND_SOURCE as usize] += 1;
                }
                ModelType::Point { model_template, .. } => {
                    cell_kind.push(KIND_LIF);
                    count_by_kind[KIND_LIF as usize] += 1;
                    match model_template.as_ref() {
                        "nrn:IntFire1" => {
                            // Taken from nrn/IntFire1.mod and adapted to Arbor.
                            let mut params = Map::from([
                                ("cm".to_string(), 1.0),
                                ("U_neutral".to_string(), 0.0),
                                ("U_reset".to_string(), 0.0),
                                ("U_th".to_string(), 1.0), // NOTE IntFire1 do be weird.
                                ("U_0".to_string(), 0.0),
                                ("t_ref".to_string(), 5.0),
                                ("tau".to_string(), 10.0),
                            ]);
                            for (k, v) in node.dynamics.iter() {
                                match k.as_ref() {
                                    "tau" => params.insert("tau".to_string(), *v),
                                    "refrac" => params.insert("t_ref".to_string(), *v),
                                    _ => bail!(
                                        "Unknown parameter <{k}> for template IntFire1 at gid {gid}"
                                    ),
                                };
                            }
                            gid_to_lif.insert(gid, params);
                        }
                        t => bail!("Unknown model template <{t}> for gid {gid}"),
                    }
                }
                mt => bail!("Cannot write ModelType {mt:?}"),
            }
        }

        for (gid, ics) in &sim.iclamps {
            let mut stim = Vec::new();
            for IClamp {
                amplitude_nA,
                delay_ms,
                duration_ms,
                tag,
                location,
            } in ics
            {
                stim.push((
                    location.clone(),
                    *delay_ms,
                    *duration_ms,
                    *amplitude_nA,
                    *tag,
                ));
            }
            current_clamps.insert(*gid as usize, stim);
        }

        for (gid, sim_probes) in &sim.reports {
            let mut prbs = Vec::new();
            for (ix, probe) in sim_probes.iter().enumerate() {
                match probe {
                    Probe::CableVoltage(ls) => {
                        for l in ls {
                            prbs.push((Some(l.clone()), "voltage".into(), ix));
                        }
                    }
                    Probe::Lif => {
                        prbs.push((None, "voltage".into(), ix));
                    }
                    Probe::CableIntConc(ion, ls) => {
                        for l in ls {
                            prbs.push((Some(l.clone()), ion.clone(), ix));
                        }
                    }
                    Probe::CableState(var, ls) => {
                        for l in ls {
                            prbs.push((Some(l.clone()), var.clone(), ix));
                        }
                    }
                    Probe::CableExtConc(ion, ls) => {
                        for l in ls {
                            prbs.push((Some(l.clone()), ion.clone(), ix));
                        }
                    }
                }
            }
            probes.insert(*gid as usize, prbs);
        }

        let max_cv_length = match &sim.cv_policy {
            CVPolicy::Default => None,
            CVPolicy::MaxExtent(l) => Some(*l),
        };

        let cable_cell_globals =
            if let Some(GlobalProperties { celsius, v_init }) = sim.global_properties {
                Some(CableGlobalProperties { celsius, v_init })
            } else {
                None
            };

        // Sort all spike sources. Just to be sure...
        gid_to_vrt.values_mut().for_each(|ts| {
            ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        });

        Ok(Bundle {
            time: sim.tfinal,
            time_step: sim.dt,
            max_cv_length,
            size: sim.size,
            cell_bio_ids,
            morphology,
            decoration,
            synapses,
            cable_cell_globals,
            probes,
            incoming_connections,
            cell_kind,
            count_by_kind,
            current_clamps,
            spike_threshold: sim.spike_threshold,
            gid_to_lif,
            gid_to_vrt,
            metadata: gid_to_meta,
        })
    }
}
