use crate::err::Result;
use std::path::Path;

/// Parameter mapping from NML2 to Allen names. _Usually_ this is not needed as
/// nmlcc just produces whole catalogues.
const PARAMS: &[(&str, &str, &str)] = &[
    ("Im", "conductance", "gbar"),
    ("Ih", "conductance", "gbar"),
    ("Nap", "conductance", "gbar"),
    ("NaV", "conductance", "gbar"),
    ("Kv3_1", "conductance", "gbar"),
    ("NaTa", "conductance", "gbar"),
    ("Kd", "conductance", "gbar"),
    ("K_P", "conductance", "gbar"),
    ("Kv2like", "conductance", "gbar"),
    ("Ca_LVA", "conductance", "gbar"),
    ("Ca_HVA", "conductance", "gbar"),
    ("Im_v2", "conductance", "gbar"),
    ("NaTs", "conductance", "gbar"),
    ("SK", "conductance", "gbar"),
    ("K_T", "conductance", "gbar"),
    ("pas", "conductance", "g"),
    ("CaDynamics", "initialConcentration", "minCai"),
];

/// Even more unneeded in normal circumstances, but due to the ways of
/// serialization of decors it is --- sadly --- unavoidable.
const GLOBAL: &[(&str, &str)] = &[("Ih", "ehcn"), ("pas", "e")];

pub fn mk_acc(data: &Path) -> Result<String> {
    let data = data.to_str().unwrap().to_string();
    let mut lems = nml2::lems::file::LemsFile::core();
    nml2::get_runtime_types(&mut lems, &[data.clone()])?;
    let cells = nml2::acc::to_cell_list(
        &lems,
        &[data],
        &["k".to_string(), "ca".to_string(), "na".to_string()],
    )?;
    if cells.len() != 1 {
        anyhow::bail!(
            "Expected exactly one cell in NML file, got {}.",
            cells.len()
        );
    }
    let (_, cell) = cells.first_key_value().unwrap();
    let mut decor = cell.decor.clone();
    for it in decor.iter_mut() {
        if let nml2::acc::Decor::Paint(_, nml2::acc::Paintable::Mech(mech, params)) = it {
            for (m, p, q) in PARAMS.iter() {
                if m != mech {
                    continue;
                }
                if let Some(v) = params.remove(*p) {
                    params.insert(q.to_string(), v);
                }
            }
            let mut sep = '/';
            let mut tail = String::new();
            for (m, g) in GLOBAL.iter() {
                if m == mech {
                    if let Some(v) = params.remove(*g) {
                        tail = format!("{sep}{g}={v}");
                        sep = ',';
                    }
                }
            }
            mech.push_str(&tail);
        }
    }
    Ok(nml2::acc::Sexp::to_sexp(&decor))
}
