use crate::{Map, err::Result};
use anyhow::bail;
use std::path::Path;
use std::str::FromStr;

/// $key = value; used to resolve file paths
pub type Manifest = Map<String, String>;

pub type Components = Map<String, String>;

pub fn find_component(file: &str, components: &Map<String, String>) -> Result<std::path::PathBuf> {
    for pth in components.values() {
        let mut src = std::path::PathBuf::from_str(pth)?;
        src.push(file);
        if src.exists() {
            return Ok(src);
        }
        src.pop();
    }
    bail!("Couldn't find required resource {file}");
}

pub fn resolve_manifest(val: &mut String, manifest: &Manifest, base: &Path) -> Result<()> {
    // Recursively replace $key with values from manifest
    'a: loop {
        // Strip out {} to reduce ${key} to $key
        *val = val.replace(['{', '}'], "");
        for (k, v) in manifest {
            if val.contains(k) {
                *val = val.replace(k, v);
                continue 'a;
            }
        }
        if val.contains('$') {
            bail!("Unresolved marker: {val}; manifest={manifest:?}");
        }
        break;
    }
    // Replace './' with the top-level of the simulation file
    if val.starts_with("./") {
        *val = format!(
            "{}/{}",
            base.to_str().unwrap(),
            val.strip_prefix("./").unwrap()
        );
    }
    if !val.starts_with('/') {
        *val = format!("{}/{}", base.to_str().unwrap(), val);
    }
    Ok(())
}
