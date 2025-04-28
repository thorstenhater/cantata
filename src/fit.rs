use anyhow::bail;
use serde::{Deserialize, Deserializer, Serialize, de};
use serde_json::Value;

use crate::{
    Map,
    err::{Context, Result},
};

use std::{fs::File, path::PathBuf};

fn de_f64_or_string_as_f64<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    Ok(match Value::deserialize(deserializer)? {
        Value::String(s) => s.parse().map_err(de::Error::custom)?,
        Value::Number(num) => num
            .as_f64()
            .ok_or_else(|| de::Error::custom("Invalid number"))?,
        _ => return Err(de::Error::custom("wrong type")),
    })
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum Attribute {
    String(String),
    Float(f64),
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Conditions {
    pub celsius: Option<f64>,
    pub erev: Vec<RevPot>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct RevPot {
    pub section: String,
    #[serde(flatten)]
    pub values: Map<String, f64>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Section {
    pub section: String,
    #[serde(deserialize_with = "de_f64_or_string_as_f64")]
    pub value: f64,
    pub mechanism: String,
    pub name: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Cm {
    pub section: String,
    pub cm: f64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Passive {
    pub ra: Option<f64>,
    pub e_pas: Option<f64>,
    #[serde(default)]
    pub cm: Vec<Cm>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Fitting {}

#[derive(Debug, Deserialize, Serialize)]
pub struct AxonMorph {}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Fit {
    pub conditions: Vec<Conditions>,
    pub genome: Vec<Section>,
    pub passive: Vec<Passive>,
    #[serde(default)]
    pub fitting: Vec<Fitting>,
    #[serde(default)]
    pub axon_morph: Vec<AxonMorph>,
}

#[derive(Debug)]
pub struct Mechanism {
    pub name: String,
    pub parameters: Map<String, f64>,
    pub globals: Map<String, f64>,
}

impl Mechanism {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            parameters: Map::new(),
            globals: Map::new(),
        }
    }
}

#[derive(Debug, Default)]
pub struct Parameter {
    pub cm: Option<f64>,
    pub ra: Option<f64>,
    pub tk: Option<f64>,
}

#[derive(Debug)]
pub struct MechanismData {
    pub name: String,
    pub parameters: Map<String, f64>,
    pub globals: Map<String, f64>,
}

#[derive(Debug)]
pub struct Decor {
    pub mechanisms: Vec<(String, MechanismData)>,
    pub parameters: Vec<(String, Parameter)>,
    pub erev: Vec<(String, String, f64)>,
    pub defaults: Parameter,
}

impl Decor {
    pub fn to_acc(&self) -> Result<String> {
        let mut acc = String::new();
        acc.push_str(
            "(arbor-component
  (meta-data
    (version \"0.10-dev\"))
      (decor",
        );
        if let Some(v) = self.defaults.cm {
            acc.push_str(&format!(
                "\n        (default
          (membrane-capacitance {} (scalar 1.0)))",
                0.01 * v
            ));
        }
        if let Some(v) = self.defaults.ra {
            acc.push_str(&format!(
                "\n        (default
          (axial-resistivity {} (scalar 1.0)))",
                v
            ));
        }
        if let Some(v) = self.defaults.tk {
            acc.push_str(&format!(
                "\n        (default
          (temperature-kelvin {} (scalar 1.0)))",
                v
            ));
        }

        for (k, v) in &self.parameters {
            if let Some(v) = v.cm {
                acc.push_str(&format!(
                    "\n        (paint (region \"{}\")
          (membrane-capacitance {} (scalar 1.0)))",
                    k,
                    0.01 * v
                ));
            }
            if let Some(v) = v.ra {
                acc.push_str(&format!(
                    "\n        (paint (region \"{}\")
          (axial-resistivity {} (scalar 1.0)))",
                    k, v
                ));
            }
            if let Some(v) = v.tk {
                acc.push_str(&format!(
                    "\n        (paint (region \"{}\")
          (temperature-kelvin {} (scalar 1.0)))",
                    k, v
                ));
            }
        }
        for (reg, ion, erev) in &self.erev {
            acc.push_str(&format!(
                "\n        (paint (region \"{}\")
             (ion-reversal-potential \"{}\" {} (scalar 1.0)))",
                reg, ion, erev
            ));
        }
        for (
            reg,
            MechanismData {
                name,
                parameters,
                globals,
            },
        ) in self.mechanisms.iter()
        {
            // NOTE This is a nasty hack, but required for automatic Nernst
            // setting in the main script!
            if let Some(v) = parameters.get("gbar") {
                if *v == 0.0 {
                    continue;
                }
            }
            let mut mech = name.to_string();
            let mut sep = '/';
            for (k, v) in globals {
                mech = format!("{}{}{}={}", mech, sep, k, v);
                sep = ',';
            }
            acc.push_str(&format!(
                "\n        (paint (region \"{}\")
          (density
            (mechanism \"{}\"",
                reg, mech
            ));
            for (k, v) in parameters {
                acc.push_str(&format!("\n              (\"{}\" {})", k, v));
            }
            acc.push_str(")))");
        }
        acc.push_str("))\n");
        Ok(acc)
    }
}

impl Fit {
    pub fn from_file(path: &PathBuf) -> Result<Self> {
        let rd = File::open(path).with_context(|| format!("Opening {path:?}"))?;
        let fit =
            serde_json::de::from_reader(rd).with_context(|| format!("Parsing fit {path:?}"))?;
        Ok(fit)
    }

    pub fn decor(&self) -> Result<Decor> {
        let mut mec = Map::new();
        let mut par: Map<String, Parameter> = Map::new();
        let mut def = Parameter::default();
        let mut erev = Vec::new();

        for section in &self.genome {
            let mut mech = section.mechanism.to_string();
            if mech.is_empty() {
                mech.push_str("pas");
            }

            let region = section.section.to_string();

            // Parameter names must end in the mechanism name, eg K_mech, or be one
            // of a series of reserved names. NB. We cannot split on the underscore
            // as the mechanism name may contain underscores.
            if let Some(param) = section.name.strip_suffix(&format!("_{}", mech)) {
                mec.entry(region.to_string())
                    .or_insert_with(Map::new)
                    .entry(mech.to_string())
                    .or_insert_with(|| Mechanism::new(&mech))
                    .parameters
                    .insert(param.to_string(), section.value);
            } else if mech == "pas" {
                let v = par.entry(section.section.to_string()).or_default();
                match section.name.as_ref() {
                    "cm" | "Cm" => v.cm = Some(section.value),
                    "ra" | "Ra" => v.ra = Some(section.value),

                    x => bail!("Unexpected key {x}"),
                }
            } else {
                bail!(
                    "Section: parameter must end in mechanism name, or be empty *and* the name key must be one of cm, ra. Found mech={} and name={}",
                    mech,
                    section.name
                );
            }
        }

        for passive in &self.passive {
            for cm in &passive.cm {
                let sec = cm.section.to_string();
                par.entry(sec).or_default().cm = Some(cm.cm);
            }
            if let Some(ra) = passive.ra {
                def.ra = Some(ra);
            }

            if let Some(e_pas) = passive.e_pas {
                for mechs in mec.values_mut() {
                    mechs
                        .entry("pas".to_string())
                        .or_insert_with(|| Mechanism::new("pas"))
                        .globals
                        .insert("e".to_string(), e_pas);
                }
            }
        }

        for cond in &self.conditions {
            if let Some(celsius) = cond.celsius {
                def.tk = Some(celsius + 273.15);
            }
            for kvs in &cond.erev {
                let sec = kvs.section.to_string();
                for (key, value) in &kvs.values {
                    erev.push((sec.clone(), key.chars().skip(1).collect(), *value));
                }
            }
        }

        let mut mechanisms = Vec::new();
        for (reg, mut mechs) in mec.into_iter() {
            // Special treatment for pas/e= :(
            if let Some(pas) = mechs.get_mut("pas") {
                if pas.parameters.contains_key("e") {
                    pas.globals.insert("e".to_string(), pas.parameters["e"]);
                    pas.parameters.remove("e");
                }
            }

            for (mech, data) in mechs {
                mechanisms.push((
                    reg.clone(),
                    MechanismData {
                        name: mech.clone(),
                        parameters: data.parameters.clone(),
                        globals: data.globals.clone(),
                    },
                ));
            }
        }

        Ok(Decor {
            mechanisms,
            parameters: par.into_iter().collect(),
            erev,
            defaults: def,
        })
    }
}
