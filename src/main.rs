#![feature(proc_macro_hygiene)]
// web frame work
#![allow(dead_code, unused_imports)]
extern crate handlebars_iron as hbs;
use hbs::{DirectorySource, HandlebarsEngine, Template};
use hbs::handlebars::to_json;
use iron::prelude::*;
use iron::status;
use params::{Params, Value};
use router::Router;
#[cfg(feature = "watch")]
use hbs::Watchable;
use iron::prelude::*;
// other
use serde_json::value::{Map};
use inline_python::python;
use std::sync::Arc;
use lazy_static::lazy_static;

fn transate(sentence: String) -> String {
    python! { #![context = &PYC]
        import sys
        import os
        sys.path.append(f"{os.getcwd()}/universal_transformer")
        from translate import translate
        result = translate('sentence, model, src_field, tgt_field)
    }
    PYC.get_global("result").unwrap().unwrap()
}

pub fn index(req: &mut Request) -> IronResult<Response> {
    let mut resp = Response::new();
    let mut data = Map::new();
    let params = req.get_ref::<Params>().unwrap();  // ?query=...を取得
    let query = match params.find(&["query"]) {
        Some(&Value::String(ref query)) => query.clone().to_string(),
        _ => "".to_string(),
    };
    let mut result: String = "".to_string();
    if query != "" {
        result = transate(query.clone());
    }
    data.insert("query".to_string(), to_json(&query.to_owned()));
    data.insert("result".to_string(), to_json(&result.to_owned()));
    resp.set_mut(Template::new("index", data)).set_mut(status::Ok);
    Ok(resp)
}

lazy_static! {
    static ref PYC: inline_python::Context = {
        let c = inline_python::Context::new();
        python! { #![context = &c]
            import sys
            import os
            sys.path.append(f"{os.getcwd()}/universal_transformer")
            from translate import load_model_and_field
            model, src_field, tgt_field, max_seq_len = load_model_and_field()
        }
        c
    };
}


#[cfg(feature = "watch")]
fn main() {
    transate("start".to_string());
    let mut hbse = HandlebarsEngine::new();

    // add a directory source, all files with .hbs suffix will be loaded as template
    hbse.add(Box::new(DirectorySource::new(
        "./templates/",
        ".hbs",
    )));

    // load templates from all registered sources
    if let Err(r) = hbse.reload() {
        panic!("{}", r);
    }

    let mut router = Router::new();
    router.get("/", index, "index");
    let mut chain = Chain::new(router);
    chain.link_after(hbse);
    println!("Server running at http://localhost:3000/");
    Iron::new(chain).http("localhost:3000").unwrap();
}

#[cfg(not(feature = "watch"))]
fn main() {
    println!("Watch only enabled via --features watch option");
}
