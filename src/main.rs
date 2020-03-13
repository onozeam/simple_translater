#![feature(proc_macro_hygiene)]
// web frame work
extern crate handlebars_iron as hbs;
use hbs::{DirectorySource, HandlebarsEngine, Template};
use hbs::handlebars::to_json;
use iron::prelude::*;
use iron::status;
use params::{Params, Value};
use router::Router;
// json
use serde::Serialize;
use serde_json::value::{Map};
// python
use inline_python::python;
//use std::io::Read;

#[derive(Serialize, Debug)]
pub struct Team {
    name: String,
    pts: u16,
}

fn call_transater(txt: String) {
    let c = inline_python::Context::new();
    python! { #![context = &c]
        import sys
        sys.path.append(".")
        from ml.translate import translate
        out = translate('txt)
    }
    let txt: String =  c.get_global("out").unwrap().unwrap();
    println!("{:?}", txt);
}

fn main() {
    call_transater("strawberry".to_string());
    return;
    let mut hbse = HandlebarsEngine::new();
    hbse.add(Box::new(DirectorySource::new("./templates/", ".hbs",)));
    if let Err(r) = hbse.reload() {
        panic!("{}", r);
    }
    let mut router = Router::new();
    router.get("/", index, "index");
    let mut chain = Chain::new(router);
    chain.link_after(hbse);

    println!("Server running at http://localhost:3000/");
    Iron::new(chain).http("localhost:3000").unwrap();

    fn index(req: &mut Request) -> IronResult<Response> {
        // # TODO
        // - python codeを読み込む. (学習済みモデルで予測する)
        let mut resp = Response::new();
        let mut data = Map::new();
        let params = req.get_ref::<Params>().unwrap();  // ?query=...を取得
        let query = match params.find(&["query"]) {
            Some(&Value::String(ref query)) => query.clone().to_string(),
            _ => return Ok(Response::with((status::BadRequest, "No data.\n"))),
        };
        println!("{}", query);
        let teams = vec![Team { name: query, pts: 43u16, }];
        data.insert("teams".to_string(), to_json(&teams));
        resp.set_mut(Template::new("index", data)).set_mut(status::Ok);
        Ok(resp)
        //Ok(Response::with((status::Ok, "index.")))
    }
}
