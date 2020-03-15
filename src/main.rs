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
use serde_json::value::{Map};
// python
use inline_python::python;

fn call_transater(txt: String) -> String {
    let c = inline_python::Context::new();
    python! { #![context = &c]
        import sys
        import os
        sys.path.append(f"{os.getcwd()}/universal_transformer")
        from translate import translate
        out = translate('txt)

    }
    let txt: String =  c.get_global("out").unwrap().unwrap();
    txt
}

fn main() {
    fn index(req: &mut Request) -> IronResult<Response> {
        let mut resp = Response::new();
        let mut data = Map::new();
        let params = req.get_ref::<Params>().unwrap();  // ?query=...を取得
        let query = match params.find(&["query"]) {
            Some(&Value::String(ref query)) => query.clone().to_string(),
            //Some(&Value::String(ref query)) => call_transater(query.clone().to_string()),
            _ => "".to_string(),
        };
        // todo: resultの定義: もしqueryが存在していたら翻訳結果を, 存在しなければNoneを
        // todo: loadingを早くする
        println!("{}", query);
        let mut result = "";
        if query != "" {
            result = "this is french sentence.";
        }
        data.insert("query".to_string(), to_json(&query.to_owned()));
        data.insert("result".to_string(), to_json(&result.to_owned()));
        resp.set_mut(Template::new("index", data)).set_mut(status::Ok);
        Ok(resp)
        //Ok(Response::with((status::Ok, "index.")))
    }

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
